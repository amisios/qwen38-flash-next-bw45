# How it works: a 125 GB MoE checkpoint on one 32 GB GPU and 128 GB of RAM

Qwen3.8-Flash-Next is a sparse mixture-of-experts model. All 48 decoder layers are MoE layers: each has 512
routed experts, of which every token uses 10, plus one shared expert. There are no dense-only layers. 36 layers
use GDN (linear attention) and 12 use full attention. The checkpoint
([albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE))
stores the routed experts in INT4 (62.8 GB), a 51.2B-parameter per-layer embedding (PLE) table in FP8 (51.3 GB),
and 9.8 GB of attention, recurrent (GDN) and other dense weights.

The runtime splits the model by how often each part is used, and keeps everything in memory while serving:

| Where | What | Size |
|---|---|---:|
| GPU (32 GB) | Attention, GDN, shared experts, router, embeddings, LM head | 9.8 GB |
| | 64 "hot" experts per layer (the most frequently routed ones; start value) | 7.8 GB |
| | MTP draft layer (it shares the embeddings with the target) | 1.6 GB |
| | INT8 KV cache: ~287K tokens | 4.7 GB |
| | Prefill staging buffers (~0.8 GB with groups of 80), CUDA graphs, workspace | ~5 GB |
| RAM (118 GiB container) | Expert arena: all 21,504 cold experts, huge pages, pinned | 50.8 GiB |
| | PLE table, in the PLE worker process | 47.8 GiB |
| | Serving processes, CUDA host buffers, page cache | ~7–10 GiB |
| Disk | The checkpoint: read once at startup | 116 GiB |

The GPU estimate (about 28 GiB with 64 hot experts per layer) extrapolates the measured 3090 profile peak
(23.4 GB with 32 hot experts and a 3.05 GB KV cache) by ~116 MiB per additional hot expert per layer and the
larger KV cache. Tuning raises the hot count to use the rest of the card. The model weights are not changed,
pruned or requantized: the GPU, the CPU and the prefill path all compute the same INT4 group-128 experts.

## Decode: the CPU computes the cold experts

A decode step verifies 4 tokens (1 + 3 MTP draft tokens). For each MoE layer:

1. The router picks 10 experts per token. Those among the layer's hot experts run on the GPU as usual.
2. The GPU writes the other token-expert pairs into a small request slot in pinned host memory.
3. A pool of CPU threads (one per physical core, pinned) computes those cold experts straight from the RAM arena:
   an AVX2/FMA kernel that reads the INT4 weights and F16 scales, with FP32 activations and accumulation.
4. Meanwhile the GPU computes a share of the cold experts itself (22% by default): a Triton kernel reads their INT4
   weights from the same pinned arena over PCIe instead of waiting idle.
5. The GPU waits for the CPU's partial sum and adds both parts to the layer output.

The request/response slots are polled by kernels, so the whole step still runs as one CUDA graph. The CPU part
reads roughly 2 GB of expert weights per step with 32 hot experts (less with more hot experts), so **decode speed
depends mostly on CPU memory bandwidth**. Every cold expert is in the arena, so decode never waits for a disk.

## Prefill: stream the cold experts to the GPU

Prompt chunks of 8,192 tokens use every expert of every layer, so the CPU would be far too slow. Instead, for each
layer:

1. The hot experts run from VRAM.
2. The cold experts are copied from the pinned arena to the GPU by DMA, 80 at a time (one contiguous range each),
   two groups ahead of the compute. A PCIe 4.0 x16 link moves ~25–27 GB/s; PCIe 5.0 roughly doubles that.
3. On the GPU each group is converted into the Humming kernel layout, byte-identical to what the model loader
   produces for the hot experts (checked at startup), and multiplied with the original indexed GEMM kernels.
4. All groups add into one FP32 accumulator, rounded to BF16 once at the end.

Small chunks (up to 384 tokens, such as the end of a prompt) run on the CPU path instead of streaming all experts.
With every cold expert in RAM there is no disk tier, so the occasional slow prefill steps that the original 64 GB
design saw (disk reads falling out of overlap with GPU work) cannot occur.

## KV cache and context

The attention KV cache is INT8 with one scale per token and KV head, dequantized inside the sparse attention (QSA)
kernel. 4.7 GB holds about 287K tokens: one 262,144-token request plus the recurrent-state blocks that async prefill
holds (the capacity must be at least 1.05x `MAX_MODEL_LEN`). The recurrent (GDN) state stays in FP32.

## PLE table in RAM

The 51.2B-parameter FP8 PLE table lives in a separate CPU worker process. With `QWEN38_PLE_MMAP=0` (default) the
worker loads the whole table into its memory at startup; each step it gathers only the rows its tokens need and
sends them to the GPU, which dequantizes them. Inside the container there is no swap (the memory and swap limits
are equal), so the table stays resident. `QWEN38_PLE_MMAP=1` instead memory-maps the checkpoint shards and reads
rows through the page cache (disk reads on first touch); it needs ~48 GiB less container memory.

## Startup checks

Before vLLM loads anything, the entrypoint resolves the host plan (`vllm/qwen38_host.py`): CPU pool, arena size,
and whether the memory limit can hold every cold expert plus the PLE table (`QWEN38_REQUIRE_RAM_RESIDENT=1`). If
not, it stops with the limit to set. After loading, the prefill path verifies its layout conversion against the
loaded hot experts byte for byte, and a hot-only layer that did not get the Humming backend stops the server
instead of silently skipping its cold experts.

## Numerical checks (from the original runtime)

- CPU kernel vs a NumPy FP32 dequantize-and-matmul reference: max relative error 1.5e-6.
- GPU share kernel vs an FP32 reference: 3.1e-4 relative error (FP16 dot products, FP32 accumulation).
- Prefill conversion vs the loader's conversion of the same experts: byte-identical.
- INT8 KV: greedy outputs drift from BF16 KV after a median of 13–29 tokens on 10 prompts, the same range as two
  runs of the same BF16 configuration, because the runtime is not bit-deterministic.
