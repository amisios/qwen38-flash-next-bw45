# Qwen3.8-Flash-Next on one 32 GB Blackwell GPU + 128 GB RAM

<p align="center"><strong>One 32 GB Blackwell GPU · 128 GB system RAM · 256K context · nothing read from disk while serving · vLLM · OpenAI-compatible API · docker compose</strong></p>

Run **Qwen3.8-Flash-Next** (a 125 GB mixture-of-experts checkpoint: INT4 experts, FP8 per-layer embedding table)
on a single 32 GB Blackwell GPU with 128 GB of RAM. Every dense part of the model and the most-used experts
live in VRAM; all other experts and the 51 GB embedding table are loaded into RAM at startup, and the **CPU
computes the cold experts straight from RAM** during decode. During prefill the cold experts are streamed through
the GPU, so long prompts stay fast. The checkpoint is read once at startup and never touched while serving.

> **Status:** the profile values are starting points derived from the original single-RTX-3090 runtime and the
> memory math in [docs/how-it-works.md](docs/how-it-works.md). Tune them once on the real machine
> ([docs/tuning.md](docs/tuning.md#first-start-tuning)); measured numbers will be added then.

## What you need

| Part | Needed |
|---|---|
| GPU | One NVIDIA Blackwell card with 32 GB (compute capability 12.0) |
| RAM | 128 GB: the serving container needs ~109 GiB (experts 50.8 GiB + embedding table 47.8 GiB + processes) |
| CPU | x86-64 with AVX2, FMA and F16C. Decode speed depends on memory bandwidth: a multi-channel workstation CPU is ideal |
| Disk | ~130 GB for the checkpoint, read only at startup (NVMe keeps startup short) |
| Software | Linux, Docker with the compose plugin, NVIDIA Container Toolkit, NVIDIA driver 580 or newer (the image uses CUDA 13.0) |

## Quick start

Download the checkpoint (about 129 GB). `hf` comes with `pip install -U huggingface_hub`.

```bash
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir /models/qwen38-flash-next
```

Configure, build once, then run with compose:

```bash
cp .env.example .env            # set MODEL_DIR; check COMPOSE_MEMORY_LIMIT (118g for 128 GB)
./scripts/preflight.sh          # optional: GPU, RAM vs. memory limit, driver, compose (or: make preflight)

docker build -f docker/Dockerfile -t qwen38-flash-next-bw45:local .
docker compose up -d
docker compose logs -f          # ready when the log shows the API server listening
```

The first start compiles GPU kernels (cached in `./jit-cache`) and loads ~100 GiB into RAM, so it takes several
minutes; later starts are faster. `docker compose ps` reports the container as healthy once `/health` answers.

```bash
python3 scripts/smoke_test.py   # a reasoning answer, a tool call, code
make bench                      # 4K, 32K, 131K and 258K prompts

curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Explain mixture-of-experts models in three sentences."}]
}'
```

Any OpenAI-compatible client works with the base URL `http://<server>:8000/v1`. Tool calls, reasoning output
(`reasoning_content`) and prefix caching are on.

## Where everything lives

The model has no dense-only layers: all 48 layers are mixture-of-experts layers (512 routed experts, 10 per token,
plus one shared expert). So "dense in VRAM, experts in RAM" means: every non-expert weight on the GPU, plus the
most frequently routed experts of each layer; the remaining experts in RAM.

| Where | What | Size |
|---|---|---:|
| GPU (32 GB) | Attention, GDN (linear attention), shared experts, router, embeddings, LM head | 9.8 GB |
| | 64 "hot" experts per layer (the most frequently routed ones) | 7.8 GB |
| | MTP draft layer (speculative decoding) | 1.6 GB |
| | INT8 KV cache for ~287K tokens (one 256K request + recurrent-state headroom) | 4.7 GB |
| | Prefill staging buffers, CUDA graphs, workspaces | ~5 GB |
| RAM (118 GiB container) | All 21,504 cold experts, pinned, on huge pages | 50.8 GiB |
| | The FP8 per-layer embedding (PLE) table, in the PLE worker process | 47.8 GiB |
| | Serving processes, CUDA host buffers | ~7 GiB |
| Disk | The checkpoint, read at startup only | 116 GiB |

The GPU estimate is about 28 GiB at the start value of 64 hot experts per layer; [tuning](docs/tuning.md) raises
the hot count until the measured peak leaves ~1 GiB free (likely ~84–88 per layer).

- **Decode:** per layer, the GPU runs the hot experts, a pinned CPU thread pool computes the cold ones from RAM
  (AVX2, FP32 accumulation), and the GPU computes a share of the cold experts itself over PCIe while it waits.
  MTP drafts 3 tokens per step.
- **Prefill:** 8,192-token chunks; each layer's cold experts are copied from RAM to the GPU in groups of 80,
  converted to the GPU kernel layout and computed there, overlapping the next copies.

Details: [how it works](docs/how-it-works.md) · [tuning and troubleshooting](docs/tuning.md) ·
[hardware](docs/hardware.md) · [FAQ](docs/faq.md).

## Network access

`compose.yaml` publishes the API on `${BIND_ADDRESS:-0.0.0.0}:${PORT:-8000}`, i.e. reachable from the network,
**without authentication** unless you set `VLLM_API_KEY` in `.env`. Ports published by Docker bypass host firewall
rules (ufw, nftables). Use this only on a trusted network, or set `BIND_ADDRESS=127.0.0.1`.

## If it does not start

- **`RAM-resident plan ... needs X GiB`**: the container memory limit cannot hold every expert and the embedding
  table. Set the `COMPOSE_MEMORY_LIMIT` the message names in `.env` (the server checks this in seconds, before it
  loads the checkpoint).
- **CUDA out of memory**: lower `QWEN38_HOT_ONLY` (each step of 1 frees ~116 MiB), see
  [GPU memory](docs/tuning.md#gpu-memory).
- **After changing the GPU or driver**: delete `./jit-cache` once so kernels are recompiled for the new setup.

## Limitations

- One request at a time (others queue), up to 262,144 tokens each.
- Not yet benchmarked on the target hardware; the sizes above are calculated, the speed is unmeasured.
- The INT8 KV cache was checked upstream with greedy-decoding agreement only (within run-to-run noise of BF16).
- Linux only.

## Credits

This runtime is derived from [qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090)
(single RTX 3090 + 64 GB) and builds on the release image of
[qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) (pinned vLLM build, Humming
kernels, Qwen3.8 runtime overlay), both Apache-2.0. The checkpoint is
[albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE).

## License

The runtime code is Apache-2.0 ([`LICENSE`](LICENSE)). The model keeps the upstream Qwen and third-party terms
listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
