# FAQ

### Does the model have dense layers that could stay on the GPU while the MoE layers go to RAM?

No: all 48 decoder layers are mixture-of-experts layers (512 routed experts, 10 per token, plus one shared expert);
there are no dense-only layers at the start of the model. What is dense is everything except the routed experts:
attention, GDN (linear attention), the shared experts, router, embeddings, LM head and the MTP draft. All of that
lives in VRAM. In addition, the most frequently routed experts of every layer ("hot" experts) live in VRAM, and the
rest are kept in RAM and computed by the CPU during decode. One whole layer's experts would take ~1.2 GiB of VRAM
and remove only that layer's share of the CPU work; the same VRAM as hot experts covers the most frequently
routed experts of all 48 layers.

### Why is everything loaded into RAM?

So the server never waits for a disk: decode reads cold experts from RAM at memory bandwidth, prefill streams them
from pinned RAM to the GPU, and the per-layer embedding (PLE) table is looked up in RAM. The checkpoint is read
once at startup. The server refuses to start (`QWEN38_REQUIRE_RAM_RESIDENT=1`) if the memory limit cannot hold it
all, instead of silently falling back to disk reads.

### How much memory does it need?

About 28–31 GiB of VRAM (the profile grows the hot-expert count until ~1 GiB is left) and ~109 GiB of RAM for the
container with 64 hot experts per layer (50.8 GiB of experts, 47.8 GiB PLE table, ~10 GiB processes). On a 128 GB
machine the container limit is 118 GiB (`COMPOSE_MEMORY_LIMIT=118g`).

### How long can prompts be?

Up to 262,144 tokens per request (prompt plus output), the model's maximum. The INT8 KV cache holds one full
request plus the recurrent-state headroom that async prefill needs.

### Can it serve several users or parallel requests?

It serves one request at a time; further requests wait in the queue. The CPU decode path handles one sequence of
up to 4 tokens per step.

### Does it support tool calling and reasoning?

Yes: OpenAI-compatible `/v1/chat/completions` and `/v1/completions`, tool calls (`qwen3_coder` parser), reasoning
output (`qwen3` parser, `reasoning_content`) and prefix caching. Point any OpenAI-compatible client at
`http://<server>:8000/v1`.

### Is the output the same as on the original runtimes?

The weights, the INT4 expert math and the sparse attention are the same as in the upstream single-GPU and two-GPU
runtimes. The KV cache is INT8 per token and head; in a greedy-decoding comparison INT8 and BF16 differ no more
than two BF16 runs differ from each other.

### Why is the first start slow?

It compiles GPU kernels (cached in `./jit-cache` for later starts) and loads ~100 GiB into RAM. The first requests
also touch memory for the first time.

### Can other machines on the network use it?

Yes, by default: compose publishes the API on all interfaces (`BIND_ADDRESS=0.0.0.0`) without authentication. Set
`VLLM_API_KEY` in `.env` to require a bearer token, or `BIND_ADDRESS=127.0.0.1` to keep it local. Ports published
by Docker bypass host firewall rules.

### Does it work with other GPUs?

It is built for 32 GB Blackwell cards (sm_120) and has run on one so far. 24 GB cards need a smaller hot set and
context ([hardware](hardware.md#less-ram-or-another-gpu)).

### Does it run on Windows or WSL2?

It is only meant for Linux. WSL2 has different memory limits and huge-page behaviour.
