# Tuning and troubleshooting

The defaults live in [`configs/bw45-128gb-256k.env`](../configs/bw45-128gb-256k.env). Put overrides in `.env`:
compose passes the whole file into the container, where it wins over the profile. Change one thing at a time,
restart with `docker compose up -d` (compose recreates the container when `.env` changed) and compare with
`make bench`.

## First-start tuning

The profile starts with values that should fit with headroom. Tune them once on the real machine:

1. **Check the plan in the log.** `docker compose logs | grep qwen38-bw45` shows the CPU pool, the memory budget,
   `PLE table: in RAM`, and `Expert arena: 21,504 of 21,504 cold experts in RAM ..., 0 on disk`. The vLLM log shows
   `GPU KV cache size: N tokens`; N must be at least 275,251 for 262,144-token requests (raise
   `KV_CACHE_MEMORY_BYTES` otherwise).
2. **Measure the GPU peak.** Run `make bench` (it includes a ~258K-token request) while watching
   `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2`. Note the peak in MiB.
3. **Use the rest for hot experts.** Each hot expert per layer costs ~116 MiB. Keep ~1,000 MiB free:
   `QWEN38_HOT_ONLY = 64 + (memory.total - 1000 - peak) / 116`, rounded down. Set it in `.env`, restart, and repeat
   step 2 once. More hot experts mean fewer cold experts for the CPU per decode step (faster decode) and slightly
   less RAM.
4. **Balance decode.** Set `QWEN38_CPU_EXPERTS_STATS=64` and generate a few hundred tokens. If the logged `cpu` time
   is well above `between layers` (GPU), raise `QWEN38_GPU_SHARE_FRAC` a little (0.22 -> 0.28 -> 0.34); if the GPU
   time grows past the CPU time, lower it. A PCIe 5.0 slot lets the GPU take a larger share.
5. Write the final values into the profile and `repro.lock.json` (or keep them in `.env`).

Optional experiments, each measured with `make bench` before and after:

- `QWEN38_GPU_SHARE_PROTECT=<512 - hot>`: by default only the 200 best-ranked cold experts per layer are eligible
  for the GPU share; with every expert in RAM nothing is ever evicted, so all of them can be (untested).
- `QWEN38_GPU_SHARE_KMAX=16` (power of 2): more cold experts per layer for the GPU share.
- `MAX_NUM_BATCHED_TOKENS=16384`: larger prefill chunks amortize per-chunk conversion and DMA waits; costs VRAM.
- `QWEN38_CPU_EXPERTS_MAX_THREADS=28`: a larger CPU pool (fewer cores left to the serving processes).

## Memory

| Setting | Default | Meaning |
|---|---|---|
| `COMPOSE_MEMORY_LIMIT` | `118g` | Memory limit of the whole container (also its swap limit, so nothing is swapped). It must hold every cold expert, the PLE table and ~10 GiB for the processes. |
| `QWEN38_REQUIRE_RAM_RESIDENT` | `1` | Stop at startup, with the limit to set, if anything would have to be read from disk while serving. |
| `QWEN38_PLE_MMAP` | `0` | `0`: PLE table loaded into RAM (47.8 GiB). `1`: rows read from the checkpoint through the page cache. |
| `QWEN38_EXPERT_ARENA_SLOTS` | `auto` | Cold experts kept in RAM. `auto` = (limit - PLE table - 10 GiB) / 2.42 MiB, at most all of them. |
| `QWEN38_SERVE_OVERHEAD_GIB` | `10` | RAM that `auto` leaves for the processes, CUDA host buffers and page cache. |

RAM needed with everything resident, by hot experts per layer: 64 -> 108.6 GiB, 80 -> 106.7 GiB, 88 -> 105.8 GiB,
96 -> 104.9 GiB (cold experts + PLE table + 10 GiB). `make preflight` prints the number for your settings.

The arena asks for transparent huge pages (`madvise` mode is enough). On a freshly booted machine all of it gets
huge pages; after days of uptime memory can be fragmented. The startup line `Expert store: ... huge pages X GiB`
shows how much you got. `sync; echo 1 | sudo tee /proc/sys/vm/compact_memory` before starting helps.

## GPU memory

The profile plans for nearly the whole 32 GB. If the server runs out of GPU memory (at startup or during the first
long prompt), free VRAM in this order:

1. Lower `QWEN38_HOT_ONLY` (each step of 1 frees ~116 MiB). Fewer hot experts mean more work for the CPU per decode
   step.
2. A shorter context: `MAX_MODEL_LEN=131072` with `KV_CACHE_MEMORY_BYTES=3050000000` frees ~1.6 GB (3.05 GB holds
   141,504 tokens). Keep the logged `GPU KV cache size` at least 1.05x `MAX_MODEL_LEN`.
3. `QWEN38_STREAM_GROUP=64` (from 80): ~155 MiB less staging memory, slightly slower prefill.

If another program (or a display) uses the GPU, check `nvidia-smi` first and leave that memory free.

## CPU

| Setting | Default | Meaning |
|---|---|---|
| `QWEN38_CPU_EXPERTS_CPUS` | `auto` | CPUs of the cold-expert pool (one thread each). `auto`: one thread per physical core except the first quarter of the cores of every L3 domain (CCD), at most 24. |
| `QWEN38_CPU_EXPERTS_MAX_THREADS` | `24` | Cap for `auto`. |
| `QWEN38_MAIN_CPUS` | `auto` | CPUs of the serving processes: everything except the SMT siblings of the pool's cores. |
| `QWEN38_PLE_THREADS` | `4` | CPU threads for the embedding-table lookups. More threads only spin and take cores from the GPU worker. |

The container prints its plan at startup, for example on a 16-core CPU with two L3 domains:

```text
[qwen38-bw45] CPU expert pool: 12 threads on CPUs 2-7,10-15
[qwen38-bw45] Serving processes: CPUs 0-17,24-25
```

Decode is limited by how fast the pool can read the cold experts. On CPUs with several CCDs each CCD has its own
link to memory, so the pool spans every CCD (`auto` does). Populate every memory channel and keep other
memory-heavy programs idle while generating.

## Decode

| Setting | Default | Meaning |
|---|---|---|
| `QWEN38_GPU_SHARE_FRAC` | `0.22` | Share of each layer's cold experts the GPU computes over PCIe while the CPU works. `0` turns it off. |
| `MTP_DEPTH` | `3` | Draft tokens per step (0-3). The CPU path handles at most 4 tokens per step. |
| `QWEN38_CPU_EXPERTS_STATS` | `0` | Log decode step timings every N steps: CPU ms, GPU gap, experts per step. |

With `QWEN38_CPU_EXPERTS_STATS=64` the server logs a line like this every 64 decode steps:

```text
cpu_moe decode steps=64: per step cpu 32.80 ms, between layers 15.83 ms, step tail 4.63 ms (48.0 requests/step),
cold experts 772/step of which tail 0.0; promoted 0, skipped 0 (total)
```

`cpu` is the CPU pool's time per step and `between layers` the GPU's. `of which tail` counts experts read from disk
and must stay 0 with this profile.

## Prefill

| Setting | Default | Meaning |
|---|---|---|
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill chunk size. |
| `QWEN38_STREAM_GROUP` | `80` | Cold experts per DMA group (~9.7 MiB of VRAM each). |
| `QWEN38_CPU_PREFILL_MAX` | `384` | Chunks up to this many tokens run on the CPU instead of streaming every expert. |
| `QWEN38_STREAM_PROFILE` | `0` | `1` logs a per-step breakdown (DMA stalls, conversion, GEMMs). |

The first request after a start is slower (kernel compilation and first-touch page faults).

## Other

| Setting | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | Tokens per request (prompt + output); the model's maximum. |
| `MAX_NUM_SEQS` | `1` | Must stay 1: requests are served one at a time and queue in the server. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `3200` | Prefix-cache checkpoint interval; a multiple of 3,200 with the INT8 KV cache. |
| `KV_CACHE_DTYPE` | `int8_per_token_head` | `auto` gives BF16 KV (half the tokens per GB). |
| `BIND_ADDRESS` / `PORT` | `0.0.0.0` / `8000` | Where compose publishes the API. |
| `VLLM_API_KEY` | unset | Require this bearer token on the API. |
| `GPU_DEVICE` | `0` | GPU index (`nvidia-smi`). |

## Common problems

- **`docker: could not select device driver "" with capabilities: [[gpu]]`**: install the NVIDIA Container
  Toolkit and restart Docker.
- **`RAM-resident plan ... needs X GiB`** at startup: raise `COMPOSE_MEMORY_LIMIT` to the value in the message
  (or raise `QWEN38_HOT_ONLY`, which moves experts from RAM to VRAM).
- **`... cold experts per layer would stay on disk, but prefill streams at most ...`**: only possible with
  `QWEN38_REQUIRE_RAM_RESIDENT=0` and a small memory limit; raise the limit.
- **The container is killed (out of memory) while loading**: the limit is larger than what is actually free on the
  host. Stop other programs or lower `COMPOSE_MEMORY_LIMIT` (and raise `QWEN38_HOT_ONLY` or use
  `QWEN38_PLE_MMAP=1` so the plan still fits).
- **CUDA out of memory**: see [GPU memory](#gpu-memory).
- **Kernel errors after a GPU or driver change**: delete `./jit-cache` and start again.
- **Illegal instruction**: the CPU lacks AVX2/FMA/F16C.
- **Very slow decode**: check that every memory channel is populated and running at its rated speed, that nothing
  else uses the memory bandwidth, and that `QWEN38_CPU_EXPERTS_STATS=64` shows `of which tail 0.0`.
- **The container restarts in a loop**: `restart: unless-stopped` retries failed starts; read the first error in
  `docker compose logs`, fix it, then `docker compose up -d`.
