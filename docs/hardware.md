# Hardware: requirements, target machine and host setup

## Requirements

| Part | Needed | Notes |
|---|---|---|
| GPU | One NVIDIA Blackwell card with 32 GB | Compute capability 12.0 (sm_120), which the image is compiled for. The profile plans for nearly all of the 32 GB. |
| System RAM | 128 GB | The container needs ~109 GiB with the default profile: all cold experts (50.8 GiB) + PLE table (47.8 GiB) + processes. |
| CPU | x86-64 with AVX2, FMA and F16C | Decode speed depends on memory bandwidth: populate every memory channel. |
| Storage | ~130 GB for the checkpoint | Read once at startup; NVMe keeps startup short. Nothing is read while serving. |
| Software | Linux, Docker with the compose plugin, NVIDIA Container Toolkit, NVIDIA driver 580+ | The image uses CUDA 13.0. |

`./scripts/preflight.sh` (or `make preflight`) checks all of this, including whether `COMPOSE_MEMORY_LIMIT` in
`.env` can hold the profile.

## Target machine

| Part | Configuration |
|---|---|
| GPU | 1x NVIDIA Blackwell card, 32 GB GDDR7, PCIe 5.0 x16 |
| CPU | Workstation-class x86-64 CPU, several L3 domains (CCDs), multi-channel memory |
| RAM | 128 GB; serving container limited to 118 GiB |
| Storage | NVMe SSD for the checkpoint (startup only) |
| Software | Debian, NVIDIA driver 595 (CUDA 13 capable), Docker with compose |

## What to expect

- **Prefill** is bound by the GPU and by PCIe: every 8,192-token chunk streams all cold experts (~54 GB with 64 hot
  experts per layer) from RAM to the GPU, overlapped with the attention work. A PCIe 5.0 x16 slot roughly doubles
  the streaming bandwidth of PCIe 4.0; check the link with
  `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv`
  (the current generation drops at idle, so read it under load).
- **Decode** is bound by how fast the CPU reads the cold experts from RAM. A multi-channel workstation CPU with all
  channels populated does best; on CPUs with several CCDs the CPU pool spans all of them automatically. Every hot
  expert added on the GPU takes work off the CPU.
- **Startup** reads ~100 GiB from the checkpoint into RAM (expert arena and PLE table) and, on the first start,
  compiles GPU kernels (cached in `./jit-cache`). Expect several minutes from NVMe.

Measured numbers will be added after the first tuning run ([tuning](tuning.md#first-start-tuning)).

## Host setup notes (Debian)

- **NVIDIA Container Toolkit**: install it from NVIDIA's apt repository, then
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. It also works with a driver
  installed from NVIDIA's `.run` installer, as long as the driver libraries are registered with `ldconfig`.
  `docker info | grep -i nvidia` should list the runtime.
- **Compose plugin**: `docker compose version` must work (the `docker-compose-plugin` package from Docker's
  repository).
- **Transparent huge pages**: `cat /sys/kernel/mm/transparent_hugepage/enabled` should show `[madvise]` or
  `[always]`. The expert arena asks for huge pages; compacting memory before a start helps after long uptimes:
  `sync; echo 1 | sudo tee /proc/sys/vm/compact_memory`.
- **Kernel**: 5.14+ (MADV_POPULATE_WRITE), 6.1+ for huge-page collapse (Debian 12 and 13 qualify).
- **Firewall**: ports published by Docker bypass ufw/nftables input rules. The default compose file listens on all
  interfaces without an API key; restrict it with `BIND_ADDRESS=127.0.0.1` or set `VLLM_API_KEY` if the network is
  not fully trusted.
- **Other GPU users**: the profile plans for nearly the whole card. Keep the display (if any) and other CUDA
  programs off this GPU, or lower `QWEN38_HOT_ONLY`.

## Less RAM or another GPU

- **Less than ~112 GB of RAM**: set `QWEN38_PLE_MMAP=1` (PLE rows through the page cache, ~48 GiB less container
  memory). Below that, `QWEN38_REQUIRE_RAM_RESIDENT=0` lets cold experts stay on disk, as long as at most
  `QWEN38_STREAM_GROUP` of them per layer do (the server checks this at startup); decode then waits for disk reads.
- **24 GB GPUs**: lower `QWEN38_HOT_ONLY` to ~32 and `MAX_MODEL_LEN`/`KV_CACHE_MEMORY_BYTES` to 131,072 / 3.05 GB;
  the upstream single-RTX-3090 runtime was tuned for exactly that case.
