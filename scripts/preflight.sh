#!/usr/bin/env bash
# Check this host against the default profile (one 32 GB Blackwell GPU, 128 GB RAM, everything RAM-resident) before
# the first start. Errors stop `make up`. Settings come from the repository's .env (variables already set in the
# environment win), so `./scripts/preflight.sh` and `make preflight` see the same values.
set -euo pipefail

load_env() {  # KEY=value lines of a .env file; variables already set in the environment win
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[1]}
    value=${BASH_REMATCH[2]}
    value=${value%$'\r'}
    if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then value=${BASH_REMATCH[1]}; fi
    if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
  done < "$1"
}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$repo_root/.env" ]]; then load_env "$repo_root/.env"; fi

fail=0
warn() { printf 'WARN:  %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; fail=1; }
ok() { printf 'ok:    %s\n' "$*"; }
note() { printf 'note:  %s\n' "$*"; }

# Same numbers as runtime/vllm-overlay/qwen38_host.py (SLOT_BYTES, PLE_TABLE_BYTES, QWEN38_SERVE_OVERHEAD_GIB).
GIB=1073741824
SLOT_BYTES=2535424
PLE_TABLE_BYTES=51300000000
hot=${QWEN38_HOT_ONLY:-64}
overhead_gib=${QWEN38_SERVE_OVERHEAD_GIB:-10}
overhead_gib=${overhead_gib%%.*}

to_gib() {  # docker memory size (bytes, or with a b/k/m/g suffix) -> whole GiB, rounded down
  local v=${1,,}
  [[ "$v" =~ ^([0-9]+)([bkmg]?)$ ]] || return 1
  local n=${BASH_REMATCH[1]}
  case "${BASH_REMATCH[2]}" in
    g) echo "$n" ;;
    m) echo $(( n / 1024 )) ;;
    k) echo $(( n / 1048576 )) ;;
    *) echo $(( n / GIB )) ;;
  esac
}

[[ "$(uname -s)" == Linux ]] || error "serving requires Linux"
[[ "$(uname -m)" == x86_64 ]] || error "the CPU expert kernel needs an x86-64 CPU"
command -v nvidia-smi >/dev/null || error "nvidia-smi is not installed (NVIDIA driver)"
if ! command -v docker >/dev/null; then
  error "docker is not installed"
elif ! docker info >/dev/null 2>&1; then
  error "the Docker daemon is not reachable for this user (add the user to the docker group, or run with sudo)"
else
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    warn "docker info does not mention the NVIDIA runtime; install the NVIDIA Container Toolkit if the GPU reservation fails"
  fi
  if compose_version=$(docker compose version --short 2>/dev/null); then
    ok "docker compose $compose_version"
  else
    error "the docker compose plugin is missing"
  fi
fi

# CPU: the cold-expert kernel uses AVX2, FMA and F16C.
if [[ -r /proc/cpuinfo ]]; then
  flags=$(grep -m1 '^flags' /proc/cpuinfo)
  for f in avx2 fma f16c; do
    [[ " $flags " == *" $f "* ]] || error "CPU lacks $f"
  done
  ok "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//'), $(nproc) threads"
fi

# NVIDIA driver: the image uses CUDA 13.0 (driver 580 or newer).
if command -v nvidia-smi >/dev/null; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ -n "$driver" ]]; then
    if (( ${driver%%.*} >= 580 )); then ok "NVIDIA driver $driver"; else error "NVIDIA driver $driver is too old: CUDA 13.0 needs 580 or newer"; fi
  fi
fi

# GPU: one 32 GB Blackwell card (sm_120). The profile plans for nearly all of its memory.
gpu=${GPU_DEVICE:-0}
if command -v nvidia-smi >/dev/null; then
  line=$(nvidia-smi -i "$gpu" --query-gpu=name,memory.total,memory.used,compute_cap,pcie.link.gen.max,pcie.link.width.max \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "$line" ]]; then
    error "GPU $gpu not found (set GPU_DEVICE)"
  else
    IFS=',' read -r name total used cap pcie_gen pcie_width <<< "$line"
    name=${name# }; total=${total// /}; used=${used// /}; cap=${cap// /}; pcie_gen=${pcie_gen// /}; pcie_width=${pcie_width// /}
    ok "GPU $gpu: $name, ${total} MiB (${used} MiB in use), compute capability $cap, PCIe gen $pcie_gen x$pcie_width (max)"
    (( total >= 30000 )) || error "GPU $gpu has ${total} MiB; the profile needs a 32 GB card (docs/tuning.md: GPU memory)"
    [[ "$cap" == 12.* ]] || warn "compute capability $cap: this image is built and tuned for Blackwell sm_120"
    (( used <= 1000 )) || warn "${used} MiB of GPU $gpu is already in use (display or other programs); the profile plans for nearly all 32 GB"
    if [[ "$pcie_gen" =~ ^[0-9]+$ ]] && (( pcie_gen < 4 )); then
      warn "PCIe gen $pcie_gen: prefill streams the cold experts over PCIe, gen 4 or 5 x16 is recommended"
    fi
  fi
fi

# RAM: every cold expert and (with QWEN38_PLE_MMAP=0) the PLE table must fit in the container's memory limit.
if [[ -r /proc/meminfo ]]; then
  mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  mem_gib=$(( mem_kib / 1048576 )); avail_gib=$(( avail_kib / 1048576 ))
  ok "RAM: ${mem_gib} GiB total, ${avail_gib} GiB available"
  cold=$(( 48 * (512 - hot) ))
  ple_bytes=$PLE_TABLE_BYTES
  ple_mode="PLE table in RAM"
  if [[ "${QWEN38_PLE_MMAP:-0}" == 1 ]]; then ple_bytes=0; ple_mode="PLE table memory-mapped"; fi
  need_bytes=$(( cold * SLOT_BYTES + ple_bytes + overhead_gib * GIB ))
  need_gib=$(( (need_bytes + GIB - 1) / GIB ))
  ok "profile needs ~${need_gib} GiB: ${cold} cold experts ($(( cold * SLOT_BYTES / GIB )) GiB) + ${ple_mode} + ${overhead_gib} GiB overhead (QWEN38_HOT_ONLY=$hot)"
  if (( need_gib > mem_gib - 4 )); then
    error "this host has ${mem_gib} GiB of RAM, too little for a ${need_gib} GiB RAM-resident plan: raise QWEN38_HOT_ONLY or set QWEN38_PLE_MMAP=1"
  fi
  limit_spec=${COMPOSE_MEMORY_LIMIT:-118g}
  if ! limit_gib=$(to_gib "$limit_spec"); then
    error "COMPOSE_MEMORY_LIMIT=$limit_spec is not a docker memory size (e.g. 118g)"
  else
    suggest=$(( need_gib + 8 < mem_gib - 4 ? need_gib + 8 : mem_gib - 4 ))
    if (( limit_gib < need_gib )); then
      error "COMPOSE_MEMORY_LIMIT=$limit_spec is below the ~${need_gib} GiB the profile needs; set COMPOSE_MEMORY_LIMIT=${suggest}g in .env"
    else
      ok "container memory limit: COMPOSE_MEMORY_LIMIT=$limit_spec"
    fi
    (( limit_gib <= mem_gib - 4 )) || warn "COMPOSE_MEMORY_LIMIT=$limit_spec leaves less than 4 GiB of the ${mem_gib} GiB to the host"
  fi
  (( avail_gib >= need_gib )) || warn "only ${avail_gib} GiB is available now; the server will use ~${need_gib} GiB (close other programs first)"
fi

# Transparent huge pages: the expert arena asks for them (madvise is enough).
thp=/sys/kernel/mm/transparent_hugepage/enabled
if [[ -r $thp ]] && grep -q '\[never\]' "$thp"; then
  warn "transparent huge pages are disabled; decode is a little slower without them"
fi
kernel=$(uname -r)
IFS=. read -r kmaj kmin _ <<< "$kernel"
if (( kmaj < 5 || (kmaj == 5 && kmin < 14) )); then
  warn "kernel $kernel: 5.14+ recommended (MADV_POPULATE_WRITE), 6.1+ for huge-page collapse"
fi

# Checkpoint: read once at startup (everything is RAM-resident while serving), so storage only sets startup time.
if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    error "MODEL_DIR=$MODEL_DIR is not the downloaded checkpoint"
  else
    [[ -f "$MODEL_DIR/runtime/mtp-int4-g32/model.safetensors.index.json" ]] || error "MTP draft missing under $MODEL_DIR/runtime/mtp-int4-g32"
    src=$(df --output=source "$MODEL_DIR" 2>/dev/null | tail -1)
    dev=$(basename "$(readlink -f "$src")")
    parent=$(lsblk -no pkname "/dev/$dev" 2>/dev/null | head -1 || true)
    disk=${parent:-$dev}
    if [[ "$disk" == nvme* ]]; then
      ok "checkpoint on NVMe ($disk), read only at startup"
    elif [[ "$(cat "/sys/block/$disk/queue/rotational" 2>/dev/null || echo 0)" == 1 ]]; then
      warn "the checkpoint is on a spinning disk ($disk): it is read only at startup, but that start will be slow"
    else
      note "checkpoint on $disk (not NVMe): read only at startup, which may take longer"
    fi
  fi
else
  warn "MODEL_DIR is not set; skipped the checkpoint checks"
fi

# API exposure (informational).
bind=${BIND_ADDRESS:-0.0.0.0}
if [[ "$bind" != 127.0.0.1 && -z "${VLLM_API_KEY:-}" ]]; then
  note "the API will listen on $bind:${PORT:-8000} without a key (set VLLM_API_KEY, or BIND_ADDRESS=127.0.0.1)"
fi

(( fail == 0 )) || exit 1
printf 'preflight passed\n'
