# SPDX-License-Identifier: Apache-2.0
"""Host plan for single-GPU serving: which CPUs run the cold-expert pool, how much RAM the expert arena gets, and
whether the profile's memory plan fits at all.

Standard library only. Imported by the GPU worker (cpu_experts.py, stream_v2.py, expert_store.py) and the PLE
worker; the container entrypoint runs this file directly to print the plan, and to stop with a readable error
before vLLM starts loading the checkpoint if the plan cannot work.

CPU pool (QWEN38_CPU_EXPERTS_CPUS=auto): one hardware thread on each physical core, except the first quarter of
the cores of every L3 domain (CCD/CCX on AMD, the whole die on most Intel desktops), which stay with the serving
processes. At most QWEN38_CPU_EXPERTS_MAX_THREADS (24) threads, spread evenly over the domains. Decode is bound by
memory bandwidth, and on AMD each CCD has its own link to memory, so the pool spans every CCD.

Serving processes (QWEN38_MAIN_CPUS=auto): every allowed CPU except the SMT siblings of the pool's cores. They may
use the pool's cores too; the pool sleeps during long prefill steps, when the serving processes need the cores.

Memory budget: the container's memory limit, or without one the installed RAM (MemTotal rounded up to a multiple
of 8 GiB) minus QWEN38_HOST_RESERVE_GIB (8 GiB).

PLE table: with QWEN38_PLE_MMAP != 1 the PLE worker keeps the whole FP8 table (51.3 GB) in RAM; with
QWEN38_PLE_MMAP=1 its rows are read from the checkpoint through the (evictable) page cache. In RAM mode the table
is subtracted from the budget before the arena is sized (QWEN38_PLE_RESIDENT_GIB overrides the size).

Expert arena (QWEN38_EXPERT_ARENA_SLOTS=auto): the budget minus the RAM-resident PLE table minus
QWEN38_SERVE_OVERHEAD_GIB (10 GiB for the processes, CUDA, pinned buffers and page cache), in 2,535,424-byte expert
slots, at most every cold expert. Cold experts that do not fit stay on disk (the "tail"), behind the page cache.

Checks: with QWEN38_REQUIRE_RAM_RESIDENT=1 every cold expert and the PLE table must fit in RAM (nothing is read
from disk while serving). In any case the per-layer tail must fit in one prefill stream group
(QWEN38_STREAM_GROUP), which stream_v2.py requires.
"""
from __future__ import annotations

import os
import sys

GIB = 1 << 30
SLOT_BYTES = 2_535_424          # expert_store.SLOT
LAYERS = 48                     # MoE layers
EXPERTS = 512                   # routed experts per layer
PLE_TABLE_BYTES = 51_300_000_000  # FP8 per-layer-embedding table of the checkpoint (docs/how-it-works.md)
STREAM_GROUP_DEFAULT = 128      # stream_v2.GROUP default (the profile sets QWEN38_STREAM_GROUP)
_SYS = "/sys"
_PROC = "/proc"
_CGROUP = "/sys/fs/cgroup"


class PlanError(RuntimeError):
    """The memory plan cannot work on this host; the message says what to change."""


def parse_cpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def format_cpus(cpus) -> str:
    cpus = sorted(set(cpus))
    runs: list[str] = []
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        runs.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(runs)


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _allowed() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def topology(allowed: list[int], sys_root: str = _SYS) -> list[dict]:
    """Physical cores among `allowed`: [{"threads": [cpu, ...], "domain": key}], ordered by first CPU id."""
    cores: dict[tuple, dict] = {}
    for cpu in allowed:
        base = f"{sys_root}/devices/system/cpu/cpu{cpu}"
        pkg = _read(f"{base}/topology/physical_package_id") or "0"
        siblings = _read(f"{base}/topology/thread_siblings_list") or str(cpu)
        domain = None
        for index in range(10):
            level = _read(f"{base}/cache/index{index}/level")
            if level is None:
                break
            if level == "3":
                domain = _read(f"{base}/cache/index{index}/shared_cpu_list")
                break
        core = cores.setdefault((pkg, siblings), {"threads": [], "domain": (pkg, domain or pkg)})
        core["threads"].append(cpu)
    return sorted(cores.values(), key=lambda c: c["threads"][0])


def auto_pool(allowed: list[int] | None = None, sys_root: str = _SYS, max_threads: int | None = None) -> list[int]:
    allowed = _allowed() if allowed is None else sorted(allowed)
    if max_threads is None:
        max_threads = int(os.environ.get("QWEN38_CPU_EXPERTS_MAX_THREADS", "24"))
    cores = topology(allowed, sys_root)
    domains: dict = {}
    for core in cores:
        domains.setdefault(core["domain"], []).append(core)
    per_domain = []
    for members in domains.values():
        keep = -(-len(members) // 4)            # ceil(n / 4) cores stay with the serving processes
        per_domain.append([c["threads"][0] for c in members[keep:]])
    pool: list[int] = []
    # round-robin over the domains so a thread cap still spans all of them
    for rank in range(max((len(p) for p in per_domain), default=0)):
        for members in per_domain:
            if rank < len(members) and len(pool) < max_threads:
                pool.append(members[rank])
    if not pool:                                # 1-3 cores: everything but the first core, or the only core
        firsts = [c["threads"][0] for c in cores]
        pool = firsts[1:] or firsts[:1]
    return sorted(pool)


def pool_cpus(sys_root: str = _SYS) -> list[int]:
    spec = os.environ.get("QWEN38_CPU_EXPERTS_CPUS", "auto").strip()
    if spec and spec != "auto":
        return parse_cpus(spec)
    return auto_pool(sys_root=sys_root)


def main_cpus(pool: list[int] | None = None, sys_root: str = _SYS) -> list[int]:
    spec = os.environ.get("QWEN38_MAIN_CPUS", "auto").strip()
    allowed = _allowed()
    if spec == "all":
        return allowed
    if spec and spec != "auto":
        return parse_cpus(spec)
    pool = set(pool_cpus(sys_root) if pool is None else pool)
    busy = set()
    for core in topology(allowed, sys_root):
        if core["threads"][0] in pool:
            busy.update(core["threads"][1:])    # the pool core's SMT siblings stay idle
    return [cpu for cpu in allowed if cpu not in busy]


def mem_total_bytes(proc_root: str = _PROC) -> int:
    text = _read(f"{proc_root}/meminfo") or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def cgroup_limit_bytes(cgroup_root: str = _CGROUP) -> int | None:
    for path in (f"{cgroup_root}/memory.max", f"{cgroup_root}/memory/memory.limit_in_bytes"):
        value = _read(path)
        if value and value != "max":
            try:
                limit = int(value)
            except ValueError:
                continue
            if limit < 1 << 60:                 # cgroup v1 reports "no limit" as a huge number
                return limit
    return None


def installed_bytes(mem_total: int) -> int:
    """Installed RAM estimated from MemTotal (the kernel keeps 1-2 GiB): rounded up to a multiple of 8 GiB."""
    step = 8 * GIB
    return -(-mem_total // step) * step


def memory_budget_bytes(proc_root: str = _PROC, cgroup_root: str = _CGROUP) -> tuple[int, str]:
    total = mem_total_bytes(proc_root)
    limit = cgroup_limit_bytes(cgroup_root)
    if limit is not None and limit < total:
        return limit, "container memory limit"
    reserve = float(os.environ.get("QWEN38_HOST_RESERVE_GIB", "8")) * GIB
    return int(min(total, installed_bytes(total) - reserve)), \
        f"{installed_bytes(total) / GIB:g} GiB installed - {reserve / GIB:g} GiB for the OS"


def ple_in_ram() -> bool:
    """Same rule as ple_layer.py: the table is memory-mapped only with QWEN38_PLE_MMAP=1."""
    return os.environ.get("QWEN38_PLE_MMAP") != "1"


def ple_resident_bytes() -> int:
    """RAM the PLE table occupies outside the arena: the whole table in RAM mode, nothing in mmap mode."""
    spec = os.environ.get("QWEN38_PLE_RESIDENT_GIB", "auto").strip()
    if spec and spec != "auto":
        return int(float(spec) * GIB)
    return PLE_TABLE_BYTES if ple_in_ram() else 0


def serve_overhead_bytes() -> int:
    return int(float(os.environ.get("QWEN38_SERVE_OVERHEAD_GIB", "10")) * GIB)


def hot_experts() -> int:
    return int(os.environ.get("QWEN38_HOT_ONLY", "64"))


def stream_group() -> int:
    return int(os.environ.get("QWEN38_STREAM_GROUP", str(STREAM_GROUP_DEFAULT)))


def arena_slots(total_cold: int, layers: int = LAYERS, proc_root: str = _PROC, cgroup_root: str = _CGROUP) -> int:
    """Number of cold experts kept in the RAM arena (the rest stay on disk), at least one per layer."""
    spec = os.environ.get("QWEN38_EXPERT_ARENA_SLOTS", "auto").strip()
    if spec and spec != "auto":
        want = int(spec)
    else:
        budget, _ = memory_budget_bytes(proc_root, cgroup_root)
        want = int((budget - ple_resident_bytes() - serve_overhead_bytes()) // SLOT_BYTES)
    return max(min(layers, total_cold), min(total_cold, want))


def needed_bytes(total_cold: int) -> int:
    """RAM for a plan that keeps every cold expert (and, in RAM mode, the PLE table) in memory."""
    return total_cold * SLOT_BYTES + ple_resident_bytes() + serve_overhead_bytes()


def _gib_up(nbytes: int) -> int:
    return -(-nbytes // GIB)


def check_plan(total_cold: int, slots: int, layers: int = LAYERS, proc_root: str = _PROC,
               cgroup_root: str = _CGROUP) -> None:
    """Raise PlanError if the plan cannot work, before vLLM spends minutes loading the checkpoint."""
    budget, source = memory_budget_bytes(proc_root, cgroup_root)
    tail = total_cold - slots
    if os.environ.get("QWEN38_REQUIRE_RAM_RESIDENT", "0") == "1":
        need = needed_bytes(total_cold)
        if tail > 0 or need > budget:
            mem_total = mem_total_bytes(proc_root)
            hint = (f"Set COMPOSE_MEMORY_LIMIT={_gib_up(need)}g or more in .env"
                    if need <= mem_total - 4 * GIB else
                    "This host has too little RAM for that: raise QWEN38_HOT_ONLY (fewer cold experts in RAM) "
                    "or use QWEN38_PLE_MMAP=1 (PLE rows read through the page cache)")
            raise PlanError(
                f"RAM-resident plan (QWEN38_REQUIRE_RAM_RESIDENT=1) needs {need / GIB:.1f} GiB: "
                f"{total_cold:,} cold experts ({total_cold * SLOT_BYTES / GIB:.1f} GiB)"
                f" + PLE table ({ple_resident_bytes() / GIB:.1f} GiB)"
                f" + overhead ({serve_overhead_bytes() / GIB:.1f} GiB), but the memory budget is "
                f"{budget / GIB:.1f} GiB ({source}; MemTotal {mem_total / GIB:.1f} GiB) and the arena holds "
                f"{slots:,} of {total_cold:,} cold experts. {hint}.")
    if tail > 0:
        per_layer_tail = total_cold // layers - slots // layers
        group = stream_group()
        if per_layer_tail > group:
            raise PlanError(
                f"{per_layer_tail} cold experts per layer would stay on disk, but prefill streams at most "
                f"QWEN38_STREAM_GROUP={group} of them per layer (memory budget {budget / GIB:.1f} GiB, {source}; "
                f"arena {slots:,} of {total_cold:,} cold experts). Raise the memory limit "
                f"(COMPOSE_MEMORY_LIMIT) or QWEN38_STREAM_GROUP.")


def describe(total_cold: int | None = None) -> str:
    if total_cold is None:
        total_cold = LAYERS * (EXPERTS - hot_experts())
    pool = pool_cpus()
    main = main_cpus(pool)
    budget, source = memory_budget_bytes()
    slots = arena_slots(total_cold)
    tail = total_cold - slots
    ple = (f"in RAM ({ple_resident_bytes() / GIB:.1f} GiB)" if ple_in_ram()
           else "memory-mapped from the checkpoint (page cache)")
    return "\n".join((
        f"CPU expert pool: {len(pool)} threads on CPUs {format_cpus(pool)}",
        f"Serving processes: CPUs {format_cpus(main)}",
        f"Memory budget: {budget / GIB:.1f} GiB ({source})",
        f"PLE table: {ple}",
        f"Expert arena: {slots:,} of {total_cold:,} cold experts in RAM ({slots * SLOT_BYTES / GIB:.1f} GiB), "
        f"{tail:,} on disk ({tail * SLOT_BYTES / GIB:.1f} GiB)",
        f"RAM needed with everything resident: {needed_bytes(total_cold) / GIB:.1f} GiB",
    ))


def main(argv: list[str]) -> int:
    total = LAYERS * (EXPERTS - hot_experts())
    try:
        slots = arena_slots(total)
        check_plan(total, slots)
    except PlanError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if len(argv) > 1 and argv[1] == "--env":
        pool = pool_cpus()
        print(f"QWEN38_CPU_EXPERTS_CPUS={format_cpus(pool)}")
        print(f"QWEN38_MAIN_CPUS={format_cpus(main_cpus(pool))}")
        print(f"QWEN38_EXPERT_ARENA_SLOTS={slots}")
    else:
        print(describe(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
