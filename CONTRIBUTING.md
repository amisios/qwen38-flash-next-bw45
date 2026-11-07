# Contributing

## Before you commit

GPU-free checks (standard-library Python and bash, no Docker needed):

```bash
make validate     # overlay manifest, pins, executables, credentials, private words
make test         # host-plan unit tests
bash -n scripts/*.sh
```

After changing files in `runtime/vllm-overlay/`, run `make manifest`: the image build verifies the checksums.

## Keep private data out

This repository is public. Never commit hostnames, user names, IP addresses, local paths, serial numbers or other
identifiers, and redact them from pasted logs and benchmark results. List such words, one per line, in an untracked
`.private-words` file at the repository root (it is git-ignored); `make validate` then fails if any of them appears
in a file that is or could be committed. `results/` (benchmark output) is git-ignored as well.

## Changes to the runtime

Say how you tested a change. For performance changes, compare the same `make bench` shapes before and after on the
same machine and include prefill and decode numbers. For numerical changes, include a correctness check against a
reference. Distinguish measured results from expectations.
