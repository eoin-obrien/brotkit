# Running the GPU check

brotkit's GPU tests can't run on hosted CI: it has no GPU, and numba-cuda-mlir can't compile
PTX without one ([#492](https://github.com/NVIDIA/numba-cuda-mlir/issues/492)). They run on
Xenon, but only when the maintainer starts them. There is no self-hosted runner. The result is
recorded as a commit status, and the hosted `gpu-gate` check enforces it. The design and its
limits are in ADR-0010 (`docs/design/decisions.md`).

## When it's needed

`gpu-gate` runs on every PR. It passes straight away with "GPU check not required" unless the
PR touches something listed in `tools/gpu_gate_paths.toml`:

- the backends (including the chunk planner), codegen and kernels packages;
- the gate, this check, the sandbox and the gpu-gate workflow itself;
- `conftest.py` files, test fixtures, and tests marked `gpu`;
- dependency changes that touch a GPU-stack package (numba-cuda-mlir, numba, llvmlite,
  `cuda-*`, `nvidia-*`, numpy, sympy, mpmath, gmpy2), `requires-python` or the `[cuda]` extra.
  Lockfile or pyproject changes the parser doesn't understand count too.

If the PR does touch one, the check stays red until a `gpu/xenon` status of `success` is on the
PR's **current** head commit. A new push needs a new run. The gate reads its rules from the
base branch, so a PR can't loosen them for itself.

## Running it

```bash
nvidia-smi                                          # is anything else using the GPU?
uv run tools/gpu_check.py 123 --dry-run             # run everything, post nothing
uv run tools/gpu_check.py 123                       # run, post the status and comment, re-run gpu-gate
```

The target is a PR number or a full 40-character SHA. A SHA with no open PR gets a status but
no comment; use this for release commits. Per run, the script:

1. resolves the PR's head SHA and author, and stops if the author isn't a maintainer (see
   below);
2. fetches that SHA into a new, throwaway git worktree in a temp directory, and refuses if
   the path is in use, the tree is dirty, or the PR's head has moved;
3. for each of Python 3.14 and 3.15:
   - runs `uv sync --locked --extra cuda` into a temp venv;
   - waits until the GPU is idle (the median of 8 utilisation samples over 1.2 s must be ≤ 5%,
     with no compute process; otherwise it checks again every 20 s, up to `--max-wait`);
   - runs `pytest -m gpu` with JUnit XML and a pytest-randomly seed;
4. checks the head again; if it moved during the run, nothing is posted;
5. posts the `gpu/xenon` status, e.g. `3.14: 48 passed, 0 failed · 3.15: 48 passed, 0 failed ·
   RTX 3090 Ti sm_86 · driver 610.57`, and updates its one PR comment with:
   - the environment table (Python, numba-cuda-mlir, nvJitLink, NVVM, cuda-core and
     cuda-bindings versions; GPU, compute capability, driver);
   - the counts and the names of failed tests;
   - the seeds and the JUnit sha256;
   - the names of the allowlisted environment variables;
6. re-runs the PR's `gpu-gate` job so it sees the new status.

| Option | Meaning |
| --- | --- |
| `--dry-run` | Run everything, print the status and comment, post nothing. |
| `--python 3.15` | Run one version only, while iterating. A narrowed run doesn't post unless `--post-partial` is given, and its description then starts `partial: 3.15 only`. |
| `--seed N` | Use this pytest-randomly seed, e.g. to reproduce a failure from the comment. |
| `--max-wait S` | Give up, posting nothing, if the GPU stays busy for S seconds (default 1800). |
| `--reviewed` | You have read the full diff of a PR from someone who isn't a maintainer. See below. |
| `--sandbox docker` | Run the suite in the sandbox container. Required with `--reviewed`. |

The tests see an allowlisted environment and nothing else:

- PATH is rebuilt from the venv, uv and system directories, so a system CUDA toolkit on your
  PATH is ignored;
- HOME and TMPDIR point into the run's temp directory, so the kernel cache starts empty;
- `CUDA_VISIBLE_DEVICES` and uv's `UV_*` variables pass through;
- `BROTKIT_*`, `NUMBA_*` and other `CUDA_*` variables from your shell don't.

Nothing posted contains environment values, home paths, usernames or tokens. The script checks
this before posting and refuses if anything slips through.

## PRs from other people

The script refuses a PR whose author doesn't have the admin or maintain role. Bots, Dependabot
included, never count as maintainers. To run one anyway:

1. Read the **full** diff, including `uv.lock`, `pyproject.toml` (the build backend runs during
   `uv sync`) and every test file. `--reviewed` is your statement that you have.
2. Build the sandbox image once (below).
3. `uv run tools/gpu_check.py 123 --reviewed --sandbox docker --dry-run`, then again without
   `--dry-run`.

In the sandbox the suite runs in two containers per Python version, on the WSL distro's own
Docker engine. A preflight refuses Docker Desktop.

- **Sync:** `uv sync` with network, the PR's source mounted read-only, and writes going only
  to that run's venv and uv cache.
- **Tests:** `pytest -m gpu` with `--network none`, the GPU, the venv mounted read-only, and
  one writable directory for the JUnit file.

Both run as your uid with all capabilities dropped, `no-new-privileges` and a read-only root
filesystem. Nothing else is mounted: no HOME, `~/.config/gh`, SSH keys, Docker socket or
`/mnt/c`. The gh token never enters the sandbox. The script reads the JUnit file as untrusted
input and posts from outside.

Coding agents never pass `--reviewed` and never run this script on other people's PRs
(AGENTS.md, "GPU check").

### Building the sandbox image

```bash
docker --context default info --format '{{.OperatingSystem}}'   # must not say Docker Desktop
docker --context default build -t brotkit-gpu-sandbox:1 tools/sandbox
docker --context default run --rm --gpus all brotkit-gpu-sandbox:1 nvidia-smi -L
```

The image is Ubuntu 24.04 plus uv and CPython 3.14 and 3.15, with base images pinned by digest.
Rebuild it when the Dockerfile changes.

## Troubleshooting

- **"head moved"**: someone pushed during the run. Run it again.
- **"GPU still busy"**: something else held the GPU (LLM inference, a render). Free it, or
  raise `--max-wait`.
- **gpu-gate is still red after posting**: the script re-runs the newest gpu-gate run for the
  SHA. If none had finished within 5 minutes, re-run it from the Actions tab.
- **A failure you can't reproduce**: the comment gives the seed. Run
  `uv run pytest -m gpu -p randomly --randomly-seed=<seed>` in a checkout of that SHA, or
  `uv run tools/gpu_check.py <PR> --seed <seed> --dry-run`. CPU CI's JUnit files are artifacts
  on each run, and its log header shows the seed.

<!-- throwaway: gpu-gate docs-only test, do not merge -->
