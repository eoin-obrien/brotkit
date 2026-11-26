# Running the GPU check

brotkit's GPU tests can't run on hosted CI: it has no GPU, and numba-cuda-mlir can't compile
PTX without one ([#492](https://github.com/NVIDIA/numba-cuda-mlir/issues/492)). They run on
Xenon, but only when the maintainer starts them. There is no self-hosted runner. The result is
recorded as a commit status, and the hosted `gpu-gate` check reports on it. The design, and
what's deferred until outside contributors arrive, is in ADR-0010 (`docs/design/decisions.md`).

## When it's needed

`gpu-gate` runs on every PR. It says "GPU check not required" unless the PR touches something
listed in `tools/gpu_gate_paths.toml`:

- the backends (including the chunk planner), codegen and kernels packages;
- the gate, this check and the gpu-gate workflow itself;
- `conftest.py` files, test fixtures, and tests marked `gpu`;
- dependency changes that touch a GPU-stack package (numba-cuda-mlir, numba, llvmlite,
  `cuda-*`, `nvidia-*`, numpy, sympy, mpmath, gmpy2), `requires-python` or the `[cuda]` extra.
  Lockfile or pyproject changes the parser doesn't understand count too.

If the PR does touch one, it looks for a `gpu/xenon` status of `success` on the PR's
**current** head commit. A new push needs a new run. The gate reads its rules from the base
branch, so a PR can't loosen them for itself.

The gate is **advisory** (`enforce = false` in the config). Without that status it still
passes, with a warning annotation and a reminder in the job summary, so there's no red X.
Don't merge a GPU-relevant PR on a warning. Setting `enforce = true` makes it fail instead.

## Running it

```bash
nvidia-smi                                          # no LLM inference or other compute job?
uv run tools/gpu_check.py 123 --dry-run             # run everything, post nothing
uv run tools/gpu_check.py 123                       # run, post the status and comment, re-run gpu-gate
uv run tools/gpu_check.py 123 --perf                # also the timing tests, on an idle GPU
```

The target is a PR number or a full 40-character SHA. A SHA with no open PR gets a status but
no comment; use this for release commits. Per run, the script:

1. resolves the PR's head SHA and author, and stops if the author isn't a maintainer (see
   below);
2. fetches that SHA into a new, throwaway git worktree in a temp directory, and refuses if
   the path is in use, the tree is dirty, or the PR's head has moved;
3. for each of Python 3.14 and 3.15:
   - runs `uv sync --locked --extra cuda` into a temp venv;
   - records the GPU's utilisation (the median of 8 samples over 1.2 s);
   - runs `pytest -m "gpu and not perf"` with JUnit XML and a pytest-randomly seed. These
     correctness tests run straight away, whatever the desktop is doing;
   - with `--perf`, first waits until the GPU is idle (that median ≤ 5%, no compute process;
     otherwise it checks again every 20 s, up to `--max-wait`), then runs `pytest -m gpu`,
     timing tests included;
4. checks the head again; if it moved during the run, nothing is posted;
5. posts the `gpu/xenon` status, e.g. `3.14: 48 passed, 0 failed · 3.15: 48 passed, 0 failed ·
   RTX 3090 Ti sm_86 · driver 610.57`, and updates its one PR comment with:
   - the environment table (Python, numba-cuda-mlir, nvJitLink, NVVM, cuda-core and
     cuda-bindings versions; GPU, compute capability, driver);
   - the counts and the names of failed tests;
   - the seeds, the JUnit sha256 and the observed utilisation;
   - whether the timing tests ran;
   - the names of the allowlisted environment variables;
6. re-runs the PR's `gpu-gate` job so it sees the new status.

| Option | Meaning |
| --- | --- |
| `--dry-run` | Run everything, print the status and comment, post nothing. |
| `--python 3.15` | Run one version only, while iterating. A narrowed run doesn't post unless `--post-partial` is given, and its description then starts `partial: 3.15 only`. |
| `--seed N` | Use this pytest-randomly seed, e.g. to reproduce a failure from the comment. |
| `--perf` | Also run the timing tests (`@pytest.mark.perf`), after waiting for an idle GPU. |
| `--max-wait S` | With `--perf`: give up, posting nothing, if the GPU stays busy for S seconds (default 1800). |
| `--reviewed` | You have read the full diff of a PR from someone who isn't a maintainer. See below. |

The tests see an allowlisted environment and nothing else:

- PATH is rebuilt from the venv, uv and system directories, so a system CUDA toolkit on your
  PATH is ignored;
- HOME and TMPDIR point into the run's temp directory, so the kernel cache starts empty;
- `CUDA_VISIBLE_DEVICES` and uv's `UV_*` variables pass through;
- `BROTKIT_*`, `NUMBA_*` and other `CUDA_*` variables from your shell don't.

Nothing posted contains environment values, home paths, usernames or tokens. The script checks
this before posting and refuses if anything slips through.

## Timing tests

A GPU test that asserts on how long something takes is marked `@pytest.mark.perf` as well as
`gpu`. A loaded GPU makes those flaky, so they run only with `--perf`. Everything else marked
`gpu` must pass while browsers, chat apps and overlays are using the card, which on Xenon is
most of the time.

## PRs from other people

The script refuses a PR whose author doesn't have the admin or maintain role. Bots, Dependabot
included, never count as maintainers. To run one anyway, read the **full** diff, including
`uv.lock`, `pyproject.toml` (the build backend runs during `uv sync`) and every test file, then
run `uv run tools/gpu_check.py 123 --reviewed --dry-run` and again without `--dry-run`.
`--reviewed` is your statement that you have read it.

This runs the PR's code on Xenon as you, under the allowlisted environment above, so it never
sees your shell's tokens, but it can read your files. That's acceptable for a Dependabot bump
whose diff you've read. For a PR from a person, ADR-0010 ("Revisit when") has the plan: make
gpu-gate required, add a ruleset, and run reviewed PRs in a Docker sandbox with no network and
none of your credentials.

Coding agents never pass `--reviewed` and never run this script on other people's PRs
(AGENTS.md, "GPU check").

## Troubleshooting

- **"head moved"**: someone pushed during the run. Run it again.
- **"GPU still busy"** (only with `--perf`): something kept the GPU above 5% (LLM inference, a
  render, a browser playing video). Free it, raise `--max-wait`, or drop `--perf`.
- **gpu-gate still warns after posting**: the script re-runs the newest gpu-gate run for the
  SHA. If none had finished within 5 minutes, re-run it from the Actions tab.
- **A failure you can't reproduce**: the comment gives the seed. Run
  `uv run pytest -m gpu -p randomly --randomly-seed=<seed>` in a checkout of that SHA, or
  `uv run tools/gpu_check.py <PR> --seed <seed> --dry-run`. CPU CI's JUnit files are artifacts
  on each run, and its log header shows the seed.
