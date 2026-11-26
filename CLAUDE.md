@AGENTS.md

## Claude-only notes

- Use the repo skills in `.claude/skills/` (a symlink to `.agents/skills/`) for the tasks they
  cover: `add-feature`, `add-zoo-formula`, `update-golden-images`, `cut-release`,
  `run-gpu-suite`.
- Never run the GPU suite (`pytest -m gpu`) while LLM inference is using the GPU. Check
  `nvidia-smi` first, and ask if any other compute job holds the card. Desktop load is fine
  for correctness tests; timing tests (`perf`) need an idle GPU.
- Ask before updating golden images, every time. Approval for one update doesn't carry over
  to the next.
- gstack's `/build-check` reads this line (it doesn't follow the `@AGENTS.md` import):

gstack:verify: uv run pre-commit run -a && uv run pytest -m "not gpu"
