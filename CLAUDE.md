@AGENTS.md

## Claude-only notes

- Use the repo skills in `.claude/skills/` (a symlink to `.agents/skills/`) for the tasks they
  cover: `add-feature`, `add-zoo-formula`, `update-golden-images`, `cut-release`,
  `run-gpu-suite`.
- Never run the GPU suite (`pytest -m gpu`) while LLM inference is using the GPU. Check
  `nvidia-smi` first, and ask if anything else holds the card.
- Ask before updating golden images, every time. Approval for one update doesn't carry over
  to the next.
