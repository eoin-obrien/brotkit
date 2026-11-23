---
name: add-zoo-formula
description: Add a curated formula to the brotkit formula zoo with tuned defaults and a thumbnail test. Use when asked to add a named fractal (Multibrot, Burning Ship, Phoenix, Nova, McMullen, ...).
---

# Add a zoo formula

Stub, to be filled in with the zoo (backlog 2.9).

1. Check "Formula zoo" and "Additional colourings and formula zoo" in `docs/design/spec.md`.
2. Define the formula as SymPy over the registered symbols. Use only whitelisted nodes or
   `@register_function` functions, so it round-trips through the safe JSON tree.
3. Record tuned defaults (view, `max_iter`, escape test, critical point, algebra).
4. Add a small thumbnail test (CPU) and, if it earns one, a golden image. Adding a golden
   image needs explicit approval: use the `update-golden-images` skill.
