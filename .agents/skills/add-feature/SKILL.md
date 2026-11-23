---
name: add-feature
description: Add a kernel feature (an orbit accumulator or output layer such as Smooth, DistanceEstimate or StripeAverage) to brotkit's codegen. Use when adding or changing a Feature subclass, its LayerSpecs or its emitted snippets.
---

# Add a feature

Stub, to be filled in when the feature API lands (backlog 1.3, 1.12, 2.11).

1. Read "Formula pipeline", "Render once, colour many" and the `Feature` skeleton in
   `docs/design/spec.md`, plus `src/brotkit/codegen/AGENTS.md`.
2. Write a frozen `Feature` subclass with a unique `kind`, declared `requires` (never
   inheritance), `layers()` and `emit()` into the template slots.
3. Set `min_escape_radius(precision)` and the escape tests the feature is valid for.
4. Add tests: generated source snapshot, CPU result against a reference, JSON round-trip.
   Mark GPU parity tests `@pytest.mark.gpu`.
5. Don't add per-pixel Python or runtime dispatch.
