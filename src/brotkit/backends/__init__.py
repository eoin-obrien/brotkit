"""Compute backends.

`brotkit.backends.resources` is pure Python (no GPU stack) and safe to import anywhere.
`brotkit.backends.cuda` needs the `[cuda]` extra and is the only module in brotkit that imports
the GPU JIT (AGENTS.md invariant 8), so it is never imported from here.
"""
