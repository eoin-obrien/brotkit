"""`import brotkit` must work without the [cuda] extra (AGENTS.md invariant 8)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


# Block the GPU stack even when the [cuda] extra is installed: a None entry in sys.modules makes
# `import numba_cuda_mlir` raise ImportError.
BLOCK = """
import sys
for name in ("numba_cuda_mlir", "cuda", "cuda.bindings", "numba.cuda"):
    sys.modules[name] = None
"""


def test_import_brotkit_without_gpu_stack() -> None:
    r = run(
        BLOCK
        + textwrap.dedent(
            """
            import brotkit, brotkit.backends, brotkit.backends.resources, brotkit.codegen.checks
            print("ok", brotkit.__version__)
            """
        )
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")


def test_cuda_backend_says_which_extra_is_missing() -> None:
    r = run(BLOCK + "import brotkit.backends.cuda\n")
    assert r.returncode != 0
    assert "[cuda] extra" in r.stderr


def test_import_brotkit_does_not_load_the_gpu_jit() -> None:
    r = run(
        """
        import sys
        import brotkit, brotkit.backends.resources, brotkit.codegen.checks
        print(sorted(m for m in sys.modules if m.startswith(("numba_cuda_mlir", "cuda."))))
        """
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
