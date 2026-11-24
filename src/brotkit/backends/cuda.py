"""CUDA backend. The only module in brotkit that imports the GPU JIT (AGENTS.md invariant 8).

Needs the `[cuda]` extra. Importing this module does not touch the driver. Compiling does:
numba-cuda-mlir 0.6.0's `compile_ptx` looks up the current device's compute capability even
when `cc` is explicit, so `compile_ptx` and `resource_usage` need a CUDA driver and a device
(spike S4, Q1). Everything after PTX (nvJitLink) runs without one.
"""

# numba-cuda-mlir and cuda-bindings ship no type information, and CPU CI type-checks this file
# without the [cuda] extra installed.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from brotkit.backends.resources import ResourceUsage, parse_nvjitlink_log

try:
    from cuda.bindings import nvjitlink
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.numba_cuda.core.sigutils import normalize_signature
except ImportError as e:  # pragma: no cover - exercised by the no-extra import test
    raise ImportError(
        "brotkit.backends.cuda needs the [cuda] extra: pip install 'brotkit[cuda]'"
    ) from e

__all__ = ["compile_ptx", "link_info", "resource_usage", "signature_args"]


def _py_func(kernel: Any) -> Callable[..., Any]:
    """Accept a `cuda.jit` dispatcher or a plain Python function."""
    return kernel.py_func if hasattr(kernel, "py_func") else kernel


def signature_args(signature: str | Sequence[Any]) -> tuple[Any, ...]:
    """Argument types of `signature`: a Numba signature string or a sequence of types.

    numba-cuda-mlir's `compile_ptx` rejects signature strings (it iterates them as types), so
    strings are parsed here. The return type, if any, must be void (kernels return nothing).
    """
    if isinstance(signature, str):
        args, _ = normalize_signature(signature)
        return tuple(args)
    return tuple(signature)


def compile_ptx(
    kernel: Any,
    signature: str | Sequence[Any],
    cc: tuple[int, int],
    *,
    fastmath: bool = False,
    launch_bounds: int | tuple[int, ...] | None = None,
) -> str:
    """PTX for `kernel` at `cc` (needs a device on numba-cuda-mlir 0.6.0, see module doc)."""
    ptx, _ = cuda.compile_ptx(
        _py_func(kernel),
        signature_args(signature),
        cc=cc,
        fastmath=fastmath,
        launch_bounds=launch_bounds,
    )
    return str(ptx).rstrip("\x00")  # compile_ptx output ends with a NUL (S3)


def link_info(ptx: str, cc: tuple[int, int], *, max_registers: int | None = None) -> str:
    """Link `ptx` to a cubin with nvJitLink and return its verbose info log.

    nvJitLink is what the dispatcher links with, so its whole-program numbers equal the
    driver's. Standalone `ptxas -v` reports the entry function alone and undercounts
    registers whenever the SASS calls subroutines such as the IEEE div/sqrt slow paths.
    """
    opts = [f"-arch=sm_{cc[0]}{cc[1]}", "-verbose", "-Xptxas=-v"]
    if max_registers is not None:
        opts.append(f"-maxrregcount={max_registers}")
    # nvJitLink caches ptxas results in the CUDA compute cache (~/.nv/ComputeCache) and, on a
    # hit, omits the ptxas block that carries the spill counts. CUDA_CACHE_DISABLE is read once
    # at load, so salt the PTX with a comment to force a fresh ptxas run (S4).
    ptx = f"{ptx}\n// brotkit resource_usage {uuid.uuid4().hex}\n"
    handle = nvjitlink.create(len(opts), opts)
    try:
        data = ptx.encode()
        nvjitlink.add_data(handle, nvjitlink.InputType.PTX, data, len(data), "kernel.ptx")
        nvjitlink.complete(handle)
        log = bytearray(nvjitlink.get_info_log_size(handle))
        nvjitlink.get_info_log(handle, log)
    finally:
        nvjitlink.destroy(handle)
    return log.decode(errors="replace").rstrip("\x00")


def resource_usage(
    kernel: Any,
    signature: str | Sequence[Any],
    cc: tuple[int, int],
    *,
    max_registers: int | None = None,
    launch_bounds: int | tuple[int, ...] | None = None,
    fastmath: bool = False,
) -> ResourceUsage:
    """Registers, spills, local and shared memory, and max block size of `kernel` at `cc`.

    Compiles to PTX for `cc` and links it with nvJitLink, as the dispatcher does, with the
    same `max_registers` cap (`cuda.jit(max_registers=...)` is applied at link time). `cc` can
    differ from the installed GPU's. All fields match the driver's attributes (S4: 175 of 175
    configurations on sm_86).
    """
    ptx = compile_ptx(kernel, signature, cc, fastmath=fastmath, launch_bounds=launch_bounds)
    log = link_info(ptx, cc, max_registers=max_registers)
    return parse_nvjitlink_log(log, cc, ptx=ptx, max_registers=max_registers)
