"""CUDA backend. The only module in brotkit that imports the GPU JIT (AGENTS.md invariant 8).

Needs the `[cuda]` extra. Importing this module does not touch the driver. Compiling does:
numba-cuda-mlir 0.6.0's `compile_ptx` looks up the current device's compute capability even
when `cc` is explicit, so `compile_ptx` and `resource_usage` need a CUDA driver and a device
(spike S4, Q1). Everything after PTX (nvJitLink) runs without one.

Kernel cache adapter (spike S2, ADR-0008): `CudaCacheBackend` builds a kernel with Numba
(`compile_ptx`) and nvJitLink into a cubin, and loads a stored cubin with `cuda.core`, which
also launches it. Numba is used only to compile, and is imported only then, so a warm start
never imports it. Kernels use brotkit's raw-pointer ABI: arrays are passed as typed device
pointers (`float32*`) and every shape, stride and count is `int64`.
"""

# numba-cuda-mlir and cuda-bindings ship no type information, and CPU CI type-checks this file
# without the [cuda] extra installed.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from brotkit.backends.resources import ResourceUsage, parse_nvjitlink_log
from brotkit.kernels.cache import Build, CacheEntry, KernelSpec

try:
    import cuda.core as cuda_core
    import numpy as np  # comes with the [cuda] extra (numba-cuda-mlir); not a base dependency
    from cuda.bindings import nvjitlink, nvvm
    from cuda.core.utils import StridedMemoryView
except ImportError as e:  # pragma: no cover - exercised by the no-extra import test
    raise ImportError(
        "brotkit.backends.cuda needs the [cuda] extra: pip install 'brotkit[cuda]'"
    ) from e

__all__ = [
    "CudaCacheBackend",
    "CudaKernel",
    "compile_ptx",
    "link",
    "link_info",
    "parse_signature",
    "resource_usage",
    "signature_args",
]


@functools.cache
def _jit() -> Any:
    """numba_cuda_mlir.cuda, imported on first compile (about 0.5 s; spike S2)."""
    try:
        from numba_cuda_mlir import cuda
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "brotkit.backends.cuda needs the [cuda] extra: pip install 'brotkit[cuda]'"
        ) from e
    return cuda


def _py_func(kernel: Any) -> Callable[..., Any]:
    """Accept a `cuda.jit` dispatcher or a plain Python function."""
    return kernel.py_func if hasattr(kernel, "py_func") else kernel


def signature_args(signature: str | Sequence[Any]) -> tuple[Any, ...]:
    """Argument types of `signature`: a Numba signature string or a sequence of types.

    numba-cuda-mlir's `compile_ptx` rejects signature strings (it iterates them as types), so
    strings are parsed here. The return type, if any, must be void (kernels return nothing).
    """
    if isinstance(signature, str):
        from numba_cuda_mlir.numba_cuda.core.sigutils import normalize_signature

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
    ptx, _ = _jit().compile_ptx(
        _py_func(kernel),
        signature_args(signature),
        cc=cc,
        fastmath=fastmath,
        launch_bounds=launch_bounds,
    )
    return str(ptx).rstrip("\x00")  # compile_ptx output ends with a NUL (S3)


def link(
    ptx: str,
    cc: tuple[int, int],
    *,
    max_registers: int | None = None,
    ptxas_options: Sequence[str] = (),
    nvjitlink_options: Sequence[str] = (),
) -> tuple[bytes, str]:
    """Link `ptx` to a cubin with nvJitLink; return the cubin and the verbose info log.

    nvJitLink is what the dispatcher links with, so its whole-program numbers equal the
    driver's. Standalone `ptxas -v` reports the entry function alone and undercounts
    registers whenever the SASS calls subroutines such as the IEEE div/sqrt slow paths.
    One link gives both the cubin brotkit stores and the resource numbers stored with it.
    """
    opts = [f"-arch=sm_{cc[0]}{cc[1]}", "-verbose", "-Xptxas=-v"]
    if max_registers is not None:
        opts.append(f"-maxrregcount={max_registers}")
    opts += [f"-Xptxas={o}" for o in ptxas_options]
    opts += list(nvjitlink_options)
    # nvJitLink caches ptxas results in the CUDA compute cache (~/.nv/ComputeCache) and, on a
    # hit, omits the ptxas block that carries the spill counts. CUDA_CACHE_DISABLE is read once
    # at load, so salt the PTX with a comment to force a fresh ptxas run (S4). A comment
    # doesn't change the cubin.
    ptx = f"{ptx}\n// brotkit link {uuid.uuid4().hex}\n"
    handle = nvjitlink.create(len(opts), opts)
    try:
        data = ptx.encode()
        nvjitlink.add_data(handle, nvjitlink.InputType.PTX, data, len(data), "kernel.ptx")
        try:
            nvjitlink.complete(handle)
        except nvjitlink.nvJitLinkError as e:
            err = bytearray(nvjitlink.get_error_log_size(handle))
            nvjitlink.get_error_log(handle, err)
            raise RuntimeError(f"nvJitLink failed: {err.decode(errors='replace')}") from e
        cubin = bytearray(nvjitlink.get_linked_cubin_size(handle))
        nvjitlink.get_linked_cubin(handle, cubin)
        log = bytearray(nvjitlink.get_info_log_size(handle))
        nvjitlink.get_info_log(handle, log)
    finally:
        nvjitlink.destroy(handle)
    return bytes(cubin), log.decode(errors="replace").rstrip("\x00")


def link_info(ptx: str, cc: tuple[int, int], *, max_registers: int | None = None) -> str:
    """The nvJitLink verbose info log for `ptx` at `cc` (see `link`)."""
    return link(ptx, cc, max_registers=max_registers)[1]


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


# --- kernel cache adapter (spike S2, ADR-0008) -------------------------------------------------

_SCALARS = frozenset(
    {"float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_ENTRY = re.compile(r"\.visible\s+\.entry\s+(\S+)\s*\(([^)]*)\)", re.S)


def parse_signature(signature: Sequence[str]) -> tuple[Any, ...]:
    """brotkit ABI type names (`float32`, `int64`, `float32*`) as numba-cuda-mlir types."""
    from numba_cuda_mlir import types

    out: list[Any] = []
    for name in signature:
        base = name.removesuffix("*")
        if base not in _SCALARS:
            raise ValueError(f"unsupported kernel argument type {name!r}")
        t = getattr(types, base)
        cpointer = getattr(types, "CPointer")  # noqa: B009 - not in the stubs' __all__
        out.append(cpointer(t) if name.endswith("*") else t)
    return tuple(out)


def _entry_point(ptx: str) -> tuple[str, list[str]]:
    """The single kernel entry's mangled symbol and its PTX parameter types."""
    found = _ENTRY.findall(ptx)
    if len(found) != 1:
        raise RuntimeError(f"expected one .entry in the PTX, found {len(found)}")
    symbol, params = found[0]
    kinds = [m.group(1) for m in re.finditer(r"\.param\s+\.(\w+)", params)]
    return symbol, kinds


def _param_class(ptx_kind: str) -> tuple[str, int]:
    """PTX `.param` type as (float | int, bytes): `.f32` -> ("f", 4), `.u64`/`.b64` -> ("i", 8)."""
    m = re.fullmatch(r"([fubs])(\d+)", ptx_kind)
    if m is None:
        raise RuntimeError(f"unexpected PTX parameter type .{ptx_kind}")
    return ("f" if m.group(1) == "f" else "i", int(m.group(2)) // 8)


def _abi_class(name: str) -> tuple[str, int]:
    if name.endswith("*"):
        return ("i", 8)
    dt = np.dtype(name)
    return ("f" if dt.kind == "f" else "i", dt.itemsize)


def _device_cc(device: Any) -> tuple[int, int]:
    cc = device.compute_capability
    return (int(cc.major), int(cc.minor))


def _usage_from_json(d: object) -> ResourceUsage:
    if not isinstance(d, dict):
        raise RuntimeError("entry has no resource usage")
    r: dict[str, Any] = d  # pyright: ignore[reportUnknownVariableType]
    cap = r.get("max_registers")
    return ResourceUsage(
        cc=(int(r["cc"][0]), int(r["cc"][1])),
        registers=int(r["registers"]),
        spill_store_bytes=int(r["spill_store_bytes"]),
        spill_load_bytes=int(r["spill_load_bytes"]),
        stack_bytes=int(r["stack_bytes"]),
        static_shared_bytes=int(r["static_shared_bytes"]),
        max_threads_per_block=int(r["max_threads_per_block"]),
        max_registers=None if cap is None else int(cap),
    )


def _str_list(v: object) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise TypeError(f"expected a list of strings, got {v!r}")
    items: list[object] = v  # pyright: ignore[reportUnknownVariableType]
    return [str(o) for o in items]


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class CudaKernel:
    """A loaded kernel: a `cuda.core` kernel plus the launch signature stored with it.

    `launch` marshals arguments by the stored signature, never by inspecting the kernel:
    pointer parameters take an object with `__cuda_array_interface__` or DLPack (or a
    `cuda.core` Buffer, or a raw int address); scalars are cast to the parameter's dtype.
    Launches go on the backend's stream, which also orders foreign arrays (cuda.core's
    `StridedMemoryView` makes the producer's stream wait, per CAI v3 / DLPack).
    """

    __slots__ = ("_kernel", "backend", "digest", "resources", "signature", "symbol")

    def __init__(
        self,
        *,
        kernel: Any,
        backend: CudaCacheBackend,
        digest: str,
        symbol: str,
        signature: tuple[str, ...],
        resources: ResourceUsage,
    ) -> None:
        self._kernel = kernel
        self.backend = backend
        self.digest = digest
        self.symbol = symbol
        self.signature = signature
        self.resources = resources

    def _arg(self, kind: str, value: Any, stream_ptr: int) -> Any:
        if kind.endswith("*"):
            if isinstance(value, int):
                return np.uintp(value)
            if isinstance(value, cuda_core.Buffer):
                return value
            view = StridedMemoryView.from_any_interface(value, stream_ptr=stream_ptr)
            if not view.is_device_accessible:
                raise TypeError(f"{kind} argument isn't device-accessible memory")
            if view.readonly:
                raise TypeError(f"{kind} argument is read-only")
            want = np.dtype(kind[:-1])
            if view.dtype != want:
                raise TypeError(f"{kind} argument has dtype {view.dtype}, expected {want}")
            if view.strides is not None:
                expect = tuple(
                    int(np.prod(view.shape[i + 1 :], dtype=np.int64))
                    for i in range(len(view.shape))
                )
                if tuple(view.strides) != expect:
                    raise TypeError(f"{kind} argument must be C-contiguous")
            return np.uintp(view.ptr)
        return np.dtype(kind).type(value)

    def launch(self, grid: tuple[int, ...] | int, block: tuple[int, ...] | int, *args: Any) -> None:
        if len(args) != len(self.signature):
            raise TypeError(f"{self.symbol} takes {len(self.signature)} arguments, got {len(args)}")
        stream = self.backend.stream
        sp = int(stream.handle)
        marshalled = [self._arg(k, v, sp) for k, v in zip(self.signature, args, strict=True)]
        cuda_core.launch(
            stream, cuda_core.LaunchConfig(grid=grid, block=block), self._kernel, *marshalled
        )


class CudaCacheBackend:
    """`brotkit.kernels.cache.CacheBackend` for CUDA: Numba + nvJitLink build, cuda.core load.

    One per device. It owns the `cuda.core` stream every kernel launches on (ADR-0008).
    """

    __slots__ = ("_toolchain", "device", "stream")

    def __init__(self, device_id: int = 0) -> None:
        self.device = cuda_core.Device(device_id)
        self.device.set_current()
        self.stream = self.device.create_stream()
        self._toolchain: dict[str, str] | None = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def target(self) -> str:
        cc = _device_cc(self.device)
        return f"sm_{cc[0]}{cc[1]}"

    def toolchain(self) -> Mapping[str, str]:
        """Everything that can change the cubin for a given spec (ADR-0008). Driver and
        cuda-core versions are left out: a cubin for one cc loads on any driver that runs that
        cc, and cuda.core only loads it."""
        if self._toolchain is None:
            tc: dict[str, str] = {
                "numba-cuda-mlir": version("numba-cuda-mlir"),
                "libnvvm": ".".join(map(str, nvvm.version()))
                + " ir "
                + ".".join(map(str, nvvm.ir_version())),
                "libnvjitlink": ".".join(map(str, nvjitlink.version())),
                "numpy": ".".join(np.__version__.split(".")[:2]),
            }
            for dist in ("nvidia-nvvm", "nvidia-nvjitlink"):
                v = _dist_version(dist)
                if v is not None:
                    tc[dist] = v
            self._toolchain = tc
        return self._toolchain

    def build(self, spec: KernelSpec, source_path: Path) -> Build:
        cc = self._cc_of(spec.target)
        fn = self._import_entry(spec, source_path)
        lb = spec.option("launch_bounds")
        ptx, _ = _jit().compile_ptx(
            fn,
            parse_signature(spec.signature),
            cc=cc,
            fastmath=bool(spec.option("fastmath")),
            lineinfo=bool(spec.option("lineinfo")),
            debug=bool(spec.option("debug")),
            opt=bool(spec.option("opt")),
            launch_bounds=tuple(lb) if isinstance(lb, list) else lb,
        )
        ptx = str(ptx).rstrip("\x00")
        cap = spec.option("max_registers")
        max_registers = cap if isinstance(cap, int) else None
        cubin, log = link(
            ptx,
            cc,
            max_registers=max_registers,
            ptxas_options=_str_list(spec.option("ptxas_options")),
            nvjitlink_options=_str_list(spec.option("nvjitlink_options")),
        )
        usage = parse_nvjitlink_log(log, cc, ptx=ptx, max_registers=max_registers)
        symbol, kinds = _entry_point(ptx)
        if [_param_class(k) for k in kinds] != [_abi_class(t) for t in spec.signature]:
            raise RuntimeError(
                f"PTX parameters {kinds} don't match the signature {spec.signature} (ABI drift?)"
            )
        res = dataclasses.asdict(usage)
        res["cc"] = list(usage.cc)
        meta = {
            "symbol": symbol,
            "signature": list(spec.signature),
            "ptx_params": kinds,
            "resources": res,
            "loader": {
                "cuda-core": _dist_version("cuda-core"),
                "cuda-bindings": _dist_version("cuda-bindings"),
            },
        }
        return Build(
            blobs=(("cubin", cubin), ("ptx", ptx.encode())),
            meta_json=json.dumps(meta, sort_keys=True),
        )

    def load(self, entry: CacheEntry) -> CudaKernel:
        meta = entry.meta
        target = str(entry.key["target"])
        if target != self.target:
            raise RuntimeError(f"entry is for {target}, this device is {self.target}")
        usage = _usage_from_json(meta["resources"])
        symbol = str(meta["symbol"])
        sig = meta["signature"]
        if not isinstance(sig, list):
            raise RuntimeError("entry has no launch signature")
        obj = cuda_core.ObjectCode.from_cubin(entry.blob("cubin"))
        return CudaKernel(
            kernel=obj.get_kernel(symbol),
            backend=self,
            digest=entry.digest,
            symbol=symbol,
            signature=tuple(str(t) for t in sig),
            resources=usage,
        )

    def dev_dispatcher(self, spec: KernelSpec, source_path: Path) -> Any:
        """The dev path: a Numba dispatcher for the same source, signature and options.

        It compiles the same raw-pointer kernel the cache stores (one ABI, two launchers), for
        debugging and for the test that both paths agree. Pointer arguments are int addresses.
        Not cached: Numba's own `cache=True` is never used (ADR-0008).
        """
        fn = self._import_entry(spec, source_path)
        opts: dict[str, Any] = {}
        for name in ("fastmath", "lineinfo", "debug", "opt", "max_registers"):
            if any(n == name for n, _ in spec.options):
                opts[name] = spec.option(name)
        lb = spec.option("launch_bounds")
        if lb is not None:
            opts["launch_bounds"] = tuple(lb) if isinstance(lb, list) else lb
        ptxas = _str_list(spec.option("ptxas_options"))
        if ptxas:
            opts["ptxas_options"] = " ".join(ptxas)
        dispatcher = _jit().jit(**opts)(fn)
        dispatcher.compile(parse_signature(spec.signature))
        return dispatcher

    def _import_entry(self, spec: KernelSpec, source_path: Path) -> Callable[..., Any]:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(spec.source)
        name = f"brotkit_kernel_{source_path.stem}"
        modspec = importlib.util.spec_from_file_location(name, source_path)
        if modspec is None or modspec.loader is None:  # pragma: no cover
            raise ImportError(f"can't import generated kernel {source_path}")
        module = importlib.util.module_from_spec(modspec)
        sys.modules[name] = module
        try:
            modspec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return _py_func(getattr(module, spec.entry))

    def spec(
        self,
        *,
        source: str,
        entry: str,
        signature: Sequence[str],
        options: Mapping[str, object] | None = None,
    ) -> KernelSpec:
        """A `KernelSpec` for this device."""
        return KernelSpec.make(
            source=source,
            entry=entry,
            signature=tuple(signature),
            backend=self.name,
            target=self.target,
            options=options,
        )

    @staticmethod
    def _cc_of(target: str) -> tuple[int, int]:
        m = re.fullmatch(r"sm_(\d+)(\d)", target)
        if m is None:
            raise ValueError(f"bad CUDA target {target!r}")
        return (int(m.group(1)), int(m.group(2)))
