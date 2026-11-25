"""Go/no-go probe: raw-pointer kernel ABI on public API (throwaway).

1. Numba compiles a kernel taking CPointer(float32) + int32 shape/pitch, rebuilding a 3D view
   with the public `cuda.carray`.
2. Numba launches it (dev path) with an int device pointer.
3. The same kernel's PTX (compile_ptx) is linked by nvJitLink into a cubin, loaded with
   cuda.core ObjectCode.from_cubin and launched with cuda.core.launch on a cuda.core Buffer
   and on a foreign array via __cuda_array_interface__ (cache path).
Prints the PTX .entry parameter list so the ABI is visible.
"""

from __future__ import annotations

import re

import numpy as np
from cuda.bindings import nvjitlink
from cuda.core import Device, LaunchConfig, ObjectCode, launch
from numba_cuda_mlir import cuda, types

L, H, W, PITCH = 3, 37, 50, 64


@cuda.jit
def kern(p_f, p_u, h, w, pitch, val):
    a = cuda.carray(p_f, (3, h, pitch), np.float32)
    b = cuda.carray(p_u, (3, h, pitch), np.uint32)
    x, y = cuda.grid(2)
    if x < w and y < h:
        for k in range(3):
            a[k, y, x] = val + np.float32(x) + np.float32(1000 * k)
            b[k, y, x] = np.uint32(y * 1000 + x + k)


SIG = (
    types.CPointer(types.float32),
    types.CPointer(types.uint32),
    types.int32,
    types.int32,
    types.int32,
    types.float32,
)


def expected() -> tuple[np.ndarray, np.ndarray]:
    ef = np.zeros((L, H, PITCH), np.float32)
    eu = np.zeros((L, H, PITCH), np.uint32)
    for k in range(L):
        ef[k, :, :W] = np.float32(0.5) + np.arange(W, dtype=np.float32) + np.float32(1000 * k)
        eu[k, :, :W] = (np.arange(H)[:, None] * 1000 + np.arange(W)[None, :] + k).astype(np.uint32)
    return ef, eu


def link_cubin(ptx: str, cc: tuple[int, int]) -> bytes:
    opts = [f"-arch=sm_{cc[0]}{cc[1]}"]
    h = nvjitlink.create(len(opts), opts)
    try:
        d = ptx.encode()
        nvjitlink.add_data(h, nvjitlink.InputType.PTX, d, len(d), "k.ptx")
        nvjitlink.complete(h)
        out = bytearray(nvjitlink.get_linked_cubin_size(h))
        nvjitlink.get_linked_cubin(h, out)
    finally:
        nvjitlink.destroy(h)
    return bytes(out)


def main() -> None:
    dev = Device(0)
    dev.set_current()
    cc = dev.compute_capability
    cc = (cc.major, cc.minor)
    ef, eu = expected()

    # --- dev path: Numba launch with int pointers -------------------------------------------
    df = cuda.to_device(np.zeros((L, H, PITCH), np.float32))
    du = cuda.to_device(np.zeros((L, H, PITCH), np.uint32))
    pf = df.__cuda_array_interface__["data"][0]
    pu = du.__cuda_array_interface__["data"][0]
    k = kern.compile(SIG) if hasattr(kern, "compile") else None
    print("dispatcher.compile:", type(k))
    grid = ((W + 15) // 16, (H + 15) // 16)
    try:
        kern[grid, (16, 16)](pf, pu, np.int32(H), np.int32(W), np.int32(PITCH), np.float32(0.5))
        cuda.synchronize()
        ok = np.array_equal(df.copy_to_host(), ef) and np.array_equal(du.copy_to_host(), eu)
        print("numba launch (int ptr):", "OK" if ok else "MISMATCH")
    except Exception as e:
        print("numba launch (int ptr) FAILED:", type(e).__name__, e)

    # --- cache path: compile_ptx -> nvJitLink -> cuda.core ----------------------------------
    ptx, _ = cuda.compile_ptx(kern.py_func, SIG, cc=cc)
    ptx = str(ptx).rstrip("\x00")
    m = re.search(r"\.visible \.entry (\S+)\(\s*(.*?)\)", ptx, re.S)
    assert m, "no entry"
    entry, params = m.group(1), m.group(2)
    print("entry:", entry)
    print("params:", [p.strip() for p in params.split(",")])
    cubin = link_cubin(ptx, cc)
    print("cubin bytes:", len(cubin))

    mod = ObjectCode.from_cubin(cubin)
    kfn = mod.get_kernel(entry)
    print("num_regs:", kfn.attributes.num_regs)
    s = dev.create_stream()
    cfg = LaunchConfig(grid=grid, block=(16, 16))

    # cuda.core Buffer
    nbf, nbu = ef.nbytes, eu.nbytes
    bf = dev.memory_resource.allocate(nbf, stream=s)
    bu = dev.memory_resource.allocate(nbu, stream=s)
    launch(s, cfg, kfn, bf, bu, np.int32(H), np.int32(W), np.int32(PITCH), np.float32(0.5))
    s.sync()
    hf = np.empty_like(ef)
    hu = np.empty_like(eu)
    # copy back via numba device array wrapping is avoided: use cuda.bindings memcpy
    from cuda.bindings import driver

    (err,) = driver.cuMemcpyDtoH(hf.ctypes.data, int(bf.handle), nbf)
    (err2,) = driver.cuMemcpyDtoH(hu.ctypes.data, int(bu.handle), nbu)
    assert err == 0
    assert err2 == 0
    print(
        "cuda.core launch (Buffer):",
        "OK" if np.array_equal(hf, ef) and np.array_equal(hu, eu) else "MISMATCH",
    )

    # foreign array via __cuda_array_interface__ (a Numba device array stands in for CuPy)
    df2 = cuda.to_device(np.zeros((L, H, PITCH), np.float32))
    du2 = cuda.to_device(np.zeros((L, H, PITCH), np.uint32))
    p2f = np.uintp(df2.__cuda_array_interface__["data"][0])
    p2u = np.uintp(du2.__cuda_array_interface__["data"][0])
    launch(s, cfg, kfn, p2f, p2u, np.int32(H), np.int32(W), np.int32(PITCH), np.float32(0.5))
    s.sync()
    ok = np.array_equal(df2.copy_to_host(), ef) and np.array_equal(du2.copy_to_host(), eu)
    print("cuda.core launch (CAI uintp):", "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
