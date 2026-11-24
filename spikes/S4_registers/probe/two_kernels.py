# Two kernels in one module, for the per-kernel vs per-module cap test and the cache-key test.
import math

from numba_cuda_mlir import cuda


def body(x, out):
    i = cuda.grid(1)
    if i < out.size:
        a = x[i]
        b = a * 1.5
        c = a + 2.0
        d = a * a
        e = math.sqrt(abs(a) + 1.0)
        acc = 0.0
        for _ in range(64):
            a = a * b + c
            b = b * c - d
            c = c * d + e
            d = d * e + a
            e = e * a + b
            acc += a + b + c + d + e
        out[i] = acc


k_capped = cuda.jit(max_registers=24)(body)
k_default = cuda.jit(body)
k_cached_cap = cuda.jit(cache=True, max_registers=24)(body)
k_cached_nocap = cuda.jit(cache=True)(body)
