"""Smoke test: chunked == single launch, device iteration sums == host sum, full coverage."""

import pathlib
import sys

import numpy as np

# kernel C writes f layers 0-2 (mu, de, trap_d) and u layers 0-1 (n, trap_n) only
import s5lib

from brotkit.backends.chunks import ChunkPolicy

root = pathlib.Path(sys.argv[1])
gpu = s5lib.Gpu.open(root)
for tier, view, mi in (("f32", "seahorse", 1000), ("dd", "whole", 300)):
    w, h = (3840, 2160) if tier == "f32" else (960, 540)
    big = ChunkPolicy(
        target_s=0.5, ceiling_s=0.5, probe_pixels=w * h, max_growth=2.0, calib_budget_s=0.5
    )
    one = s5lib.render(gpu, tier, "C", view, w=w, h=h, max_iter=mi, policy=big, sentinel=True)
    f1, u1 = s5lib.host_layers(gpu, one)
    ch = s5lib.render(gpu, tier, "C", view, w=w, h=h, max_iter=mi, sentinel=True)
    f2, u2 = s5lib.host_layers(gpu, ch)
    cov = s5lib.coverage(ch)
    esc = u2[0] < mi  # mu and de are written only for escaped pixels
    print(
        tier,
        view,
        "single chunks",
        len(one.records),
        "| chunked",
        len(ch.records),
        "| identical f",
        bool(
            np.array_equal(f1[2], f2[2])
            and all(np.array_equal(f1[L][esc], f2[L][esc], equal_nan=True) for L in (0, 1))
        ),
        "u",
        np.array_equal(u1[:2], u2[:2]),
        "| coverage",
        cov.all(),
        "| iters dev",
        ch.iterations,
        "host",
        int(u2[0].astype(np.int64).sum()),
        "| overhead",
        round(ch.overhead, 4),
        "first",
        ch.summary()["calibration_chunks"],
        ch.summary()["calibration_s"],
    )
print("kernel lookups", gpu.lookups)
