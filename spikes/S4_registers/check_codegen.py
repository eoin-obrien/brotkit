"""Run brotkit.codegen.checks.find_bare_float_ops over every generated S4 kernel module.

Multi-word tiers (whole module: device library + kernel) must have zero findings. Controls:
plain f32/f64 (bare by design) and a dd build with the _rn rewrite switched off must be flagged.
"""

import json
import pathlib

import gen

from brotkit.codegen.checks import find_bare_float_ops

gen.TIERS["dd_naive"] = ("float64", 2, False)  # control: multi-word without the rewrite
gen.TIERS["qf_naive"] = ("float32", 4, False)
rows = {}
for tier in [*gen.MULTIWORD, "f32", "f64", "f32rn", "dd_naive", "qf_naive"]:
    for kern in gen.KERNELS:
        src = gen.emit(tier, kern)
        found = find_bare_float_ops(src, int_names=gen.INT_NAMES)
        rows[f"{tier}/{kern}"] = dict(findings=len(found), first=[str(f) for f in found[:2]])
        print(f"{tier:9} {kern:3} {len(found):4} {found[0] if found else ''}")
pathlib.Path("results/check_codegen.json").write_text(json.dumps(rows, indent=1))
