"""Turn results/q3_*.json into the markdown tables in docs/design/spikes/S4.md (CPU only).

usage: uv run python analyze.py > results/tables.md
"""

from __future__ import annotations

import json
import pathlib

R = pathlib.Path(__file__).parent / "results"
TIERS = ("f32", "df64", "f64", "dd", "qf")
KERNS = ("A", "B", "C", "F6", "S2")


def load(name: str) -> dict:
    p = R / f"q3_{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main_table(main: dict) -> None:
    print("### Throughput and occupancy, sm_86 (RTX 3090 Ti), default build\n")
    print(
        "Giter/s, median of 5 after warm-up; occupancy is theoretical (resource_usage + "
        "occupancy, equal to the driver's).\n"
    )
    print(
        "| Tier | Kernel | Features | Regs | Spill st/ld (B) | Local (B) | Occ@128 | Occ@256 "
        "| Giter/s @128 | Giter/s @256 | SM clock (MHz) | Max temp (C) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = [(t, k) for t in TIERS for k in KERNS] + [("f32rn", "A"), ("qfb", "C")]
    for t, k in rows:
        a, b = main.get(f"{t}/{k}/default/128"), main.get(f"{t}/{k}/default/256")
        if not (a and b):
            continue
        clk = sorted(set(a["sm_clock_mhz"]) | set(b["sm_clock_mhz"]))
        flag = " (flag)" if a["clock_flag"] or b["clock_flag"] else ""
        print(
            f"| {t} | {k} | {b['features']} | {b['regs']} | {b['spill_st']}/{b['spill_ld']} "
            f"| {b['stack']} | {a['occupancy']:.3f} | {b['occupancy']:.3f} "
            f"| {a['giter_s']:.4g} | {b['giter_s']:.4g} "
            f"| {min(clk):.0f}-{max(clk):.0f}{flag} | {max(a['temp_c'] + b['temp_c']):.0f} |"
        )
    print()


def split_table(main: dict) -> None:
    print("### Fused 6-feature pass vs split (C + S2), seconds per 4K render\n")
    print("| Tier | Block | F6 fused (s) | C + S2 split (s) | Split / fused | Fused spills? |")
    print("| --- | --- | --- | --- | --- | --- |")
    for t in TIERS:
        for bs in (128, 256):
            f = main.get(f"{t}/F6/default/{bs}")
            c = main.get(f"{t}/C/default/{bs}")
            s = main.get(f"{t}/S2/default/{bs}")
            if not (f and c and s):
                continue
            tf, ts = f["seconds_median"], c["seconds_median"] + s["seconds_median"]
            sp = "yes" if f["spill_st"] or f["spill_ld"] else "no"
            print(f"| {t} | {bs} | {tf:.4g} | {ts:.4g} | {ts / tf:.2f} | {sp} |")
    print()


def ratio_table(main: dict) -> None:
    print("### qf vs dd (block 256 and 128)\n")
    print("| Kernel | dd @256 | qf @256 | qf/dd @256 | dd @128 | qf @128 | qf/dd @128 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for k in KERNS:
        v = [main.get(f"{t}/{k}/default/{bs}") for bs in (256, 128) for t in ("dd", "qf")]
        if not all(v):
            continue
        d2, q2, d1, q1 = (x["giter_s"] for x in v)
        print(
            f"| {k} | {d2:.4g} | {q2:.4g} | {q2 / d2:.1f}x | {d1:.4g} | {q1:.4g} | {q1 / d1:.1f}x |"
        )
    print()


def caps_table(main: dict, caps: dict) -> None:
    print("### Register caps (block 256, sm_86)\n")
    print("Throughput relative to the uncapped default build of the same kernel.\n")
    print(
        "| Tier | Kernel | Option | Regs | Spill st/ld (B) | Local (B) | Occ@256 | Giter/s "
        "| vs default |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, r in caps.items():
        if key.startswith("_"):
            continue
        base = main.get(f"{r['tier']}/{r['kernel']}/default/256")
        rel = f"{r['giter_s'] / base['giter_s']:.2f}" if base else "-"
        opt = ", ".join(f"{k}={v}" for k, v in r["opts"].items())
        print(
            f"| {r['tier']} | {r['kernel']} | {opt} | {r['regs']} "
            f"| {r['spill_st']}/{r['spill_ld']} | {r['stack']} | {r['occupancy']:.3f} "
            f"| {r['giter_s']:.4g} | {rel}{' (flag)' if r['clock_flag'] else ''} |"
        )
    print()


def resources_table(res: dict) -> None:
    if not res:
        return
    print(
        "### Compile-only resources at sm_86 / sm_100 / sm_120 (registers, spill B, local B, "
        "occupancy @128/@256)\n"
    )
    print("| Tier | Kernel | Feat | sm_86 | sm_100 | sm_120 |")
    print("| --- | --- | --- | --- | --- | --- |")
    tiers = ("f32", "f32rn", "f64", "df64", "dd", "qf", "qfb")
    for t in tiers:
        for k in ("Z", "A", "B", "C", "F6", "S2"):
            cells = []
            feat = "-"
            for cc in ("86", "100", "120"):
                r = res.get(f"sm_{cc}/{t}/{k}")
                if r is None:
                    cells.append("-")
                elif "error" in r:
                    cells.append("error")
                else:
                    feat = str(r["features"])
                    sp = r["spill_st"] + r["spill_ld"]
                    cells.append(
                        f"{r['regs']} r, {sp} sp, {r['stack']} L, "
                        f"{r['occ128']:.2f}/{r['occ256']:.2f}"
                    )
            print(f"| {t} | {k} | {feat} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main = load("main")
    main_table(main)
    split_table(main)
    ratio_table(main)
    caps_table(main, load("caps"))
    resources_table(load("resources"))
