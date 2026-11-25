"""Bands vs square tiles (256, 512 px) for f32 and qf on the worst-case views (throwaway).

Per-chunk fixed cost (chunks x per-chunk overhead), load balance (achieved Giter/s; the tail
of a chunk with few busy blocks idles SMs), longest chunk. Alternating order, 2 rounds.

usage: .venv/bin/python tiles.py <cache root> -> results/tiles.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import s5lib

from brotkit.backends.chunks import ChunkPolicy


def main() -> None:
    gpu = s5lib.Gpu.open(pathlib.Path(sys.argv[1]))
    s5lib.s4_on_path()
    import q3_throughput as q

    baseline = int(q.smi("memory.used")[0])
    out = []
    for tier in ("f32", "qf"):
        gpu.kernel(tier, "C")
        for view in ("whole", "interior"):
            for rnd in range(2):
                for tile in (None, 256, 512):
                    s5lib.wait_idle(q, baseline, f"{tier}/{view}/{tile}")
                    r = s5lib.render(gpu, tier, "C", view, policy=ChunkPolicy(tile=tile))
                    s = r.summary() | {"round": rnd}
                    out.append(s)
                    print(
                        f"{tier:4} {view:9} tile={tile!s:5} r{rnd} chunks={s['chunks']:5} "
                        f"wall={s['wall_s']:.3f}s overhead={s['overhead'] * 100:.3f}% "
                        f"giter/s={s['giter_s']} max={s['chunk_s_max']}s "
                        f"over_ceil={s['over_ceiling']}",
                        flush=True,
                    )
                    (s5lib.HERE / "results" / "tiles.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
