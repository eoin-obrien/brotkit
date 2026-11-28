# Golden references

Phase 1's renderer must match a small set of exact references on every precision tier. They
are computed with mpmath, independently of brotkit's kernels, so they can tell a correct
renderer from a subtly wrong one: a half-pixel shift, a flipped row, a transposition, a tier
that silently runs out of precision. Backlog 0.4 ([#12](https://github.com/eoin-obrien/brotkit/issues/12));
the pixel convention they rely on is ADR-0011.

- **Data:** `tests/golden/<view>.npz` (Git LFS, uncompressed, byte-reproducible) and
  `tests/golden/manifest.json` (a normal file).
- **Generator:** `tools/make_goldens.py`, a dev tool that never runs in CI.
- **Comparison:** `brotkit.testing.golden` (`load_golden`, `compare`), pure NumPy.
- **Exact mapping and iteration:** `brotkit.testing.reference`, shared by the generator and
  the tests.
- **Previews:** `docs/_static/golden/<view>.png`, for humans only; tests never read them.
  Colour is smooth mu, interior is black, and sensitive pixels are tinted magenta.

## The views

R = 10⁶ and one sample per pixel everywhere; P = 256 bits, checked at 512. Centres are exact
decimal strings (SEA and M41 abbreviate the long ones below).

- **SEA** (seahorse valley) = −0.743643887037158704752191506114774 + 0.131825904205311970493132056385139i
- **M41** (Misiurewicz point M(4,1), from S3) = −0.101096363845622161025785445738622565463805442826253483876931 + 0.956286510809141500771096057729977435809833336510529170034314i

| View | Centre | Spacing (width) | Pixels | max_iter | n range | Escaped | Sensitive (max) | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `overview` | −0.75 + 0i | 0.05 (3.2) | 64 × 64 | 256 | 5–194 | 0.851 | 0.021 (0.04) | 3.9 s |
| `seahorse` | SEA | 1.6e-5 (1.024e-3) | 64 × 64 | 1000 | 40–902 | 1.000 | 0.156 (0.17) | 4.5 s |
| `seahorse_2e-6` | SEA | 2e-6 (1.28e-4) | 64 × 64 | 2000 | 71–1236 | 1.000 | 0.199 (0.21) | 6.0 s |
| `m41_1e-12` | M41 | 1e-12 (6.4e-11) | 64 × 64 | 1000 | 87–140 | 1.000 | 0.030 (0.04) | 6.5 s |
| `m41_1e-20` | M41 | 1e-20 (6.4e-19) | 64 × 64 | 5000 | 152–234 | 1.000 | 0.025 (0.04) | 10.3 s |
| `julia` | 0 + 0i, c = −0.8 + 0.156i | 0.05 (3.2) | 64 × 64 | 1000 | 5–913 | 1.000 | 0.094 (0.11) | 1.7 s |
| `seahorse_rotated` | SEA, rotation 0.5 rad | 1.6e-5 (1.024e-3) | 64 × 64 | 1000 | 39–945 | 1.000 | 0.158 (0.17) | 4.6 s |
| `elephant_61x37` | 0.2925 + 0.0145i | 5e-4 (0.0305) | 61 × 37 | 500 | 16–488 | 0.693 | 0.087 (0.10) | 7.1 s |
| `period2_bulb` | −1 + 0i | 0.0078125 (0.5) | 64 × 64 | 5000 | 14–4614 | 0.188 | 0.045 (0.06) | 267 s |

Times are on Xenon's 24 cores (plain mpmath, Python backend); about 5 minutes in all
(7 under desktop load), almost all of it the bulb's interior at 5000 iterations. **P vs 2P:** zero disagreements in
outcome, n or the float64 values of z, mu and DE on every non-sensitive pixel of every view,
and none on the sensitive pixels either. The NPZs total 1.5 MB.

### What each tier must do

Decidable fraction per tier (`expect` in brackets when it isn't "pass"). Each tier has a view
it passes and one it must fail; a failing view that passes is itself a test failure.

| View | f32 | df64 | f64 | dd | qf |
| --- | --- | --- | --- | --- | --- |
| `overview` | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 |
| `seahorse` | 0.673 (last pass) | 0.844 | 0.844 | 0.844 | 0.844 |
| `seahorse_2e-6` | 0.476 **[fail]** | 0.801 | 0.801 | 0.801 | 0.801 |
| `m41_1e-12` | [n/a] | 0.843 | 0.970 | 0.970 | 0.970 |
| `m41_1e-20` | [n/a] | 0.000 **[fail]** | [n/a] | 0.975 | 0.975 |
| `julia` | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 |
| `seahorse_rotated` | 0.672 | 0.842 | 0.842 | 0.842 | 0.842 |
| `elephant_61x37` | 0.913 | 0.913 | 0.913 | 0.913 | 0.913 |
| `period2_bulb` | 0.955 | 0.955 | 0.955 | 0.955 | 0.955 |

The negative tests fail on the decidable-fraction guard: f32 decides 0.476 of `seahorse_2e-6`
against the 0.79 the other tiers manage, and df64 decides none of `m41_1e-20`. Their renders
really are wrong (the GPU test asserts it): f32 gets n wrong on 1078 of 4096 pixels there, 303
of them non-sensitive, and df64 on all 4096. None of those wrong pixels is decidable, even at
k = 2, so the mask predicts where a tier can be trusted rather than hiding errors. A
renderer *bug* on a tier that should pass is caught directly: the dd stand-in without the
half-pixel offset gets 1483 decidable pixels wrong on `seahorse` (mu 3e5 eps, DE 5e7 eps), and
passes `elephant_61x37`, where odd sizes make the two conventions agree.

## How a golden is computed

For every pixel, at one sample per pixel on its centre (no supersampling):

1. Map the pixel centre to (z₀, c) exactly (ADR-0011) and iterate z² + c at a working
   precision P = max(256 bits, zoom bits + 128), where zoom bits = log₂(max(1, |centre|) / s).
   Every current view has P = 256.
2. Record the layers below, with R = 10⁶ on every view (the spec's 1e3 for f32 is a minimum).
3. **Sensitivity.** The pixel is `sensitive` if its n or outcome changes when its plane point
   moves by ±1/1000 px in u or in v, or when R changes by 1 part in 10⁶.
4. **Stability radius.** `stable_px` is the largest nudge on the ladder
   1/1000, 3/1000, 1/100, 3/100, 1/10, 3/10 px for which n and outcome stay the same under
   ±nudge in u and in v (checked in order; the first change stops it). 0 means even 1/1000
   px changes it.
5. **Convergence.** Recompute the pixel at 2P. The run fails unless P and 2P agree on outcome
   and n, and on the float64 values of z, mu and DE, on every pixel not flagged sensitive.

| Layer | Type | Meaning |
| --- | --- | --- |
| `outcome` | uint8 | `Outcome.ESCAPED` (0) or `Outcome.MAX_ITER` (3) |
| `n` | uint32 | first k with \|z_k\| > R; max_iter if none |
| `z_final` | complex128 | z_n at escape (z_max_iter inside), rounded to float64 |
| `mu` | float64 | n + 1 − log(log\|z_n\| / log R) / log 2; NaN inside |
| `de` | float64 | 2·\|z_n\|·ln\|z_n\| / \|z′_n\| (`estimate`), z′ = dz/dc (Mandelbrot) or dz/dz₀ (Julia); NaN inside |
| `sensitive` | bool | step 3 |
| `stable_px` | float32 | step 4 |

## What "matches" means

`compare(layers, golden, tier)` returns a `Report` and applies, per tier:

- **Decidable pixels** must match outcome and n exactly. A pixel is decidable for a tier when
  it isn't sensitive and `stable_px ≥ 4 × c_rounding_px`, where
  c_rounding_px = eps(tier) · max(|u|, |v|) / s is how far rounding its plane coordinates to
  the tier can move it.
- **Decidable fraction.** It must be at least the manifest's `min_decidable` for that tier
  and view, and the report always prints it, so the mask can't excuse everything. Tiers the
  view expects to pass get their own measured fraction less 0.01; tiers it expects to fail
  get the smallest `min_decidable` of the passing tiers on that view.
- **Sensitive pixels** aren't checked one by one. The report counts those within ±1 in n or
  with a flipped outcome, and the rest as outliers; the view's sensitive fraction must not
  exceed the manifest's `max_sensitive_fraction`. (A ±1 rule was tried and dropped: stand-ins
  that pass everything else put up to 26% of a view's sensitive pixels outside it (df64 at
  its limit on `m41_1e-12`; f64 up to 11 of 817 on `seahorse_2e-6`; dd none), because a pixel
  that moves under a 1/1000 px nudge moves under orbit rounding too.)
- **mu and DE** are compared on decidable escaped pixels whose n matches, as a relative
  error in units of the eps of the precision they were *computed in*, against the tier's
  budget in `TOLERANCES` (table below).
- **float32 DE.** When DE comes from a float32 derivative, it is compared only on pixels
  whose |dz|² fits in float32 (|dz| < √FLT_MAX / 2 ≈ 9.2e18), and it is *not applicable* to
  a view where fewer than half the escaped pixels do (`f32_de_ok`).
- **Not applicable** tiers (`expect: "n/a"`) are refused, not given loose tolerances.

Tier eps (`TIER_EPS`): f32 2⁻²⁴, df64 2⁻⁴⁸, f64 2⁻⁵³, qf 2⁻⁹⁶, dd 2⁻¹⁰⁶.

## Tolerances

Budgets are about 4× the largest error measured on any view the tier passes (S4 stand-ins,
`uv run python -m tests.golden_standin`). Every stand-in computes mu and DE in float32 from the
hi word, so every budget is in float32 eps (2⁻²⁴); the "computed in" column will differ once
the renderer does. Errors are relative, max over decidable escaped pixels with matching n.

| Tier | Computed in | mu budget | mu max seen (view) | DE budget | DE max seen (view) |
| --- | --- | --- | --- | --- | --- |
| f32 | f32 | 25 000 | 6 122 (`seahorse`) | 1.7e6 | 4.2e5 (`seahorse`) |
| df64 | f32 | 24 000 | 5 943 (`m41_1e-12`) | 11 000 | 2 720 (`seahorse_2e-6`) |
| f64 | f32 | 24 000 | 5 807 (`m41_1e-12`) | 15 | 3.73 (`seahorse_2e-6`) |
| dd | f32 | 4 | 1.00 (`elephant_61x37`) | 15 | 3.73 (`seahorse_2e-6`) |
| qf | f32 | 600 | 141 (`overview`) | 11 000 | 2 720 (`seahorse_2e-6`) |

All measurements (mu / DE max in f32 eps; DE "n/a" where float32 DE is not applicable):

| View | f32 | df64 | f64 | dd | qf |
| --- | --- | --- | --- | --- | --- |
| `overview` | 103 / 1277 | 141 / 1366 | 0.98 / 3.12 | 0.98 / 3.12 | 141 / 1366 |
| `seahorse` | 6122 / 4.2e5 | 38 / 2356 | 0.97 / 3.36 | 0.97 / 3.36 | 38 / 2356 |
| `seahorse_2e-6` | [fail] 22534 / 1.4e6 | 27 / 2720 | 0.97 / 3.73 | 0.97 / 3.73 | 27 / 2717 |
| `m41_1e-12` | n/a | 5943 / n/a | 5807 / n/a | 0.77 / n/a | 61 / n/a |
| `m41_1e-20` | n/a | [fail] - | n/a | 0.86 / n/a | 34 / n/a |
| `elephant_61x37` | 4248 / 1.3e5 | 100 / 2448 | 1.00 / 3.39 | 1.00 / 3.39 | 100 / 2448 |
| `period2_bulb` | 1087 / 13025 | 0.94 / 2.96 | 0.94 / 3.05 | 0.94 / 3.05 | 0.94 / 2.96 |

Readings:

- **df64 and qf match each other, not f64,** except on `period2_bulb`. That is the stand-in,
  not the tiers: S4's kernel takes the spacing `dx` in the word type, so for df64 and qf it is
  rounded to float32, moving pixels by up to about 1e-6 px, which the derivative amplifies
  near the boundary. `period2_bulb`'s spacing (2⁻⁷) is exact in float32, and there df64 and
  qf equal f64. The phase 1 renderer should pass the spacing in tier precision; then df64's
  and qf's budgets will tighten a lot.
- **f64 at `m41_1e-12`** (5807) and **df64 there** (5943) are real: both tiers are at their
  limit, and the budget is per tier, so it is loose on their easy views (where they reach
  about 1). Per-view budgets are an option once the renderer replaces the stand-ins.
- **float32 DE stops being applicable at spacing 1e-12.** |dz|² fits in float32 on 85–100% of
  escaped pixels down to `seahorse_2e-6`, on 7.6% at `m41_1e-12` and on none at `m41_1e-20`
  (`f32_de_in_range_fraction`). That is the spec's open question on mu/DE precision for the
  multi-word tiers.

## Regenerating

Changing a golden needs the maintainer's explicit approval every time (AGENTS.md invariant 6,
`update-golden-images` skill). Then:

```bash
git commit ...                                # the generator refuses uncommitted changes to itself
uv run tools/make_goldens.py                  # all views; or name some: ... elephant_61x37
uv run tools/make_goldens.py --check          # regenerate into a temp dir and compare sha256
```

It uses every core (`--jobs N` to limit). `tests/test_golden.py`'s slow test regenerates
`elephant_61x37` and checks its sha256 against the manifest, so a change in mpmath, numpy or
the generator that alters results is caught in CPU CI.

## Stand-ins until phase 1

There is no renderer yet, so `tests/test_golden_gpu.py` runs S4's spike kernel B (smooth +
DE) for each tier against every view it can render (Mandelbrot plane, no rotation):
`tests/fixtures/codegen/s4_<tier>_B_r1e6.py.txt`, emitted by
`spikes/S4_registers/gen.py <tier> B 1e12` so R = 10⁶ on every tier. Three spike conventions
differ from the goldens, and `tests/golden_standin.py` adjusts each, marked `SPIKE`:

- **Pixel mapping:** the spike maps x to c₀ + (x − ⌊W/2⌋)·dx, with no half-pixel offset.
  Passing ADR-0011's centre of pixel (⌊W/2⌋, ⌊H/2⌋) as c₀ makes them identical (a CPU test
  checks this on every view).
- **mu:** the spike computes n + 1 − log₂(ln|z|), without the spec's `/ ln R`; the stand-in
  adds log₂(ln R).
- **DE:** the spike computes |z|·ln|z| / |dz|, half the spec's `estimate`; the stand-in
  doubles it.

The spike rounds the spacing to its word type (float32 for df64 and qf), which moves pixels by
up to about 1e-6 px: well inside the 1/1000 px ladder. Julia (`julia`) and rotated
(`seahorse_rotated`) views have CPU mapping and reproduction tests only until phase 1.
**Delete the stand-ins and their adjustments when the phase 1 renderer replaces them**
(backlog 1.9), and recalibrate the tolerances against it.
