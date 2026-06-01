# Warmstart iFBO — The Story (Real Transformer Learning-Curve Data)

**Scope:** PAWS slice of `demo_analysis/warmstart_runs_flattened.parquet`. Transformer widths
32M / 77M / 134M / 287M / 610M params. FT-PFN v0.0.1, hybrid geomspace+linspace grid at k=256,
LOO normalization. Strict lower→higher guard enforced everywhere: a context curve may only be used
as a partner if `partner_target_N < query_target_N` (32M is never a target — it has no lower partner).

**Question (supervisor framing):** Can iFBO use *full* learning curves from smaller architectures to
improve prediction and HP-ranking at a *larger* target it has only partially observed — and where does
it break?

---

## TL;DR

1. **iFBO transfer genuinely helps.** Adding lower-scale curves to context drops final-point NLL from
   **1.79 (prefix-only) → 0.32 (all same-scale-below context)** — an 82% reduction — and lifts ranking
   quality at small/mid targets to **Spearman ρ ≈ 0.92–0.96**.
2. **But it collapses at the largest width (610M).** Final-target ranking ρ *degrades* as we observe
   more of the target prefix and goes **negative** (**0.60 → 0.38 → 0.00 → −0.31** as obs goes
   5% → 50%) — the model inverts the config ranking — while decision regret stays persistently bad
   (**~0.20**, vs ≤0.08 for every other width). More data makes the prediction worse, not better. This
   is the LC-shape-change problem in concrete numbers.
3. **The mechanism that helps is diversity, not similarity.** The nearest lower scale carries most of
   the signal; adding *all* lower curves helps most; a partner curve helps in proportion to how
   *different* it is from the anchor already in context, not how *similar* it is to the query.
4. **The unsolved problem for iFBO v2 is calibration.** Even the best context policy sits at
   **coverage90 ≈ 0.44** against a 0.90 target. Tight uncertainty bands are systematically overconfident.

---

## The dataset and how curves "change shape" across width

In normalized space the curves are *roughly* self-similar across scales (see
`ex0_6_supervisor_prefix_only/curve_shape_overlay.png`) — which is why transfer works at all. But the
similarity is not exact: the larger the target, the lower and later its loss bends, and the prefix of a
big-width curve looks deceptively like the *body* of a small-width curve. That mismatch is the root of
the 610M failure below.

---

## Foundation: single-slice sanity checks (Ex0.5–Ex0.12)

Before scaling to the full PAWS dataset, we tested basic transfer mechanics on a single HP slice
(`base_N=14M, shrink=0.4, tkpm=20`) containing exactly 5 curves: one per target scale (32M, 77M, 134M,
287M, 610M). All experiments use `obs_frac=0.2` and LOO normalization within the slice.

**Ex0.5 — Demo reproduction.** Reproduced the supervisor's demo notebook plots on the new flattened
dataset. Confirmed the raw data looks correct and matches expectations.

**Ex0.6 — Prefix-only (no transfer baseline).** Each target sees only its own observed prefix — no
lower-scale context. Bands are uniformly wide (3.2–3.9) for all targets. This is the "no transfer"
baseline. Note: 77M and 134M show a visual band artifact at the prediction boundary (see
`BAND_ARTIFACT_NOTE.md`) — this is genuine model uncertainty under minimal context, not a normalization
bug.

**Ex0.7 — Sibling context (oracle).** Each target sees the other 4 sibling-scale curves as full context.
This is the "oracle" regime — the model sees both lower AND higher scales. Not a valid lower→higher
test, but shows what FT-PFN can do with rich context.

**Ex0.8 — All-lower-only (strict lower→higher).** Each target sees only strictly lower-scale curves.
This is the first real lower→higher transfer test. Key result: 134M becomes dramatically tight
(band 0.16) with just 2 lower partners (32M + 77M), while 77M stays wide (1.71) with only 1 far partner
(32M). 610M is moderate (1.08) with 4 lower partners.

**Ex0.9 — Nearest-lower-only.** Each target sees only the single nearest lower-scale curve. Key finding:
for 134M, nearest-only (77M) nearly matches all-lower (band 0.17 vs 0.16) — the nearest neighbor carries
almost all the signal. But for 287M and 610M, nearest-only is much wider than all-lower (2.62 vs 2.03
and 2.20 vs 1.08), showing that additional lower curves help at larger scales.

**Ex0.10–Ex0.12 — How much curve shape is needed?** These test whether full curves are necessary or
whether sparse summaries suffice:

| policy (610M target)         | points per partner | final band width |
|------------------------------|-------------------|------------------|
| Ex0.12 start+final pt only  | 2                 | 5.16 (worse than prefix-only!) |
| Ex0.10 final pt only         | 1                 | 3.90 (≈ prefix-only) |
| Ex0.11 geom sparse k=8      | 8                 | 3.20 |
| Ex0.8 all-lower full curves | 256               | 1.08 |

**The finding:** Full curve shape matters enormously. Endpoints alone are useless (no better than
prefix-only). Even 8 geometrically-spaced points only modestly help. The full k=256 hybrid-grid
representation is needed for effective transfer. This justifies the current subsampling approach.

**What the Ex0.x series established:**
1. Transfer works — lower-scale context dramatically tightens predictions (Ex0.8 vs Ex0.6).
2. Nearest lower scale carries most signal for close targets (Ex0.9 vs Ex0.8 for 134M).
3. Additional lower scales help for larger targets (Ex0.9 vs Ex0.8 for 287M/610M).
4. Full curve shape is essential — sparse summaries are insufficient (Ex0.10–Ex0.12).
5. The normalization coordinate system is critical (LOO fix; see `normalization_decision.md`).

These findings motivated scaling up to the full PAWS dataset (Ex3) to test whether the patterns hold
with many more curves and diverse HP configurations.

---

## Q1 — The problem iFBO faces with real, shape-changing curves

**Hypothesis.** A freeze-thaw surrogate trained mostly on smaller-scale dynamics should extrapolate to
larger widths, getting *more* accurate as it observes more of the target prefix.

**What we observe (Ex3 width-extrapolation, policy `all_lower_only`, ranking at final horizon
`t_target=1.0` — the decision-relevant point, since warmstarting picks configs by predicted *final*
performance):**

| target | ρ@obs5% | ρ@obs10% | ρ@obs20% | ρ@obs50% | regret@50% | coverage90@10% |
|--------|---------|----------|----------|----------|------------|----------------|
| 77M    | 0.655   | 0.756    | 0.950    | **0.946**| 0.029      | 0.600 |
| 134M   | 0.854   | 0.931    | 0.946    | **0.950**| 0.011      | 0.846 |
| 287M   | 0.773   | 0.841    | 0.868    | 0.890    | 0.081      | 0.275 |
| 610M   | 0.595   | 0.381    | 0.000    | **−0.310**| **0.203** | 0.614 |

(Numbers are Spearman ρ at `t_target=1.0`. Averaging ρ over all 10 horizons *masks* the collapse —
early horizons sit near the observed prefix and look healthy — so the final-horizon figure is the honest
one for a warmstarting decision.)

**The finding.** For 77M / 134M / 287M, iFBO behaves exactly as a learning-curve predictor should:
final-target ranking improves with more observation, regret stays small (≤0.08). **At 610M it inverts** —
observing *more* of the target curve drives ranking from weak-positive to **negative** (ρ 0.60 → −0.31:
the model orders configs *backwards*), and regret sits stuck at ~0.20, roughly 2–18× worse than any other
width. The surrogate latches onto the early shape of the 610M prefix, matches it to the wrong region of
the smaller-scale curves it knows, and grows *more* confident in the wrong extrapolation as the prefix
lengthens.

This is the concrete statement of "real data with LC shape change is a problem iFBO faces": **the
failure is not random noise, it is a systematic, monotone breakdown that gets worse with more data, and
it appears precisely at the largest width — the regime warmstarting is supposed to serve.**

---

## Q2 — Can iFBO be made to handle it? (context selection)

**Hypothesis.** If the surrogate can't extrapolate from the prefix alone, the right *lower-scale context*
should supply the missing shape information.

**What we observe.**

**(a) Transfer clearly helps — Ex4B baselines (610M target, obs=10%):**

| context policy            | final NLL | Δ vs prefix | coverage90 |
|---------------------------|-----------|-------------|------------|
| prefix_only (no transfer) | 1.789     |  —          | 0.005 |
| + exact 287M anchor       | 1.148     | −0.64       | 0.579 |
| + all lower scales        | 0.837     | −0.95       | 0.614 |
| + all 287M (nearest)      | **0.318** | **−1.47**   | 0.439 |

Adding context cuts NLL by up to 82%. **The single nearest lower scale (287M) does most of the work**
(`ex3_1_nearest_scale_ablation_A`): nearest-only bands are ~3–4× tighter than far-scales-only.

**(b) It is diversity, not raw count — Ex3.1E duplicate-context ablation (610M):**

| context                          | final band width |
|----------------------------------|------------------|
| 4 real lower curves              | 3.95 |
| same 4 cycled to 101 entries     | 4.13 (no better) |
| 35 real nearest (287M) curves    | 1.08 |
| 101 real lower curves            | **0.55** |

Padding context to 101 entries by *duplication* buys nothing. Genuine variety across the lower pool is
what tightens predictions.

**(c) Which single partner to add? Dissimilarity-from-anchor, not similarity-to-query — Ex4D
(100 partners × 8 queries, anchor = exact-matched 287M):**

- Best partner: **77M, G=5.30, different_shrink → Δ NLL vs anchor = −0.107** (most dissimilar to anchor).
- Useless partner: **287M, G=19.72, same_shrink → Δ = −0.002** (a near-duplicate of the anchor).
- The predicted "G-proximity to query" ranking **failed**: closeness to the query G does not order the
  partners. Complementarity to what's already in context does.

**The finding for Q2.** iFBO *can* be helped by context, and there is a clean, defensible rule:
**include all available lower-scale curves; the benefit is ensemble diversity, with the nearest scale as
the anchor.** This recovers small/mid-target performance. **It does not fully repair 610M** — context
lowers NLL there too, but the ranking collapse and the calibration gap persist (all-287M coverage90 is
still only 0.44).

---

## Q3 — How should iFBO be updated / what benchmark for iFBO v2?

**What's still broken after the best context policy:**

1. **Calibration.** coverage90 ≈ 0.44 (best) vs 0.90 target; 287M sits at 0.22–0.28. Bands are tight but
   systematically overconfident — the p50 is biased high (the model expects loss to stop dropping before
   it actually does). Tight ≠ correct.
2. **Largest-width ranking.** 610M final-target ρ falls *below zero* (to −0.31) and regret stays stuck
   at ~0.20 as observation grows. The surrogate has no notion that a long prefix of a big curve is *not*
   the same as a short curve — so more data drives it to order configs backwards.

**Implications for iFBO v2:**

- **The training distribution, not just the context, needs the large-width shape.** FT-PFN v0.0.1 was
  not trained to know that wider curves keep descending past where narrow curves plateau. Either retrain
  the PFN with scale as an explicit conditioning input, or add a scale-aware prior.
- **The benchmark must score the regime that breaks.** Evaluate on the *strict* lower→higher task
  (`partner_target_N < query_target_N`), report per-target *and* break out the largest width separately,
  and sweep obs_frac (5/10/20/50%) — because the pathology only shows up as a function of prefix length.
- **The metric set must include calibration and decision quality, not just point error.** Band width and
  MAE both looked "fine" while ranking collapsed. The honest scorecard is:
  **Spearman ρ (ranking) · normalized regret + top-k hit-rate (decision) · coverage90 (calibration) ·
  NLL/MAE/RMSE (closeness)** — and a policy only "passes" if calibration holds, not just if bands are
  narrow.

**One-line answer to the supervisor:** *Warmstarting from lower widths works and is worth keeping — the
nearest lower scale plus full lower-curve diversity gives large NLL and ranking gains at 77–287M — but
iFBO v0.0.1 cannot extrapolate to the largest width and is overconfident everywhere, so iFBO v2 needs
scale-aware training and a benchmark that explicitly scores the largest-width, calibration-sensitive
regime.*

---

## Caveats

- **PAWS-only, one optimizer slice.** All numbers are the PAWS method on the flattened dataset; MLP (D1)
  and nanoTabPFN (D3) generalization is **not yet tested** (deferred per supervisor).
- **Small query counts.** Ex4B/Ex4D use 8 × 610M queries; partner win-rates are 44–69% (helpful on
  average, not guaranteed per-query). Treat single-partner deltas as directional.
- **Normalization is load-bearing.** Results are only comparable because of the LOO normalization fix
  (`normalization_decision.md`). The earlier lower-partner-only floor produced a catastrophic ~18%
  coverage artifact and 100% over-prediction; that bug is fixed but shows how sensitive the pipeline is.
- **Band width ≠ accuracy.** Several earlier notes ranked policies by band width alone; the Ex3 aggregate
  shows tight bands can be badly mis-calibrated (287M: tight bands, coverage 0.22). Always pair width
  with coverage90.
- **The 134M "anomaly" in the Ex0.x single-slice notes is noise.** On the 376-curve Ex3 aggregate, 134M
  is the *best*-behaved target (ρ≈0.95, coverage≈0.85), not anomalous.

---

## Experiment index (which experiment supports which claim)

| Claim | Evidence |
|-------|----------|
| Raw dataset matches supervisor expectations | `ex0_5_demo_reproduction/` |
| Prefix-only is weak; uncertainty huge under minimal context | `ex0_6_supervisor_prefix_only/` (band artifact note) |
| Curves roughly self-similar across width (transfer is plausible) | `ex0_6_supervisor_prefix_only/curve_shape_overlay.png` |
| Sibling context (oracle) shows FT-PFN can use multi-scale info | `ex0_7_context_ifbo/` |
| Lower→higher transfer works on single slice | `ex0_8_all_lower_only/` |
| Nearest lower scale carries most signal (single slice) | `ex0_9_nearest_lower_only/` |
| Full curve shape essential; endpoints/sparse insufficient | `ex0_10`…`ex0_12` |
| Lower→higher transfer works at scale; tightest at 610M w/ richest pool | `ex3_width_extrapolation/` |
| **610M ranking collapses as obs grows; regret climbs** | `ex3_width_extrapolation/rank_metrics.csv`, `regret_metrics.csv` |
| Nearest lower scale drives most of the tightening (PAWS-wide) | `ex3_1_nearest_scale_ablation_A/` |
| Diversity not count (duplication fails) | `ex3_1_E_duplicate_context_ablation/` |
| Context cuts NLL 82%; nearest-scale ensemble best | `ex4_2hp_lower_target_context_ablation/baseline_summary.csv` |
| Partner value ∝ dissimilarity-from-anchor (G-proximity fails) | `ex4d_multi_scale_partner_ablation/summary_by_scale_and_G.csv` |
| LOO normalization is the correct coordinate system | `normalization_decision.md` |
| Partner value ∝ dissimilarity-from-anchor (G-proximity fails) | `ex4d_multi_scale_partner_ablation/summary_by_scale_and_G.csv` |
| LOO normalization is the correct coordinate system | `normalization_decision.md` |
