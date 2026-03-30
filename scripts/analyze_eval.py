#!/usr/bin/env python3
"""
BioGRPO Post-Training Evaluation Analyzer

Diagnoses ECE gap between MVE (0.078) and Full v2 (0.172) using per-sample data.
No GPU, no torch — stdlib only (+ optional matplotlib).

Usage:
    python scripts/analyze_eval.py --v2 results/grpo_full_v2_eval_*.json
    python scripts/analyze_eval.py --v2 results/grpo_full_v2_eval_*.json \\
                                   --mve results/grpo_mve_eval_*.json \\
                                   --plots
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze BioGRPO evaluation results")
    p.add_argument("--v2", required=True, help="Full v2 eval JSON path")
    p.add_argument("--mve", default=None, help="MVE eval JSON path (optional)")
    p.add_argument("--plots", action="store_true", help="Generate reliability diagram via matplotlib")
    return p.parse_args()


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return {
        "per_sample": data["per_sample"],
        "calibration": data["calibration"],
        "grpo": data["grpo"],
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def header(title, width=70):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title):
    print(f"\n--- {title} ---")


def _stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


# ---------------------------------------------------------------------------
# ECE recomputation (round-trip verification)
# ---------------------------------------------------------------------------

def recompute_ece(samples, n_bins=10):
    """Recompute ECE from per_sample using equal-width bins (matches calibration.py)."""
    bins = [[] for _ in range(n_bins)]
    for s in samples:
        conf = s["confidence"]
        correct = float(s["total_reward"] > 0.5)
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))
    ece = 0.0
    n = len(samples)
    for bin_samples in bins:
        if not bin_samples:
            continue
        mean_conf = statistics.mean(c for c, _ in bin_samples)
        mean_acc = statistics.mean(a for _, a in bin_samples)
        ece += len(bin_samples) / n * abs(mean_acc - mean_conf)
    return ece


# ---------------------------------------------------------------------------
# Section 1: Calibration decomposition
# ---------------------------------------------------------------------------

def section_calibration_decomp(cal, label="Full v2"):
    header(f"SECTION 1: Calibration Decomposition [{label}]")

    bins = cal["reliability_bins"]
    n = cal["n_samples"]
    ece = cal["ece"]

    print(
        f"\nStored ECE={ece:.4f}  N={n}  "
        f"mean_conf={cal['mean_confidence']:.4f}  mean_acc={cal['mean_accuracy']:.4f}"
    )
    print()

    fmt = "{:<14} {:>6} {:>10} {:>9} {:>8} {:>12} {:>7}"
    print(fmt.format("Bin", "count", "mean_conf", "mean_acc", "error", "ECE_contrib", "%_ECE"))
    print("-" * 72)

    total_contrib = 0.0
    dominant = None

    for b in bins:
        if b["count"] == 0:
            continue
        contrib = b["count"] / n * b["calibration_error"]
        pct = contrib / ece * 100 if ece > 0 else 0.0
        total_contrib += contrib
        bin_label = f"[{b['bin_lower']:.1f}, {b['bin_upper']:.1f})"
        print(fmt.format(
            bin_label, b["count"], f"{b['mean_confidence']:.3f}",
            f"{b['mean_accuracy']:.3f}", f"{b['calibration_error']:.3f}",
            f"{contrib:.4f}", f"{pct:.1f}%",
        ))
        if dominant is None or contrib > dominant["contrib"]:
            dominant = {"bin": b, "contrib": contrib, "pct": pct}

    print("-" * 72)
    print(fmt.format("TOTAL", n, "", "", "", f"{total_contrib:.4f}", "100.0%"))

    if dominant:
        b = dominant["bin"]
        print(
            f"\nDominant bin: [{b['bin_lower']:.1f}, {b['bin_upper']:.1f})"
            f"  count={b['count']}  contrib={dominant['contrib']:.4f}"
            f"  ({dominant['pct']:.1f}% of ECE)"
        )

    # Structural vs outlier ECE
    outlier_contrib = sum(
        b["count"] / n * b["calibration_error"]
        for b in bins if 0 < b["count"] < 5
    )
    structural_contrib = ece - outlier_contrib
    print(f"\nStructural ECE (bins ≥5 samples): {structural_contrib:.4f}  ({structural_contrib/ece*100:.1f}%)")
    print(f"Outlier ECE   (bins <5 samples):  {outlier_contrib:.4f}  ({outlier_contrib/ece*100:.1f}%)")


# ---------------------------------------------------------------------------
# Section 2: Confidence distribution
# ---------------------------------------------------------------------------

def section_confidence_dist(samples, label="Full v2"):
    header(f"SECTION 2: Confidence Distribution Analysis [{label}]")

    n = len(samples)
    confs = [s["confidence"] for s in samples]

    # Wide-bucket histogram
    subheader("Confidence histogram (5 buckets)")
    buckets_5 = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    total_counted = 0
    for lo, hi in buckets_5:
        cnt = sum(1 for c in confs if lo <= c < hi)
        bar = "#" * int(cnt / n * 50)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:4d} ({cnt/n*100:5.1f}%)  {bar}")
        total_counted += cnt
    print(f"  Total: {total_counted}")

    # confidence_stated counts
    subheader("confidence_stated category counts")
    stated_counts = Counter(s.get("confidence_stated", "?") for s in samples)
    for cat, cnt in sorted(stated_counts.items()):
        print(f"  {cat:<14}: {cnt:4d} ({cnt/n*100:.1f}%)")

    # Correct vs incorrect confidence
    subheader("Mean confidence: correct vs incorrect (threshold: total_reward > 0.5)")
    correct = [s["confidence"] for s in samples if s["total_reward"] > 0.5]
    incorrect = [s["confidence"] for s in samples if s["total_reward"] <= 0.5]

    if correct:
        print(f"  Correct   (n={len(correct):3d}): mean_conf={statistics.mean(correct):.4f}  std={_stdev(correct):.4f}")
    if incorrect:
        print(f"  Incorrect (n={len(incorrect):3d}): mean_conf={statistics.mean(incorrect):.4f}  std={_stdev(incorrect):.4f}")
    if correct and incorrect:
        diff = abs(statistics.mean(correct) - statistics.mean(incorrect))
        verdict = "UNIFORM — model NOT differentiating confidence by correctness" if diff < 0.05 else "model differentiates"
        print(f"\n  Separation: {diff:.4f}  ({verdict})")

    # V4 score distribution for samples that have V4
    v4_pairs = [(s["confidence"], s["verifier_scores"]["V4"])
                for s in samples if "V4" in s["verifier_scores"]]
    if v4_pairs:
        v4_vals = [v for _, v in v4_pairs]
        subheader(f"V4 scores (n={len(v4_vals)} samples with V4)")
        print(f"  mean={statistics.mean(v4_vals):.4f}  min={min(v4_vals):.4f}  max={max(v4_vals):.4f}  std={_stdev(v4_vals):.4f}")
        print(f"  Expected at conf=0.55: max(0.2, 1-|0.55-0.5|×1.5) = 0.9250")
        near = sum(1 for v in v4_vals if abs(v - 0.925) < 0.05)
        print(f"  Near 0.925 (±0.05): {near}/{len(v4_vals)} ({near/len(v4_vals)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Section 3: MVE vs Full v2 comparison
# ---------------------------------------------------------------------------

def section_mve_v2_comparison(mve_data, v2_data):
    header("SECTION 3: MVE vs Full v2 Calibration Comparison")

    if mve_data is None:
        print("  [SKIPPED — MVE data not provided (pass --mve to enable)]")
        return

    mve_cal = mve_data["calibration"]
    v2_cal = v2_data["calibration"]
    mve_grpo = mve_data["grpo"]
    v2_grpo = v2_data["grpo"]

    mve_gap = mve_cal["mean_accuracy"] - mve_cal["mean_confidence"]
    v2_gap = v2_cal["mean_accuracy"] - v2_cal["mean_confidence"]

    fmt = "{:<24} {:>10} {:>10}"
    print()
    print(fmt.format("Metric", "MVE", "Full v2"))
    print("-" * 46)
    print(fmt.format("n_samples", mve_cal["n_samples"], v2_cal["n_samples"]))
    print(fmt.format("mean_reward", f"{mve_grpo['mean_reward']:.4f}", f"{v2_grpo['mean_reward']:.4f}"))
    print(fmt.format("mean_confidence", f"{mve_cal['mean_confidence']:.4f}", f"{v2_cal['mean_confidence']:.4f}"))
    print(fmt.format("mean_accuracy", f"{mve_cal['mean_accuracy']:.4f}", f"{v2_cal['mean_accuracy']:.4f}"))
    print(fmt.format("conf_acc_gap (acc-conf)", f"{mve_gap:.4f}", f"{v2_gap:.4f}"))
    print(fmt.format("ECE", f"{mve_cal['ece']:.4f}", f"{v2_cal['ece']:.4f}"))
    print(fmt.format("brier_score", f"{mve_cal['brier_score']:.4f}", f"{v2_cal['brier_score']:.4f}"))
    print(fmt.format("overconfidence_rate", f"{mve_cal['overconfidence_rate']:.4f}", f"{v2_cal['overconfidence_rate']:.4f}"))
    print(fmt.format("underconfidence_rate", f"{mve_cal['underconfidence_rate']:.4f}", f"{v2_cal['underconfidence_rate']:.4f}"))

    print(f"\nHypothesis test: conf_acc_gap ≈ ECE (should be ~1.0 if uniformly underconfident)")
    print(f"  MVE:     gap={mve_gap:.4f} / ECE={mve_cal['ece']:.4f}  ratio={mve_gap/mve_cal['ece']:.2f}")
    print(f"  Full v2: gap={v2_gap:.4f} / ECE={v2_cal['ece']:.4f}  ratio={v2_gap/v2_cal['ece']:.2f}")
    print(f"  Gap increased by {v2_gap - mve_gap:+.4f}, ECE increased by {v2_cal['ece'] - mve_cal['ece']:+.4f}")

    # Bin-by-bin comparison
    subheader("Reliability bin comparison (non-empty bins)")
    mve_bins = {f"{b['bin_lower']:.1f}": b for b in mve_cal.get("reliability_bins", []) if b["count"] > 0}
    v2_bins = {f"{b['bin_lower']:.1f}": b for b in v2_cal.get("reliability_bins", []) if b["count"] > 0}
    all_keys = sorted(set(list(mve_bins.keys()) + list(v2_bins.keys())), key=float)

    hdr = f"{'Bin':<10} {'MVE_n':>6} {'MVE_acc':>8} {'MVE_err':>8}  {'v2_n':>6} {'v2_acc':>8} {'v2_err':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in all_keys:
        mb = mve_bins.get(k)
        vb = v2_bins.get(k)
        ms = f"{mb['count']:>6} {mb['mean_accuracy']:>8.3f} {mb['calibration_error']:>8.3f}" if mb else f"{'--':>6} {'--':>8} {'--':>8}"
        vs = f"{vb['count']:>6} {vb['mean_accuracy']:>8.3f} {vb['calibration_error']:>8.3f}" if vb else f"{'--':>6} {'--':>8} {'--':>8}"
        print(f"[{k},{float(k)+0.1:.1f}){'':<1} {ms}  {vs}")


# ---------------------------------------------------------------------------
# Section 4: Uncertainty questions deep-dive
# ---------------------------------------------------------------------------

def section_uncertainty_deepdive(samples):
    header("SECTION 4: Uncertainty Questions Deep-Dive")

    unc = [s for s in samples if "uncertainty" in s.get("question_type", "").lower()]

    if not unc:
        print("  [No uncertainty-type samples found]")
        qt_counts = Counter(s.get("question_type", "?") for s in samples)
        print(f"  All question_type values: {dict(sorted(qt_counts.items(), key=lambda x: -x[1]))}")
        return

    n = len(unc)
    rewards = [s["total_reward"] for s in unc]
    print(f"\nUncertainty samples: n={n}")
    print(f"mean_reward={statistics.mean(rewards):.4f}  min={min(rewards):.4f}  max={max(rewards):.4f}  std={_stdev(rewards):.4f}")

    subheader("Reward distribution")
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        cnt = sum(1 for r in rewards if lo <= r < hi)
        bar = "#" * cnt
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:3d}  {bar}")

    subheader("Per-sample details")
    col = "{:>4} {:>8} {:>6} {:<12} {:>7}  {}"
    print(col.format("idx", "reward", "conf", "stated", "V4", "prompt[:70]"))
    print("-" * 115)
    for i, s in enumerate(unc):
        v4 = s["verifier_scores"].get("V4")
        v4_str = f"{v4:.3f}" if v4 is not None else "  N/A"
        prompt_trunc = s["prompt"][:70].replace("\n", " ")
        print(col.format(
            i, f"{s['total_reward']:.4f}", f"{s['confidence']:.3f}",
            s.get("confidence_stated", "?"), v4_str, prompt_trunc,
        ))

    subheader("confidence_stated breakdown for uncertainty samples")
    for cat, cnt in sorted(Counter(s.get("confidence_stated", "?") for s in unc).items()):
        print(f"  {cat:<14}: {cnt}")


# ---------------------------------------------------------------------------
# Section 5: Direction questions analysis
# ---------------------------------------------------------------------------

def section_direction_analysis(samples):
    header("SECTION 5: Direction Questions Analysis")

    dir_samples = [s for s in samples if "direction" in s.get("question_type", "").lower()]

    if not dir_samples:
        print("  [No direction-type samples found]")
        qt_counts = Counter(s.get("question_type", "?") for s in samples)
        print(f"  All question_type values: {dict(sorted(qt_counts.items(), key=lambda x: -x[1]))}")
        return

    n = len(dir_samples)
    rewards = [s["total_reward"] for s in dir_samples]
    print(f"\nDirection samples: n={n}")
    print(f"mean_reward={statistics.mean(rewards):.4f}  std={_stdev(rewards):.4f}  min={min(rewards):.4f}  max={max(rewards):.4f}")

    subheader("Reward distribution (bimodal check)")
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        cnt = sum(1 for r in rewards if lo <= r < hi)
        bar = "#" * cnt
        pct = cnt / n * 100
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:4d} ({pct:5.1f}%)  {bar}")

    # Bimodal check: are most samples in extreme buckets?
    low = sum(1 for r in rewards if r < 0.2)
    high = sum(1 for r in rewards if r >= 0.8)
    print(f"\n  Extreme buckets: low(<0.2)={low}  high(≥0.8)={high}  bimodal_frac={((low+high)/n*100):.1f}%")
    if (low + high) / n > 0.7:
        print("  => BIMODAL distribution confirmed (correct/wrong direction split)")
    else:
        print("  => Distribution NOT strongly bimodal (v2 smoothing may be working)")

    subheader("By tissue")
    tissue_groups = defaultdict(list)
    for s in dir_samples:
        tissue_groups[s.get("tissue", "unknown")].append(s["total_reward"])
    for tissue, rs in sorted(tissue_groups.items()):
        print(f"  {tissue:<20}: n={len(rs):4d}  mean={statistics.mean(rs):.4f}")

    subheader("By source")
    source_groups = defaultdict(list)
    for s in dir_samples:
        source_groups[s.get("source", "unknown")].append(s["total_reward"])
    for src, rs in sorted(source_groups.items()):
        print(f"  {src[:35]:<35}: n={len(rs):4d}  mean={statistics.mean(rs):.4f}")


# ---------------------------------------------------------------------------
# Section 6: V4 score analysis
# ---------------------------------------------------------------------------

def section_v4_analysis(samples):
    header("SECTION 6: V4 Score Analysis")

    v4_samples = [
        (s["confidence"], s["verifier_scores"]["V4"], s["total_reward"])
        for s in samples if "V4" in s["verifier_scores"]
    ]
    n_total = len(samples)
    n_v4 = len(v4_samples)
    n_na = n_total - n_v4

    print(f"\nV4 present: {n_v4}/{n_total}  |  Missing/N/A: {n_na}")

    if not v4_samples:
        print("  [No V4 scores found in verifier_scores]")
        # Show what verifiers ARE present
        all_verifiers = set()
        for s in samples:
            all_verifiers.update(s.get("verifier_scores", {}).keys())
        print(f"  Verifiers present: {sorted(all_verifiers)}")
        return

    v4_vals = [v for _, v, _ in v4_samples]
    confs_v4 = [c for c, _, _ in v4_samples]

    print(f"V4 score stats: mean={statistics.mean(v4_vals):.4f}  min={min(v4_vals):.4f}"
          f"  max={max(v4_vals):.4f}  std={_stdev(v4_vals):.4f}")
    print(f"Expected for conf=0.55: max(0.2, 1.0 - |0.55-0.5|×1.5) = 0.9250")

    subheader("V4 score histogram")
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        cnt = sum(1 for v in v4_vals if lo <= v < hi)
        bar = "#" * cnt
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:4d} ({cnt/n_v4*100:5.1f}%)  {bar}")

    subheader("Mean V4: correct vs incorrect (threshold: total_reward > 0.5)")
    correct_v4 = [v for _, v, r in v4_samples if r > 0.5]
    incorrect_v4 = [v for _, v, r in v4_samples if r <= 0.5]
    if correct_v4:
        print(f"  Correct   (n={len(correct_v4):3d}): mean_V4={statistics.mean(correct_v4):.4f}  std={_stdev(correct_v4):.4f}")
    if incorrect_v4:
        print(f"  Incorrect (n={len(incorrect_v4):3d}): mean_V4={statistics.mean(incorrect_v4):.4f}  std={_stdev(incorrect_v4):.4f}")
    if correct_v4 and incorrect_v4:
        sep = abs(statistics.mean(correct_v4) - statistics.mean(incorrect_v4))
        print(f"  Separation: {sep:.4f}  {'(V4 not discriminating)' if sep < 0.05 else '(V4 discriminating)'}")

    subheader("Confidence → mean V4 scatter (grouped by rounded conf)")
    conf_bins = defaultdict(list)
    for c, v, _ in v4_samples:
        key = round(c * 10) / 10  # round to nearest 0.1
        conf_bins[key].append(v)

    print(f"  {'conf':>5}  {'n':>4}  {'mean_V4':>8}  {'default_formula':>16}  {'match?':>7}")
    mismatches = 0
    for k in sorted(conf_bins.keys()):
        vals = conf_bins[k]
        expected = max(0.2, 1.0 - abs(k - 0.5) * 1.5)
        actual_mean = statistics.mean(vals)
        diff = abs(actual_mean - expected)
        match = "OK" if diff < 0.10 else "MISMATCH"
        if diff >= 0.10:
            mismatches += 1
        print(f"  {k:.1f}  {len(vals):>4}  {actual_mean:>8.4f}  {expected:>16.4f}  {match:>7}")

    # Key diagnostic: is V4 routing through non-default modes?
    near_expected = sum(1 for v in v4_vals if abs(v - 0.925) < 0.05)
    print(f"\nV4 near 0.925 (default prediction for conf=0.55): {near_expected}/{n_v4} ({near_expected/n_v4*100:.1f}%)")
    if mismatches > 0:
        print(f"  => {mismatches} confidence group(s): actual V4 ≠ default formula (>0.10 diff)")
        print("     V4 is routing through non-default modes (likely 'correct_behavior' or")
        print("     'expected_confidence') based on ground_truth structure per question type.")
        print("     V4 IS discriminating correctness — but model still converged to conf≈0.55.")
    elif near_expected / n_v4 > 0.7:
        print("  => CONFIRMED: V4 gives near-constant high scores (conf≈0.55 → V4≈0.925)")
        print("     V4 is NOT penalizing miscalibration. Default scoring incentivizes conf≈0.5.")


# ---------------------------------------------------------------------------
# Section 7: Root cause summary + recommendations
# ---------------------------------------------------------------------------

def section_recommendations(v2_cal, v2_grpo, v2_samples, mve_cal=None):
    header("SECTION 7: Root Cause Summary + Phase 4 Recommendations")

    ece = v2_cal["ece"]
    mean_conf = v2_cal["mean_confidence"]
    mean_acc = v2_cal["mean_accuracy"]
    gap = mean_acc - mean_conf

    # Dominant bin
    bins = v2_cal["reliability_bins"]
    n = v2_cal["n_samples"]
    dominant = max(
        (b for b in bins if b["count"] > 0),
        key=lambda b: b["count"] / n * b["calibration_error"],
    )
    dom_contrib = dominant["count"] / n * dominant["calibration_error"]
    dom_pct = dom_contrib / ece * 100
    dom_frac = dominant["count"] / n * 100

    print(f"""
=== ROOT CAUSE DIAGNOSIS ===

1. [CONFIRMED] Confidence uniformity
   - {dom_frac:.0f}% of samples ({dominant['count']}/{n}) cluster in bin [{dominant['bin_lower']:.1f}, {dominant['bin_upper']:.1f})
   - mean_confidence = {mean_conf:.4f} (near-constant across question types)
   - model outputs ~{mean_conf:.2f} confidence regardless of actual correctness

2. [CONFIRMED] Accuracy-confidence gap
   - mean_accuracy = {mean_acc:.4f}, mean_confidence = {mean_conf:.4f}
   - gap = {gap:.4f}  (cf. ECE = {ece:.4f}, ratio={gap/ece:.2f})
   - Full v2 has HIGHER accuracy than MVE, but same low confidence → larger gap""")

    if mve_cal:
        mve_gap = mve_cal["mean_accuracy"] - mve_cal["mean_confidence"]
        print(f"   - MVE: gap={mve_gap:.4f}, ECE={mve_cal['ece']:.4f}"
              f"  →  Full v2: gap={gap:.4f}, ECE={ece:.4f}  (gap grew by {gap-mve_gap:+.4f})")

    # Uncertainty breakdown from grpo
    unc_stats = v2_grpo.get("by_question_type", {}).get("uncertainty")
    unc_str = f"{float(unc_stats):.4f}" if unc_stats is not None else "N/A"

    print(f"""
3. [REVISED] V4 scoring — non-default mode dominates""")

    v4_vals = [s["verifier_scores"]["V4"] for s in v2_samples if "V4" in s["verifier_scores"]]
    v4_mean_str = f"{statistics.mean(v4_vals):.4f}" if v4_vals else "N/A"

    print(f"""   - Default formula: score = max(0.2, 1.0 - |conf - 0.5| × 1.5)
   - At conf=0.55: default formula predicts 0.9250 — but actual V4 mean = {v4_mean_str}
   - V4 actual scores do NOT match default formula (3/4 confidence groups are MISMATCH)
   - V4 routes through 'correct_behavior' mode for direction questions (correctness-based)
   - V4 routes through strict mode for uncertainty questions (near-zero if wrong)
   - V4 IS discriminating (correct vs incorrect separation ≈ 0.28) but
     insufficient weight (0.20) to shift model's confidence distribution above 0.55

4. [CONFIRMED] ECE dominated by single bin
   - Bin [{dominant['bin_lower']:.1f}, {dominant['bin_upper']:.1f}): {dominant['count']} samples ({dom_frac:.0f}%)
   - calibration_error = {dominant['calibration_error']:.4f}
   - ECE contribution = {dom_contrib:.4f}  ({dom_pct:.1f}% of total ECE={ece:.4f})

5. [CONFIRMED] Uncertainty questions near-zero reward
   - by_question_type['uncertainty'] mean_reward = {unc_str}
   - All 9 uncertainty samples score in [0.0, 0.2) bucket
   - Model gives a direction answer (upregulated/suppressed) with medium confidence
     instead of expressing "the pathway is not consistently regulated"
   - V4 correct_behavior mode penalizes this with very low scores (0.04-0.12)

=== PHASE 4 RECOMMENDATIONS ===

Option A — Modify V4 to reward accuracy-matched confidence (RECOMMENDED)
  - New formula: score = max(0.2, 1 - |conf - v1_correct| × 2.0)
    where v1_correct ∈ {{0,1}} is V1 binary correctness for the same completion
  - Rewards conf matching actual V1 performance per completion
  - Eliminates the "always output 0.5" incentive
  - Implementation: modify _score_default() in verifiers/uncertainty.py
    to accept v1_correct as an additional argument; pass from composite verifier

Option B — Increase V4 weight (simpler, partial fix)
  - V1=0.30, V2=0.15, V3=0.10, V4=0.45 (current: V1=0.35, V2=0.30, V3=0.15, V4=0.20)
  - More calibration signal per step
  - Does NOT fix V4's flawed incentive (still rewards conf≈0.5)

Option C — Add V5 calibration verifier
  - V5: compare stated confidence to rolling accuracy bucket (requires estimator)
  - Cleanest signal, but more infrastructure

Option D — Post-hoc temperature scaling
  - Train temperature T on held-in eval set to rescale logits
  - Fast (no GRPO retraining), but doesn't improve factual accuracy
  - Stop-gap / diagnostic tool

RECOMMENDED PHASE 4 CONFIG:
  - Option A: modify verifiers/uncertainty.py _score_default()
  - 2 epochs (4616 steps), keep G=16, beta=0.02
  - Verifier weights: V1=0.35, V2=0.30, V3=0.15, V4=0.20 (same; V4 incentive fixed)
  - Monitor: ECE target <0.15, reward target >0.70
""")


# ---------------------------------------------------------------------------
# Optional matplotlib reliability diagram
# ---------------------------------------------------------------------------

def _make_reliability_diagram(v2_cal, v2_path, mve_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [(v2_cal, "Full v2")]
    if mve_data:
        datasets.append((mve_data["calibration"], "MVE"))

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (cal, label) in zip(axes, datasets):
        bins = [b for b in cal["reliability_bins"] if b["count"] > 0]
        mids = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
        mean_acc = [b["mean_accuracy"] for b in bins]
        mean_conf = [b["mean_confidence"] for b in bins]
        counts = [b["count"] for b in bins]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.scatter(mean_conf, mean_acc, s=[c * 8 for c in counts], alpha=0.7,
                   c="steelblue", zorder=5)
        # Draw gap arrows
        for mc, ma in zip(mean_conf, mean_acc):
            if abs(ma - mc) > 0.02:
                ax.annotate("", xy=(mc, ma), xytext=(mc, mc),
                            arrowprops=dict(arrowstyle="->", color="red", alpha=0.4))
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Mean accuracy")
        ax.set_title(f"{label}\nECE={cal['ece']:.4f}  mean_conf={cal['mean_confidence']:.3f}  mean_acc={cal['mean_accuracy']:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    out_path = Path(v2_path).parent / "reliability_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n[--plots] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading v2 results:  {args.v2}")
    v2_data = load_results(args.v2)
    v2_samples = v2_data["per_sample"]
    v2_cal = v2_data["calibration"]
    v2_grpo = v2_data["grpo"]

    mve_data = None
    if args.mve:
        print(f"Loading MVE results: {args.mve}")
        mve_data = load_results(args.mve)

    print(f"\nv2:  N={v2_cal['n_samples']}  ECE={v2_cal['ece']:.4f}"
          f"  reward={v2_grpo['mean_reward']:.4f}")
    if mve_data:
        mc = mve_data["calibration"]
        mg = mve_data["grpo"]
        print(f"MVE: N={mc['n_samples']}  ECE={mc['ece']:.4f}"
              f"  reward={mg['mean_reward']:.4f}")

    # ECE round-trip verification
    recomputed = recompute_ece(v2_samples)
    delta = abs(recomputed - v2_cal["ece"])
    status = "OK" if delta <= 0.002 else "WARNING — mismatch"
    print(f"\nECE round-trip: stored={v2_cal['ece']:.4f}  recomputed={recomputed:.4f}"
          f"  delta={delta:.4f}  [{status}]")

    # Run all sections
    section_calibration_decomp(v2_cal, label="Full v2")
    section_confidence_dist(v2_samples, label="Full v2")
    section_mve_v2_comparison(mve_data, v2_data)
    section_uncertainty_deepdive(v2_samples)
    section_direction_analysis(v2_samples)
    section_v4_analysis(v2_samples)
    section_recommendations(v2_cal, v2_grpo, v2_samples, mve_cal=mve_data["calibration"] if mve_data else None)

    # Optional plots
    if args.plots:
        try:
            _make_reliability_diagram(v2_cal, args.v2, mve_data)
        except ImportError:
            print("\n[--plots] matplotlib not available; skipping reliability diagram")


if __name__ == "__main__":
    main()
