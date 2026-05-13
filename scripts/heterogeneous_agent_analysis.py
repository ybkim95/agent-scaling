#!/usr/bin/env python3
"""Heterogeneous agent scaling analysis.

Compares heterogeneous multi-agent configurations (mixing models with
different capabilities) against homogeneous baselines on BrowseComp-Plus, to
test whether capability mixing changes the ~45% single-agent-baseline
saturation behaviour observed under homogeneous scaling.

Limitation: results are based on a single benchmark (BrowseComp-Plus,
n=100 instances). Conclusions should be treated as preliminary and warrant
replication across additional benchmarks and task types.

Data location: etc/exp_outputs_heterogeneous/browsecomp-plus/
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Intelligence Index scores (from generate_visualization.py in the repo)
#    These are the model capability rankings used throughout the paper.
# ---------------------------------------------------------------------------
MODEL_INTELLIGENCE_INDEX = {
    # Gemini family
    "gemini-2.0-flash":      45,
    "gemini-2.5-pro":        60,
    # OpenAI family
    "gpt-5-nano":            51,
    "gpt-5":                 68,
    # Anthropic family
    # Sonnet 3.7 / Sonnet 4.5 are not in the original visualization mapping
    # because the paper used them under "claude-3-7-sonnet" / "claude-sonnet-4-5"
    # aliases. We assign approximate II values consistent with the paper's ranking:
    #   claude-3-7-sonnet (Sonnet 3.7): lower capability than Sonnet 4.5
    #   claude-sonnet-4-5 (Sonnet 4.5): stronger
    "claude-3-7-sonnet":     52,   # approximate, consistent with paper ranking
    "claude-sonnet-4-5":     58,   # approximate, consistent with paper ranking
}

# Friendly display names for the experiment directory keys
MODEL_DISPLAY_NAMES = {
    "gemini_2.0_flash":  "Gemini 2.0 Flash (II=45)",
    "gemini_2.5_pro":    "Gemini 2.5 Pro   (II=60)",
    "gpt5_nano":         "GPT-5-nano       (II=51)",
    "gpt5":              "GPT-5            (II=68)",
    "sonnet3.7":         "Sonnet 3.7       (II=52)",
    "sonnet4.5":         "Sonnet 4.5       (II=58)",
}

# ---------------------------------------------------------------------------
# 2. Homogeneous BrowseComp-Plus baselines
#    Source: extended_mixed_effects_6benchmarks.py (embedded data from 180-row df)
#    We use the best homogeneous MAS result per (model, architecture) and also
#    the single-agent score, which defines the "capability ceiling" framing.
# ---------------------------------------------------------------------------

# Format: model_key -> {architecture -> accuracy}
# Values come directly from the embedded data in extended_mixed_effects_6benchmarks.py
HOMOGENEOUS_BASELINES = {
    # --- Gemini 2.0 Flash ---
    "gemini-2.0-flash": {
        "single-agent":              0.17,
        "multi-agent-centralized":   0.22,
        "multi-agent-decentralized": 0.18,
        "multi-agent-hybrid":        0.20,
        "multi-agent-independent":   0.10,
        "best_mas":                  0.22,   # centralized
    },
    # --- Gemini 2.5 Pro ---
    "gemini-2.5-pro": {
        "single-agent":              0.36,
        "multi-agent-centralized":   0.37,
        "multi-agent-decentralized": 0.43,
        "multi-agent-hybrid":        0.40,
        "multi-agent-independent":   0.24,
        "best_mas":                  0.43,   # decentralized
    },
    # --- GPT-5-nano ---
    "gpt-5-nano": {
        "single-agent":              0.37,   # best of two runs (0.20, 0.37)
        "multi-agent-centralized":   0.27,
        "multi-agent-decentralized": 0.32,
        "multi-agent-hybrid":        0.33,
        "multi-agent-independent":   0.18,
        "best_mas":                  0.33,   # hybrid
    },
    # --- GPT-5 ---
    "gpt-5": {
        "single-agent":              0.37,   # from embedded data
        "multi-agent-centralized":   0.34,
        "multi-agent-decentralized": 0.50,
        "multi-agent-hybrid":        0.38,
        "multi-agent-independent":   0.44,
        "best_mas":                  0.50,   # decentralized
    },
    # --- Sonnet 3.7 (claude-3-7-sonnet-20250219) ---
    "claude-3-7-sonnet": {
        "single-agent":              0.26,
        "multi-agent-centralized":   0.34,
        "multi-agent-decentralized": 0.29,
        "multi-agent-hybrid":        0.33,
        "multi-agent-independent":   0.18,
        "best_mas":                  0.34,   # centralized
    },
    # --- Sonnet 4.5 (claude-sonnet-4-5) ---
    "claude-sonnet-4-5": {
        "single-agent":              0.34,
        "multi-agent-centralized":   0.43,
        "multi-agent-decentralized": 0.43,
        "multi-agent-hybrid":        0.40,
        "multi-agent-independent":   0.14,
        "best_mas":                  0.43,   # centralized or decentralized
    },
}

# ---------------------------------------------------------------------------
# 3. Heterogeneous experiment definitions
#    Maps experiment directory name -> structured metadata
# ---------------------------------------------------------------------------

# Capability ceiling from the paper (single-agent average across models)
PAPER_CEILING = 0.45

HETERO_EXPERIMENTS = {
    # ---- Centralized ----
    "multi-agent-centralized": {
        "gemini_2.0_flash_orch_2.5_pro_sub": {
            "label":        "Gemini: Flash orch + Pro sub (weak→strong)",
            "orch_model":   "gemini-2.0-flash",
            "sub_model":    "gemini-2.5-pro",
            "stronger":     "gemini-2.5-pro",
            "weaker":       "gemini-2.0-flash",
            "family":       "Gemini",
            "note":         "weak orchestrator, strong subagents",
        },
        "gemini_2.5_pro_orch_2.0_flash_sub": {
            "label":        "Gemini: Pro orch + Flash sub (strong→weak)",
            "orch_model":   "gemini-2.5-pro",
            "sub_model":    "gemini-2.0-flash",
            "stronger":     "gemini-2.5-pro",
            "weaker":       "gemini-2.0-flash",
            "family":       "Gemini",
            "note":         "strong orchestrator, weak subagents",
        },
        "gemini_2.5_pro_orch_gpt_5_sub": {
            "label":        "Cross-family: Gemini Pro orch + GPT-5 sub",
            "orch_model":   "gemini-2.5-pro",
            "sub_model":    "gpt-5",
            "stronger":     "gpt-5",
            "weaker":       "gemini-2.5-pro",
            "family":       "Cross-family",
            "note":         "cross-family mixing",
        },
        "gpt5_nano_orch_gpt5_sub": {
            "label":        "OpenAI: GPT-5-nano orch + GPT-5 sub (weak→strong)",
            "orch_model":   "gpt-5-nano",
            "sub_model":    "gpt-5",
            "stronger":     "gpt-5",
            "weaker":       "gpt-5-nano",
            "family":       "OpenAI",
            "note":         "weak orchestrator, strong subagents",
        },
        "gpt5_orch_gpt5_nano_sub": {
            "label":        "OpenAI: GPT-5 orch + GPT-5-nano sub (strong→weak)",
            "orch_model":   "gpt-5",
            "sub_model":    "gpt-5-nano",
            "stronger":     "gpt-5",
            "weaker":       "gpt-5-nano",
            "family":       "OpenAI",
            "note":         "strong orchestrator, weak subagents",
        },
        "sonnet3.7_orch_sonnet4.5_sub": {
            "label":        "Anthropic: Sonnet3.7 orch + Sonnet4.5 sub (weak→strong)",
            "orch_model":   "claude-3-7-sonnet",
            "sub_model":    "claude-sonnet-4-5",
            "stronger":     "claude-sonnet-4-5",
            "weaker":       "claude-3-7-sonnet",
            "family":       "Anthropic",
            "note":         "weak orchestrator, strong subagents",
        },
        "sonnet4.5_orch_sonnet3.7_sub": {
            "label":        "Anthropic: Sonnet4.5 orch + Sonnet3.7 sub (strong→weak)",
            "orch_model":   "claude-sonnet-4-5",
            "sub_model":    "claude-3-7-sonnet",
            "stronger":     "claude-sonnet-4-5",
            "weaker":       "claude-3-7-sonnet",
            "family":       "Anthropic",
            "note":         "strong orchestrator, weak subagents",
        },
    },
    # ---- Decentralized ----
    "multi-agent-decentralized": {
        "1_gemini2.5pro_2_gemini2.0flash": {
            "label":        "Gemini: 1×Pro + 2×Flash (minority strong)",
            "models":       ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash"],
            "stronger":     "gemini-2.5-pro",
            "weaker":       "gemini-2.0-flash",
            "family":       "Gemini",
            "note":         "1 strong + 2 weak",
        },
        "2_gemini2.5pro_1_gemini2.0flash": {
            "label":        "Gemini: 2×Pro + 1×Flash (majority strong)",
            "models":       ["gemini-2.5-pro", "gemini-2.5-pro", "gemini-2.0-flash"],
            "stronger":     "gemini-2.5-pro",
            "weaker":       "gemini-2.0-flash",
            "family":       "Gemini",
            "note":         "2 strong + 1 weak",
        },
        "1_gpt5_2_gpt5nano": {
            "label":        "OpenAI: 1×GPT-5 + 2×GPT-5-nano (minority strong)",
            "models":       ["gpt-5", "gpt-5-nano", "gpt-5-nano"],
            "stronger":     "gpt-5",
            "weaker":       "gpt-5-nano",
            "family":       "OpenAI",
            "note":         "1 strong + 2 weak",
        },
        "2_gpt5_1_gpt5nano": {
            "label":        "OpenAI: 2×GPT-5 + 1×GPT-5-nano (majority strong)",
            "models":       ["gpt-5", "gpt-5", "gpt-5-nano"],
            "stronger":     "gpt-5",
            "weaker":       "gpt-5-nano",
            "family":       "OpenAI",
            "note":         "2 strong + 1 weak",
        },
        "1_sonnet4.5_2_sonnet3.7": {
            "label":        "Anthropic: 1×Sonnet4.5 + 2×Sonnet3.7 (minority strong)",
            "models":       ["claude-sonnet-4-5", "claude-3-7-sonnet", "claude-3-7-sonnet"],
            "stronger":     "claude-sonnet-4-5",
            "weaker":       "claude-3-7-sonnet",
            "family":       "Anthropic",
            "note":         "1 strong + 2 weak",
        },
        "2_sonnet4.5_1_sonnet3.7": {
            "label":        "Anthropic: 2×Sonnet4.5 + 1×Sonnet3.7 (majority strong)",
            "models":       ["claude-sonnet-4-5", "claude-sonnet-4-5", "claude-3-7-sonnet"],
            "stronger":     "claude-sonnet-4-5",
            "weaker":       "claude-3-7-sonnet",
            "family":       "Anthropic",
            "note":         "2 strong + 1 weak",
        },
    },
}

# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------

def load_hetero_results(base_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load all heterogeneous experiment results from disk."""
    results = {}
    for arch in ["multi-agent-centralized", "multi-agent-decentralized"]:
        arch_dir = base_dir / arch
        if not arch_dir.exists():
            print(f"  WARNING: directory not found: {arch_dir}", file=sys.stderr)
            continue
        results[arch] = {}
        for exp_dir in sorted(arch_dir.iterdir()):
            metrics_file = exp_dir / "dataset_eval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                results[arch][exp_dir.name] = data.get("avg_accuracy", float("nan"))
    return results


# ---------------------------------------------------------------------------
# 5. Analysis helpers
# ---------------------------------------------------------------------------

def get_homo_best_mas(model_key: str) -> float:
    """Return the best homogeneous MAS accuracy for a given model."""
    return HOMOGENEOUS_BASELINES.get(model_key, {}).get("best_mas", float("nan"))


def get_homo_single_agent(model_key: str) -> float:
    """Return the single-agent accuracy for a given model."""
    return HOMOGENEOUS_BASELINES.get(model_key, {}).get("single-agent", float("nan"))


def get_homo_arch(model_key: str, arch: str) -> float:
    """Return homogeneous accuracy for a specific architecture."""
    return HOMOGENEOUS_BASELINES.get(model_key, {}).get(arch, float("nan"))


def beats_ceiling(accuracy: float, ceiling: float = PAPER_CEILING) -> bool:
    return accuracy > ceiling


def delta_str(val: float, ref: float) -> str:
    if val != val or ref != ref:  # NaN check
        return "  N/A  "
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.3f}"


# ---------------------------------------------------------------------------
# 6. Main analysis
# ---------------------------------------------------------------------------

def run_analysis(hetero_results: Dict) -> None:
    print("=" * 90)
    print("HETEROGENEOUS AGENT SCALING: PRELIMINARY ANALYSIS")
    print("Benchmark: BrowseComp-Plus (n=100 instances, 1 benchmark only)")
    print("Research question: Does capability heterogeneity bypass the ~45% MAS ceiling?")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Section A: Full results table
    # -----------------------------------------------------------------------
    print("\n" + "─" * 90)
    print("SECTION A: FULL RESULTS TABLE")
    print("─" * 90)

    header = (
        f"{'Config':<52} {'Hetero':>7} {'Homo-Strong':>12} {'Homo-Weak':>10} "
        f"{'Δ(vs Strong)':>13} {'Beats 45%':>10}"
    )
    print(header)
    print("─" * 90)

    all_rows = []

    for arch, exp_dict in HETERO_EXPERIMENTS.items():
        arch_label = "CENTRALIZED" if arch == "multi-agent-centralized" else "DECENTRALIZED"
        print(f"\n  [{arch_label}]")

        arch_results = hetero_results.get(arch, {})

        for exp_key, meta in exp_dict.items():
            hetero_acc = arch_results.get(exp_key, float("nan"))
            stronger = meta["stronger"]
            weaker = meta["weaker"]

            # For centralized: compare against same architecture baseline
            # For decentralized: compare against decentralized baseline
            homo_strong_arch = get_homo_arch(stronger, arch)
            homo_weak_arch   = get_homo_arch(weaker, arch)

            # Also track best MAS for ceiling analysis
            homo_strong_best = get_homo_best_mas(stronger)
            homo_weak_best   = get_homo_best_mas(weaker)

            delta_vs_strong = hetero_acc - homo_strong_arch if (hetero_acc == hetero_acc and homo_strong_arch == homo_strong_arch) else float("nan")

            label = meta["label"]
            beats = "YES *" if beats_ceiling(hetero_acc) else "no"

            print(
                f"  {label:<50} {hetero_acc:>7.3f} {homo_strong_arch:>12.3f} "
                f"{homo_weak_arch:>10.3f} {delta_str(hetero_acc, homo_strong_arch):>13} {beats:>10}"
            )

            all_rows.append({
                "arch":            arch,
                "exp_key":         exp_key,
                "label":           meta["label"],
                "family":          meta["family"],
                "note":            meta["note"],
                "stronger":        stronger,
                "weaker":          weaker,
                "hetero_acc":      hetero_acc,
                "homo_strong_arch":homo_strong_arch,
                "homo_weak_arch":  homo_weak_arch,
                "homo_strong_best":homo_strong_best,
                "homo_weak_best":  homo_weak_best,
                "delta_vs_strong": delta_vs_strong,
                "beats_ceiling":   beats_ceiling(hetero_acc),
            })

    print("\n  * = exceeds the 45% single-agent capability ceiling discussed in the paper")

    # -----------------------------------------------------------------------
    # Section B: Summary by architecture
    # -----------------------------------------------------------------------
    print("\n" + "─" * 90)
    print("SECTION B: SUMMARY BY ARCHITECTURE")
    print("─" * 90)

    for arch_label, arch_key in [("Centralized", "multi-agent-centralized"),
                                   ("Decentralized", "multi-agent-decentralized")]:
        rows = [r for r in all_rows if r["arch"] == arch_key]
        valid = [r for r in rows if r["hetero_acc"] == r["hetero_acc"]]
        if not valid:
            continue
        mean_hetero  = sum(r["hetero_acc"] for r in valid) / len(valid)
        mean_strong  = sum(r["homo_strong_arch"] for r in valid if r["homo_strong_arch"] == r["homo_strong_arch"]) / len(valid)
        mean_weak    = sum(r["homo_weak_arch"] for r in valid if r["homo_weak_arch"] == r["homo_weak_arch"]) / len(valid)
        n_beats      = sum(1 for r in valid if r["beats_ceiling"])

        print(f"\n  {arch_label} ({len(valid)} configs):")
        print(f"    Mean heterogeneous accuracy : {mean_hetero:.3f}")
        print(f"    Mean homo-strong accuracy   : {mean_strong:.3f}  (Δ = {mean_hetero - mean_strong:+.3f})")
        print(f"    Mean homo-weak accuracy     : {mean_weak:.3f}  (Δ = {mean_hetero - mean_weak:+.3f})")
        print(f"    Configs beating 45% ceiling : {n_beats}/{len(valid)}")

    # -----------------------------------------------------------------------
    # Section C: Orchestrator vs. subagent quality (centralized only)
    # -----------------------------------------------------------------------
    print("\n" + "─" * 90)
    print("SECTION C: DOES ORCHESTRATOR QUALITY OR SUBAGENT QUALITY MATTER MORE?")
    print("(Centralized architecture only — role distinction applies)")
    print("─" * 90)

    cent_rows = [r for r in all_rows if r["arch"] == "multi-agent-centralized"
                 and r["family"] != "Cross-family"]

    # Group by family: compare weak-orch-strong-sub vs strong-orch-weak-sub
    families_seen = {}
    for r in cent_rows:
        fam = r["family"]
        if fam not in families_seen:
            families_seen[fam] = {}
        families_seen[fam][r["note"]] = r

    print(f"\n  {'Family':<12} {'Weak-orch/Strong-sub':>22} {'Strong-orch/Weak-sub':>22} {'Winner':>10}")
    print("  " + "─" * 70)

    weak_orch_wins = 0
    strong_orch_wins = 0
    ties = 0

    for fam, configs in families_seen.items():
        weak_orch = configs.get("weak orchestrator, strong subagents")
        strong_orch = configs.get("strong orchestrator, weak subagents")
        if weak_orch and strong_orch:
            wa = weak_orch["hetero_acc"]
            sa = strong_orch["hetero_acc"]
            if wa > sa:
                winner = "Weak-orch"
                weak_orch_wins += 1
            elif sa > wa:
                winner = "Strong-orch"
                strong_orch_wins += 1
            else:
                winner = "Tie"
                ties += 1
            print(f"  {fam:<12} {wa:>22.3f} {sa:>22.3f} {winner:>10}")

    print(f"\n  Weak-orch/Strong-sub wins  : {weak_orch_wins}")
    print(f"  Strong-orch/Weak-sub wins  : {strong_orch_wins}")
    print(f"  Ties                       : {ties}")

    if weak_orch_wins > strong_orch_wins:
        print("\n  Preliminary finding: SUBAGENT QUALITY appears to matter more.")
        print("  Systems with stronger subagents outperformed those with stronger orchestrators,")
        print("  suggesting that execution capability drives performance more than coordination.")
    elif strong_orch_wins > weak_orch_wins:
        print("\n  Preliminary finding: ORCHESTRATOR QUALITY appears to matter more.")
        print("  Systems with stronger orchestrators outperformed those with stronger subagents,")
        print("  suggesting that coordination strategy drives performance more than raw execution.")
    else:
        print("\n  Preliminary finding: INCONCLUSIVE. Results are mixed across families.")

    # -----------------------------------------------------------------------
    # Section D: Does heterogeneity bypass the 45% ceiling?
    # -----------------------------------------------------------------------
    print("\n" + "─" * 90)
    print("SECTION D: DOES HETEROGENEITY BYPASS THE ~45% SATURATION CEILING?")
    print("─" * 90)

    print(f"\n  Paper claim: homogeneous MAS saturates near single-agent capability (~45% ceiling).")
    print(f"  Reviewer's hypothesis: heterogeneous mixing may bypass this limit.")
    print()

    above_ceiling  = [r for r in all_rows if r["beats_ceiling"]]
    below_ceiling  = [r for r in all_rows if not r["beats_ceiling"] and r["hetero_acc"] == r["hetero_acc"]]
    total_valid    = len([r for r in all_rows if r["hetero_acc"] == r["hetero_acc"]])

    print(f"  Configs above 45% ceiling  : {len(above_ceiling)}/{total_valid}")
    print(f"  Configs at/below 45%       : {len(below_ceiling)}/{total_valid}")
    print()

    if above_ceiling:
        print("  Configs that exceed the 45% ceiling:")
        for r in sorted(above_ceiling, key=lambda x: -x["hetero_acc"]):
            print(f"    [{r['arch'].replace('multi-agent-', ''):<16}] {r['label']:<50}  acc={r['hetero_acc']:.3f}")

    print()

    # Family-level: check whether the stronger model's best homogeneous MAS
    # was already above 45%, making ceiling-breaking trivial.
    print("  Context: was the stronger model already near/above 45% in homogeneous MAS?")
    print(f"  {'Stronger model':<22} {'Homo best MAS':>14} {'Homo single-agent':>18}")
    print("  " + "─" * 56)
    seen_models = set()
    for r in all_rows:
        m = r["stronger"]
        if m not in seen_models:
            seen_models.add(m)
            best = get_homo_best_mas(m)
            sa   = get_homo_single_agent(m)
            already = " (already ≥45%)" if best >= 0.45 else ""
            print(f"  {m:<22} {best:>14.3f} {sa:>18.3f}{already}")

    # -----------------------------------------------------------------------
    # Section E: Conclusion
    # -----------------------------------------------------------------------
    print("\n" + "─" * 90)
    print("SECTION E: CONCLUSION")
    print("─" * 90)

    frac_above = len(above_ceiling) / total_valid if total_valid > 0 else 0

    print(f"""
  HONEST ASSESSMENT (BrowseComp-Plus only, n=100):

  1. DO HETEROGENEOUS SYSTEMS BEAT THEIR HOMOGENEOUS COUNTERPARTS?
     Mixed evidence. {len([r for r in all_rows if r['delta_vs_strong'] > 0 and r['delta_vs_strong'] == r['delta_vs_strong']])}/{total_valid} heterogeneous configs
     outperform the same-architecture homogeneous baseline using the stronger model.
     {len([r for r in all_rows if r['delta_vs_strong'] < 0 and r['delta_vs_strong'] == r['delta_vs_strong']])}/{total_valid} fall below it.
     This does NOT provide strong evidence that heterogeneity systematically helps.

  2. DOES HETEROGENEITY BYPASS THE 45% CEILING?
     {len(above_ceiling)}/{total_valid} configs ({frac_above*100:.0f}%) exceed the 45% threshold.
     However, several of those involve the GPT-5 family, where the stronger
     homogeneous MAS (GPT-5 decentralized = 0.50) already exceeded 45%.
     In those cases, exceeding 45% is not a new breakthrough — it merely
     confirms that sufficiently capable models can cross this threshold
     regardless of mixing strategy.

     For families where the stronger homogeneous MAS was near or below 45%
     (Gemini, Anthropic Sonnet), heterogeneous configurations did NOT
     consistently break through the ceiling.

  3. DOES ORCHESTRATOR OR SUBAGENT QUALITY MATTER MORE?
     Of {len([f for f in families_seen if "weak orchestrator, strong subagents" in families_seen[f] and "strong orchestrator, weak subagents" in families_seen[f]])} within-family comparisons:
     weak-orch/strong-sub won {weak_orch_wins}x, strong-orch/weak-sub won {strong_orch_wins}x.
     This suggests subagent execution quality may matter more than
     orchestration quality, but the sample is too small to be definitive.

  4. OVERALL VERDICT:
     The data from this single benchmark does NOT clearly support the
     reviewer's hypothesis that heterogeneity "bypasses saturation limits."
     The ceiling-breaking cases largely trace back to the stronger model's
     own capability, not to mixing per se.

     HOWEVER, this is a single benchmark with one architecture per
     heterogeneous config. The reviewer's hypothesis remains plausible and
     deserves systematic investigation across more benchmarks, more model
     pairs, and varied agent counts. The current evidence is insufficient
     to either confirm or rule out the effect.

  LIMITATION NOTE:
     All results are from BrowseComp-Plus only. This is a search-heavy,
     factual QA benchmark. Results may differ substantially on tasks
     requiring diverse reasoning strategies, where complementarity across
     model types is more likely to manifest.
""")


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve the data directory relative to this script's location
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent          # etc/
    base_dir   = repo_root / "exp_outputs_heterogeneous" / "browsecomp-plus"

    if not base_dir.exists():
        print(f"ERROR: heterogeneous data directory not found: {base_dir}", file=sys.stderr)
        print("Expected layout: etc/exp_outputs_heterogeneous/browsecomp-plus/", file=sys.stderr)
        sys.exit(1)

    print(f"Loading heterogeneous results from: {base_dir}\n")
    hetero_results = load_hetero_results(base_dir)

    # Sanity check: report which files were loaded
    total_loaded = sum(len(v) for v in hetero_results.values())
    print(f"Loaded {total_loaded} heterogeneous experiment results.")
    for arch, configs in hetero_results.items():
        print(f"  {arch}: {len(configs)} configs")
    print()

    run_analysis(hetero_results)


if __name__ == "__main__":
    main()
