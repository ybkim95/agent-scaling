#!/usr/bin/env python
"""
Architecture-selection evaluation with complete choice sets held out.

Evaluates how often the fitted scaling model (Equation 1) picks the
best-performing architecture for a (benchmark, model) pair when that entire pair
is unseen during fitting. Selecting an architecture is a decision over a complete
set of alternatives, so the unit of holdout here is the choice set rather than
the individual configuration.

Inputs
------
* data/Source_Data_Fig1_Fig2.csv : the 260 configurations deposited with the
  article (dataset, architecture, family, model, performance).
* The constants of scripts/extended_mixed_effects_6benchmarks.py (coordination
  metrics, Intelligence Index, tool counts, agent counts, Equation 1 formula).

Protocol (stated in full so that every number is reproducible)
--------------------------------------------------------------
Unit of holdout. A choice set is one (benchmark, model) pair and its five
architectures. There are 52 such sets (4 benchmarks x 9 models + 2 benchmarks x
8 models). Each set is held out in turn and Equation 1 is refitted on the
remaining 51 sets (255 rows).

No row of the held-out set enters the fit, and the measured single-agent outcome of
the held-out pair enters only where the protocol below says it does. This is a
conditional evaluation, not one in which every feature is free of held-out
measurement: the architecture-level coordination summaries are trace-derived and are
held fixed at their published values throughout. The standardisation of all
features, the centering of the Intelligence Index, and the dataset-level mean of
single-agent performance that serves as the `single_agent_baseline` feature of
single-agent rows are all recomputed on the training rows only. The released
`build_dataframe` computes that dataset-level mean on all rows; doing so here
would let the held-out set's own single-agent value enter a feature of an option
being predicted, so it is recomputed per fold.

The measured single-agent baseline of the held-out (benchmark, model) pair is a
legitimate input to Equation 1 for the four multi-agent rows (it is the
`single_agent_baseline` covariate, known at decision time) and is used as such.

Capability metric. Both specifications of the article are evaluated. The
Intelligence Index specification is the one tabulated in the article, and the
Agentic Capability Index (ACI), each model's mean single-agent performance
across benchmarks, is the one the article recommends. ACI is recomputed from
the training rows of each fold.

Specifications reported.
  P1  measured single-agent supplied : the single-agent option is scored by its
      measured performance, the four multi-agent options by prediction.
  P2  all five predicted             : all five options scored by prediction.
  P3  ties not credited              : as P1, but a decision is correct only if
      the selected architecture is the unique observed best.
Scoring. A decision is correct when the selected architecture attains the
observed maximum of its set (ties credited, except P3). Regret is the observed
maximum minus the performance of the selected architecture.

Baselines.
  Highest mean performer on the same benchmark : for each held-out set, the
      architecture with the highest mean performance over the other sets of the
      same benchmark. Ties in that mean occur in 4 sets and are broken toward the
      multi-agent option (between multi-agent options, toward the alphabetically
      first name); the alternative of breaking them toward single-agent is also
      reported and leaves the total unchanged.
  Always single-agent                          : a zero-parameter rule.
  Uniform random among five                    : exact expectation. With ties
      credited the chance rate is mean over sets of (number of architectures
      attaining the maximum)/5; with ties not credited it is 1/5 on sets with a
      unique best and 0 otherwise. Random regret is the exact expectation of
      (max - mean of the five).

Confidence intervals are exact (Clopper-Pearson) binomial intervals.

Run `python scripts/architecture_selection_eval.py --check` to assert the
reported values, and `--decisions out.csv` to write the 52 decision-level
outputs, one row per choice set, giving what each rule selected, what that
selection scored, the observed optimum of the set and the resulting regret.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from extended_mixed_effects_6benchmarks import (  # noqa: E402
    AGENT_COUNTS, COORDINATION_METRICS, FEATURE_COLS, FORMULA,
    INTELLIGENCE_SCORES, TOOL_COUNTS, build_dataframe, normalize_model_name,
)

import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SOURCE_DATA = os.path.join(HERE, "..", "data", "Source_Data_Fig1_Fig2.csv")

# The deposited Source Data uses two dataset labels that differ from the
# run_conf names used by the released analysis script.
DATASET_ALIASES = {
    "swe-bench-verified": "swebench-verified",
    "terminal-bench": "terminalbench",
}
DISPLAY = {
    "browsecomp_plus_sampled_100": "BrowseComp-Plus",
    "finance-agent": "Finance Agent",
    "plancraft-test": "PlanCraft",
    "swebench-verified": "SWE-bench Verified",
    "terminalbench": "Terminal-Bench",
    "workbench": "WorkBench",
}
ORDER = list(DISPLAY)
SA = "single-agent"
EPS = 1e-12

# The article's alternative capability metric. ACI replaces the Intelligence
# Index and the three interactions that involve it.
ACI_FORMULA = (FORMULA.replace("intelligence_centered", "aci_centered")
                      .replace("intelligence_sq_centered", "aci_sq_centered")
                      .replace("intel_x_efficiency", "aci_x_efficiency")
                      .replace("msg_density_x_intel", "msg_density_x_aci")
                      .replace("intel_x_tools", "aci_x_tools"))
ACI_COLS = [c for c in FEATURE_COLS if not c.startswith("intel")] + [
    "aci_centered", "aci_sq_centered", "aci_x_efficiency",
    "msg_density_x_aci", "aci_x_tools"]


def add_aci(df, aci_by_model, aci_mean):
    """Attach the ACI features using a mapping estimated on training rows."""
    df = df.copy()
    df["aci"] = df["model_short"].map(aci_by_model)
    if df["aci"].isna().any():
        raise ValueError("a held-out model has no ACI in the training rows")
    df["aci_centered"] = df["aci"] - aci_mean
    df["aci_sq_centered"] = df["aci_centered"] ** 2
    df["aci_x_efficiency"] = df["aci_centered"] * df["efficiency"]
    df["msg_density_x_aci"] = df["message_density"] * df["aci_centered"]
    df["aci_x_tools"] = df["aci_centered"] * np.log1p(df["n_tools"])
    return df


# ----------------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------------
def load_source_data(path=SOURCE_DATA):
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].map(lambda d: DATASET_ALIASES.get(d, d))
    rows = df[["dataset", "architecture", "family", "model", "performance"]].values.tolist()
    if len(rows) != 260:
        raise ValueError(f"expected 260 rows, found {len(rows)}")
    return rows


def featurize(rows, intel_mean=None, sa_mean_by_dataset=None):
    """Mirror of build_dataframe() that accepts training-set statistics.

    With both overrides None this reproduces build_dataframe() exactly (asserted
    in self_check()). During cross-validation the two statistics are supplied
    from the training rows so that nothing from the held-out set enters a
    feature.
    """
    df = pd.DataFrame(rows, columns=["dataset", "architecture", "vendor", "model", "performance"])
    df["model_short"] = df["model"].apply(normalize_model_name)
    df["intelligence"] = df["model_short"].map(INTELLIGENCE_SCORES)
    if df["intelligence"].isna().any():
        raise ValueError(f"unmapped models: {df[df.intelligence.isna()].model_short.unique()}")
    df["n_tools"] = df["dataset"].map(TOOL_COUNTS)
    if df["n_tools"].isna().any():
        raise ValueError(f"unmapped datasets: {df[df.n_tools.isna()].dataset.unique()}")
    df["n_agents"] = df["architecture"].map(AGENT_COUNTS)
    for key in ["overhead_pct", "message_density", "redundancy", "efficiency",
                "error_amplification", "success_per_1k_tokens"]:
        col = key.replace("_tokens", "")
        df[col] = df["architecture"].map({k: v[key] for k, v in COORDINATION_METRICS.items()})

    sa_perf = df[df.architecture == SA][["dataset", "model_short", "performance"]].copy()
    sa_perf.columns = ["dataset", "model_short", "single_agent_baseline"]
    df = df.merge(sa_perf, on=["dataset", "model_short"], how="left")
    if sa_mean_by_dataset is None:
        sa_mean_by_dataset = df[df.architecture == SA].groupby("dataset")["performance"].mean()
    mask = df.architecture == SA
    df.loc[mask, "single_agent_baseline"] = df.loc[mask, "dataset"].map(sa_mean_by_dataset)

    if intel_mean is None:
        intel_mean = df["intelligence"].mean()
    df["intelligence_centered"] = df["intelligence"] - intel_mean
    df["intelligence_sq_centered"] = df["intelligence_centered"] ** 2
    df["log_tools"] = np.log1p(df["n_tools"])
    df["log_agents"] = np.log1p(df["n_agents"])
    df["log_overhead"] = np.log1p(df["overhead_pct"])
    df["log_error_amp"] = np.log1p(df["error_amplification"])
    df["intel_x_efficiency"] = df["intelligence_centered"] * df["efficiency"]
    df["error_x_baseline"] = df["error_amplification"] * df["single_agent_baseline"]
    df["overhead_x_tools"] = df["overhead_pct"] * df["n_tools"]
    df["redundancy_x_agents"] = df["redundancy"] * df["n_agents"]
    df["msg_density_x_intel"] = df["message_density"] * df["intelligence_centered"]
    df["efficiency_x_tools"] = df["efficiency"] * df["n_tools"]
    df["baseline_x_agents"] = df["single_agent_baseline"] * np.log1p(df["n_agents"])
    df["intel_x_tools"] = df["intelligence_centered"] * np.log1p(df["n_tools"])
    df["error_x_tools"] = df["error_amplification"] * df["n_tools"]
    df["g"] = df["dataset"] + "|" + df["model"]
    return df


def self_check(rows):
    """featurize() with no overrides must equal the released build_dataframe()."""
    ref, _ = build_dataframe(rows)
    mine = featurize(rows)
    cols = [c for c in FEATURE_COLS if c in ref.columns]
    diff = np.abs(ref[cols].values - mine[cols].values).max()
    if diff > 1e-9:
        raise AssertionError(f"featurize() deviates from build_dataframe() by {diff}")


# ----------------------------------------------------------------------------
# Leave-one-choice-set-out predictions
# ----------------------------------------------------------------------------
def loco_predictions(rows):
    full = featurize(rows)
    sets = sorted(full["g"].unique())
    pred = pd.Series(np.nan, index=full.index)
    pred_aci = pd.Series(np.nan, index=full.index)
    pred_bm = pd.Series(np.nan, index=full.index)
    bm_base = pd.Series(np.nan, index=full.index)
    raw = pd.DataFrame(rows, columns=["dataset", "architecture", "vendor", "model", "performance"])
    raw["g"] = raw["dataset"] + "|" + raw["model"]
    for g in sets:
        tr_rows = [r for r, gg in zip(rows, raw["g"]) if gg != g]
        te_rows = [r for r, gg in zip(rows, raw["g"]) if gg == g]
        tr_tmp = featurize(tr_rows)
        intel_mean = tr_tmp["intelligence"].mean()
        sa_mean = tr_tmp[tr_tmp.architecture == SA].groupby("dataset")["performance"].mean()
        tr = featurize(tr_rows, intel_mean, sa_mean)
        te = featurize(te_rows, intel_mean, sa_mean)
        scaler = StandardScaler()
        tr[FEATURE_COLS] = scaler.fit_transform(tr[FEATURE_COLS])
        te[FEATURE_COLS] = scaler.transform(te[FEATURE_COLS])
        model = smf.ols(FORMULA, data=tr).fit()
        pred[full.index[full["g"] == g]] = model.predict(te).values

        # ACI specification, with ACI estimated on the training rows only
        aci_map = (tr_tmp[tr_tmp.architecture == SA]
                   .groupby("model_short")["performance"].mean().to_dict())
        aci_mean = float(np.mean(list(aci_map.values())))
        tr_a = add_aci(featurize(tr_rows, intel_mean, sa_mean), aci_map, aci_mean)
        te_a = add_aci(featurize(te_rows, intel_mean, sa_mean), aci_map, aci_mean)
        sc_a = StandardScaler()
        tr_a[ACI_COLS] = sc_a.fit_transform(tr_a[ACI_COLS])
        te_a[ACI_COLS] = sc_a.transform(te_a[ACI_COLS])
        pred_aci[full.index[full["g"] == g]] = smf.ols(ACI_FORMULA, data=tr_a).fit().predict(te_a).values

        # Benchmark-mean baseline throughout: the pair-specific single-agent value is
        # replaced by the training dataset mean in BOTH the training and the test
        # design, so the held-out pair's own single-agent outcome is not used. The
        # architecture-level coordination summaries remain fixed here as elsewhere,
        # so this is not a claim that every feature is free of held-out measurement.
        tr_b = featurize(tr_rows, intel_mean, sa_mean)
        te_b = featurize(te_rows, intel_mean, sa_mean)
        for d_ in (tr_b, te_b):
            d_["single_agent_baseline"] = d_["dataset"].map(sa_mean)
            d_["error_x_baseline"] = d_["error_amplification"] * d_["single_agent_baseline"]
            d_["baseline_x_agents"] = d_["single_agent_baseline"] * np.log1p(d_["n_agents"])
        bm_base[full.index[full["g"] == g]] = te_b["single_agent_baseline"].values
        sc_b = StandardScaler()
        tr_b[FEATURE_COLS] = sc_b.fit_transform(tr_b[FEATURE_COLS])
        te_b[FEATURE_COLS] = sc_b.transform(te_b[FEATURE_COLS])
        pred_bm[full.index[full["g"] == g]] = smf.ols(FORMULA, data=tr_b).fit().predict(te_b).values
    full["pred"] = pred
    full["pred_aci"] = pred_aci
    full["pred_bm"] = pred_bm
    full["bm_baseline_feature"] = bm_base
    return full, sets


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def ci(k, n):
    r = binomtest(int(k), int(n), 0.5).proportion_ci(0.95)
    return r.low, r.high


def score_selection(full, sets, measured_sa, strict, col="pred"):
    out = []
    for g in sets:
        s = full[full.g == g]
        v = s[col].copy()
        if measured_sa:
            m = s.architecture == SA
            v[m[m].index] = s.loc[m, "performance"]
        best = s.performance.max()
        sel = s.loc[v.idxmax(), "performance"]
        attains = sel >= best - EPS
        unique = (s.performance >= best - EPS).sum() == 1
        out.append(dict(dataset=s.dataset.iloc[0], ok=int(attains and (unique or not strict)),
                        regret=float(best - sel)))
    return pd.DataFrame(out)


def score_benchmark_rule(full, sets, prefer_single=False):
    out = []
    for g in sets:
        s = full[full.g == g]
        d = s.dataset.iloc[0]
        others = full[(full.dataset == d) & (full.g != g)]
        means = others.groupby("architecture")["performance"].mean()
        top = sorted(means[means >= means.max() - EPS].index)
        if prefer_single and SA in top:
            pick = SA
        else:
            multi = [a for a in top if a != SA]
            pick = multi[0] if multi else top[0]
        best = s.performance.max()
        sel = s.loc[s.architecture == pick, "performance"].iloc[0]
        out.append(dict(dataset=d, ok=int(sel >= best - EPS), regret=float(best - sel)))
    return pd.DataFrame(out)


def score_always_single(full, sets):
    out = []
    for g in sets:
        s = full[full.g == g]
        best = s.performance.max()
        sel = s.loc[s.architecture == SA, "performance"].iloc[0]
        out.append(dict(dataset=s.dataset.iloc[0], ok=int(sel >= best - EPS), regret=float(best - sel)))
    return pd.DataFrame(out)


def random_baseline(full, sets):
    k_at_max, spread, regret, gap = [], [], [], []
    for g in sets:
        p = full[full.g == g].performance
        k_at_max.append(int((p >= p.max() - EPS).sum()))
        spread.append(float(p.max() - p.min()))
        top2 = sorted(p, reverse=True)[:2]
        gap.append(float(top2[0] - top2[1]))
        regret.append(float(p.max() - p.mean()))
    k = np.array(k_at_max)
    return dict(
        chance_ties_credited=float(np.mean(k / 5)),
        chance_strict=float(np.mean(np.where(k == 1, 0.2, 0.0))),
        regret=float(np.mean(regret)),
        spread=float(np.mean(spread)),
        gap=float(np.mean(gap)),
        n_unique=int((k == 1).sum()), n_two=int((k == 2).sum()), n_three=int((k == 3).sum()),
    )


def _rule_pick_and_best(full, g):
    s = full[full.g == g]
    d = s.dataset.iloc[0]
    means = full[(full.dataset == d) & (full.g != g)].groupby("architecture")["performance"].mean()
    top = sorted(means[means >= means.max() - EPS].index)
    multi = [a for a in top if a != SA]
    pick = multi[0] if multi else top[0]
    return float(s.loc[s.architecture == pick, "performance"].iloc[0]), float(s.performance.max())


def _single_pick_and_best(full, g):
    s = full[full.g == g]
    return float(s.loc[s.architecture == SA, "performance"].iloc[0]), float(s.performance.max())


def mcnemar(a, b):
    a = a.astype(bool).values
    b = b.astype(bool).values
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def decision_table(full, sets):
    """One row per choice set, giving what every rule selected and what it cost.

    This is the decision-level output referred to in the Reply. It is what
    another group needs in order to locate the source of any difference between
    their counts and ours without re-deriving the protocol.
    """
    recs = []
    for g in sets:
        s = full[full.g == g]
        best = float(s.performance.max())
        n_at_max = int((s.performance >= best - EPS).sum())
        rec = dict(benchmark=DISPLAY[s.dataset.iloc[0]], model=s.model.iloc[0],
                   observed_optimum=round(best, 6), n_architectures_at_optimum=n_at_max)
        picks = {
            "eq1_measured_sa": _argmax_arch(s, "pred", measured_sa=True),
            "eq1_all_predicted": _argmax_arch(s, "pred", measured_sa=False),
            "eq1_benchmark_mean_baseline": _argmax_arch(s, "pred_bm", measured_sa=False),
            "benchmark_mean_rule": _rule_arch(full, g),
            "always_single_agent": SA,
        }
        for name, arch in picks.items():
            perf = float(s.loc[s.architecture == arch, "performance"].iloc[0])
            rec[name + "_selected"] = arch
            rec[name + "_performance"] = round(perf, 6)
            rec[name + "_attained"] = int(perf >= best - EPS)
            rec[name + "_regret"] = round(best - perf, 6)
        recs.append(rec)
    return pd.DataFrame(recs)


def option_scores(full, sets):
    """One row per candidate option, giving the score each specification assigned it.

    The per-decision table records the observed performance of the option that was
    selected. That is not enough to explain why two implementations diverge on a
    fold, because it does not expose the quantity the selection was made on. This
    table does. For the measured-single-agent specification the single-agent option
    carries its measured value and the four multi-agent options carry predictions.
    """
    out = full[["g", "dataset", "model", "architecture", "performance",
                "single_agent_baseline", "bm_baseline_feature", "pred", "pred_bm"]].copy()
    out["benchmark"] = out["dataset"].map(DISPLAY)
    sel_measured = out["pred"].copy()
    m = out["architecture"] == SA
    sel_measured[m] = out.loc[m, "performance"]
    out["score_eq1_measured_sa"] = sel_measured
    out = out.rename(columns={"g": "choice_set",
                              "pred": "score_eq1_all_predicted",
                              "pred_bm": "score_eq1_benchmark_mean_baseline",
                              "single_agent_baseline": "baseline_feature_pair_specific",
                              "bm_baseline_feature": "baseline_feature_benchmark_mean"})
    out["is_observed_optimum"] = out.groupby("choice_set")["performance"].transform(
        lambda p: (p >= p.max() - EPS).astype(int))
    cols = ["choice_set", "benchmark", "model", "architecture", "performance",
            "is_observed_optimum", "score_eq1_measured_sa", "score_eq1_all_predicted",
            "score_eq1_benchmark_mean_baseline", "baseline_feature_pair_specific",
            "baseline_feature_benchmark_mean"]
    return out[cols].sort_values(["choice_set", "architecture"]).reset_index(drop=True)


def _argmax_arch(s, col, measured_sa):
    v = s[col].copy()
    if measured_sa:
        m = s.architecture == SA
        v[m[m].index] = s.loc[m, "performance"]
    return s.loc[v.idxmax(), "architecture"]


def _rule_arch(full, g):
    s = full[full.g == g]
    d = s.dataset.iloc[0]
    means = full[(full.dataset == d) & (full.g != g)].groupby("architecture")["performance"].mean()
    top = sorted(means[means >= means.max() - EPS].index)
    multi = [a for a in top if a != SA]
    return multi[0] if multi else top[0]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(check=False, decisions=None):
    rows = load_source_data()
    self_check(rows)
    full, sets = loco_predictions(rows)
    n = len(sets)

    P1 = score_selection(full, sets, measured_sa=True, strict=False)
    P2 = score_selection(full, sets, measured_sa=False, strict=False)
    P3 = score_selection(full, sets, measured_sa=True, strict=True)
    A1 = score_selection(full, sets, measured_sa=True, strict=False, col="pred_aci")
    A2 = score_selection(full, sets, measured_sa=False, strict=False, col="pred_aci")
    A3 = score_selection(full, sets, measured_sa=True, strict=True, col="pred_aci")
    BM = score_selection(full, sets, measured_sa=False, strict=False, col="pred_bm")
    RULE = score_benchmark_rule(full, sets)
    RULE_SA = score_benchmark_rule(full, sets, prefer_single=True)
    ALWAYS = score_always_single(full, sets)
    RND = random_baseline(full, sets)

    if decisions is not None:
        tab = decision_table(full, sets)
        assert len(tab) == 52, len(tab)
        tab.to_csv(decisions, index=False, float_format="%.10g")
        print(f"wrote {len(tab)} decision-level rows to {decisions}")
        opt_path = decisions.replace(".csv", "_option_scores.csv")
        if opt_path == decisions:
            opt_path = decisions + ".option_scores.csv"
        opts = option_scores(full, sets)
        assert len(opts) == 260, len(opts)
        opts.to_csv(opt_path, index=False, float_format="%.10g")
        print(f"wrote {len(opts)} option-level rows to {opt_path}")

    def line(label, R):
        k = int(R.ok.sum())
        lo, hi = ci(k, n)
        return f"| {label:46s} | {k:2d} of {n} | {k/n*100:5.1f}% | {lo*100:4.1f} to {hi*100:4.1f} | {R.regret.mean():.4f} |"

    print("REPLY TABLE 1a. Architecture selection with complete choice sets held out,")
    print("260 configurations, 52 decisions. These rows, and only these rows, are the")
    print("Reply's Table 1a. The capability metric throughout is the Intelligence Index,")
    print("which is the specification the article tabulates.")
    print("| Selection procedure | Correct | Accuracy | 95% CI | Mean regret |")
    print("|---|---|---|---|---|")
    print(line("Intelligence Index, measured single-agent supplied", P1))
    print(line("Intelligence Index, all five options predicted", P2))
    print(line("Benchmark-mean baseline throughout, all five predicted", BM))
    print(line("Intelligence Index, ties not credited", P3))
    print(line("Highest mean performer on the same benchmark", RULE))
    print(line("Always single-agent", ALWAYS))
    print(f"| {'Uniform random among five, ties credited':46s} |          | {RND['chance_ties_credited']*100:5.1f}% |              | {RND['regret']:.4f} |")
    print(f"  chance rate with ties not credited: {RND['chance_strict']*100:.1f}%")
    print()
    print("ADDITIONAL SENSITIVITY, NOT IN THE REPLY. The Agentic Capability Index is the")
    print("article's alternative capability metric. These rows are reported for")
    print("completeness only and no claim in the Reply rests on them.")
    print("| Selection procedure | Correct | Accuracy | 95% CI | Mean regret |")
    print("|---|---|---|---|---|")
    print(line("ACI metric, measured single-agent supplied", A1))
    print(line("ACI metric, all five options predicted", A2))
    print(line("ACI metric, ties not credited", A3))
    print()

    print("Table 1b. Decisions attained per benchmark")
    print("| Benchmark | Sets | ACI, measured SA | Intelligence Index, measured SA | Benchmark mean rule |")
    print("|---|---|---|---|---|")
    per = {}
    for d in ORDER:
        c = lambda R: int(R[R.dataset == d].ok.sum())
        per[d] = (int((P1.dataset == d).sum()), c(A1), c(P1), c(RULE))
        print(f"| {DISPLAY[d]:18s} | {per[d][0]} | {per[d][1]} | {per[d][2]} | {per[d][3]} |")
    print(f"| {'Total':18s} | {n} | {int(A1.ok.sum())} | {int(P1.ok.sum())} | {int(RULE.ok.sum())} |")
    print()

    print("Notes")
    print(f"  sets with a unique best / two-way tie / three-way tie: {RND['n_unique']} / {RND['n_two']} / {RND['n_three']}")
    print(f"  mean within-set spread (best minus worst): {RND['spread']:.4f}")
    print(f"  mean gap between best and second-best architecture: {RND['gap']:.4f}")
    print(f"  benchmark mean rule with ties broken toward single-agent: {int(RULE_SA.ok.sum())} of {n}, mean regret {RULE_SA.regret.mean():.4f}")
    dt = decision_table(full, sets)
    diff_pick = int((dt["eq1_measured_sa_selected"] != dt["benchmark_mean_rule_selected"]).sum())
    print(f"  equation 1 and the benchmark mean rule select different architectures in "
          f"{diff_pick} of {n} sets")
    mc = {}
    for lab, R in [("ACI, measured SA", A1), ("Intelligence Index, measured SA", P1)]:
        mc[lab] = mcnemar(R.ok, RULE.ok)
        a_, b_, p_ = mc[lab]
        print(f"  exact McNemar, {lab} vs benchmark mean rule: model-only {a_}, rule-only {b_}, discordant {a_+b_}, P = {p_:.4f}")
    print("  note: selecting a different architecture and differing in outcome are distinct. "
          f"The counts are {diff_pick} and {sum(mc['Intelligence Index, measured SA'][:2])}.")
    mc_alw = mcnemar(P1.ok, ALWAYS.ok)
    print(f"  exact McNemar, Intelligence Index, measured SA vs always single-agent: "
          f"model-only {mc_alw[0]}, rule-only {mc_alw[1]}, discordant {mc_alw[0]+mc_alw[1]}, "
          f"P = {mc_alw[2]:.4f}")
    npc = A1[A1.dataset != "plancraft-test"]
    npr = RULE[RULE.dataset != "plancraft-test"]
    print(f"  excluding PlanCraft: ACI {int(npc.ok.sum())} of {len(npc)}, benchmark mean rule {int(npr.ok.sum())} of {len(npr)}")
    # like-for-like: the comparison rules scored under the strict convention
    rule_s = int(sum(1 for g in sets
                     if (lambda t: t[0] >= t[1] - EPS and (full[full.g == g].performance >= t[1] - EPS).sum() == 1)
                        (_rule_pick_and_best(full, g))))
    alw_s = int(sum(1 for g in sets
                    if (lambda t: t[0] >= t[1] - EPS and (full[full.g == g].performance >= t[1] - EPS).sum() == 1)
                       (_single_pick_and_best(full, g))))
    print(f"  ties not credited, comparison rules scored the same way: "
          f"benchmark mean rule {rule_s} of {n}, always single-agent {alw_s} of {n}")
    for lab, R, p0 in [("ACI, measured SA", A1, RND["chance_ties_credited"]),
                       ("ACI, ties not credited", A3, RND["chance_strict"])]:
        print(f"  above chance, {lab}: one-sided exact binomial P = "
              f"{binomtest(int(R.ok.sum()), n, p0, alternative='greater').pvalue:.2e}")

    if check:
        exp = {
            "A1": (31, 0.0379), "A2": (21, 0.0508), "A3": (26, 0.0379),
            "BM": (22, 0.0562),
            "P1": (23, 0.0535), "P2": (22, 0.0717), "P3": (19, 0.0535),
            "RULE": (27, 0.0489), "ALWAYS": (22, 0.0960),
        }
        got = {"A1": A1, "A2": A2, "A3": A3, "BM": BM, "P1": P1, "P2": P2, "P3": P3, "RULE": RULE, "ALWAYS": ALWAYS}
        for k_, (kk, rr) in exp.items():
            assert int(got[k_].ok.sum()) == kk, (k_, int(got[k_].ok.sum()))
            assert abs(got[k_].regret.mean() - rr) < 5e-4, (k_, got[k_].regret.mean())
        assert abs(RND["chance_ties_credited"] - 0.2538) < 5e-4
        assert abs(RND["chance_strict"] - 0.1577) < 5e-4
        assert abs(RND["regret"] - 0.1116) < 5e-4
        assert abs(RND["spread"] - 0.2465) < 5e-4
        assert abs(RND["gap"] - 0.0646) < 5e-4, RND["gap"]
        assert (RND["n_unique"], RND["n_two"], RND["n_three"]) == (41, 8, 3)
        assert per["browsecomp_plus_sampled_100"] == (9, 4, 1, 2)
        assert per["finance-agent"] == (9, 2, 2, 7)
        assert per["plancraft-test"] == (9, 9, 8, 9)
        assert per["swebench-verified"] == (8, 4, 4, 0)
        assert per["terminalbench"] == (8, 4, 4, 4)
        assert per["workbench"] == (9, 8, 4, 5)
        assert mc["ACI, measured SA"][:2] == (15, 11), mc["ACI, measured SA"]
        assert abs(mc["ACI, measured SA"][2] - 0.5572) < 5e-4, mc["ACI, measured SA"]
        assert mc["Intelligence Index, measured SA"][:2] == (9, 13), mc["Intelligence Index, measured SA"]
        assert abs(mc["Intelligence Index, measured SA"][2] - 0.5235) < 5e-4
        assert (int(npc.ok.sum()), len(npc), int(npr.ok.sum())) == (22, 43, 18)
        for key, lo_hi in {"A1": (45.1, 73.0), "A2": (27.0, 54.9), "A3": (35.8, 64.2),
                           "P1": (30.5, 58.7), "P2": (28.7, 56.8), "P3": (23.6, 51.0),
                           "RULE": (37.6, 66.0), "ALWAYS": (28.7, 56.8)}.items():
            lo, hi = ci(int(got[key].ok.sum()), n)
            assert abs(lo * 100 - lo_hi[0]) < 0.05 and abs(hi * 100 - lo_hi[1]) < 0.05, (key, lo, hi)
        for key, p0, cap in [("A1", RND["chance_ties_credited"], 1e-4),
                             ("A3", RND["chance_strict"], 1e-4)]:
            pv = binomtest(int(got[key].ok.sum()), n, p0, alternative="greater").pvalue
            assert pv < cap, (key, pv)
        assert int(RULE_SA.ok.sum()) == 27 and abs(RULE_SA.regret.mean() - 0.0537) < 5e-4
        assert (rule_s, alw_s) == (21, 18), (rule_s, alw_s)
        assert diff_pick == 35, diff_pick
        assert mc_alw[:2] == (10, 9), mc_alw
        assert abs(mc_alw[2] - 1.0) < 1e-12, mc_alw
        print("\nCHECK PASSED: all reported values reproduce.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert the reported values")
    ap.add_argument("--decisions", metavar="PATH",
                    help="write the 52 decision-level outputs to PATH as CSV")
    a = ap.parse_args()
    main(check=a.check, decisions=a.decisions)
