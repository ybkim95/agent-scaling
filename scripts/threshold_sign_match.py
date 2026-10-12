#!/usr/bin/env python
"""
Sign-match check for the ~45% capability-saturation threshold.

The article reports a sign-match rate for the ~45% threshold on the 16 SWE-bench
Verified and Terminal-Bench model x benchmark configurations. This script computes
it from the deposited Source Data.

Definition
----------
For each of the 16 (benchmark, model) pairs on SWE-bench Verified and
Terminal-Bench:
  * P_SA  = measured single-agent performance
  * gain  = (best of the four multi-agent architectures) - P_SA
  * the threshold predicts a negative gain when P_SA >= 0.45 and a positive gain
    otherwise
  * a configuration matches when sign(gain) agrees with the prediction.
With this definition the match is 15 of 16. The 16 pairs are among the 260
configurations analysed in the article.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA = os.path.join(HERE, "..", "data", "Source_Data_Fig1_Fig2.csv")
BENCHMARKS = ["swe-bench-verified", "terminal-bench"]
THRESHOLD = 0.45


def main():
    df = pd.read_csv(SOURCE_DATA)
    sub = df[df.dataset.isin(BENCHMARKS)]
    rows = []
    for (d, m), g in sub.groupby(["dataset", "model"]):
        sa = g[g.architecture == "single-agent"].performance.iloc[0]
        mas = g[g.architecture != "single-agent"].performance
        rows.append(dict(dataset=d, model=m, P_SA=sa, gain_best=mas.max() - sa))
    R = pd.DataFrame(rows)
    assert len(R) == 16, len(R)

    print(R.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    pred_neg = R.P_SA >= THRESHOLD
    match = ((R.gain_best < 0) == pred_neg).sum()
    print(f"  P_SA >= {THRESHOLD:.2f}, gain vs best of four: {match} of 16 = {match/16*100:.1f}%")
    assert match == 15, match


def sensitivity():
    """Sign agreement across fixed cut-offs, on all 52 pairs and on the 16 pairs."""
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import numpy as np
    from architecture_selection_eval import load_source_data, featurize

    df = featurize(load_source_data())

    # The cut-off that maximises sign agreement over the same data (descriptive).
    print("\nThreshold that best separates positive from negative gains, all 52 pairs")
    allrows = []
    for (d_, m_), g in df.groupby(["dataset", "model"]):
        sa_ = g[g.architecture == "single-agent"].performance.iloc[0]
        mas_ = g[g.architecture != "single-agent"].performance
        allrows.append(dict(P_SA=sa_, gain_best=mas_.max() - sa_))
    A = pd.DataFrame(allrows)
    assert len(A) == 52, len(A)
    grid = [round(x, 2) for x in np.arange(0.20, 0.81, 0.01)]
    counts = {t: int(((A.gain_best < 0) == (A.P_SA >= t)).sum()) for t in grid}
    top = max(counts.values())
    argmax = [t for t, k in counts.items() if k == top]
    print(f"  maximising threshold {', '.join(f'{t:.2f}' for t in argmax)} "
          f"at {top} of 52, and 0.40 gives {counts[0.40]} of 52")
    assert argmax == [0.41] and top == 40, (argmax, top)
    assert counts[0.40] == 39, counts[0.40]

    print("\nSign-match on the 16 configurations across the threshold range")
    sub = pd.read_csv(SOURCE_DATA)
    sub = sub[sub.dataset.isin(BENCHMARKS)]
    rows = []
    for (d_, m_), g in sub.groupby(["dataset", "model"]):
        sa = g[g.architecture == "single-agent"].performance.iloc[0]
        mas = g[g.architecture != "single-agent"].performance
        rows.append(dict(P_SA=sa, gain_best=mas.max() - sa, gain_mean=mas.mean() - sa))
    R = pd.DataFrame(rows)
    # Gain of the best of the four multi-agent architectures, as in the article, and of their
    # mean, reported alongside.
    for t in [0.40, 0.41, 0.43, 0.45]:
        kb = int(((R.gain_best < 0) == (R.P_SA >= t)).sum())
        km = int(((R.gain_mean < 0) == (R.P_SA >= t)).sum())
        print(f"  threshold {t:.4f}: best-of-four {kb} of 16, mean-of-four {km} of 16")
    assert all(int(((R.gain_best < 0) == (R.P_SA >= t)).sum()) == 15
               for t in [0.41, 0.43, 0.45]), "sign-match not stable over 0.41-0.45"
    assert all(int(((R.gain_mean < 0) == (R.P_SA >= t)).sum()) == 13
               for t in [0.41, 0.43, 0.45]), "mean-endpoint rate not stable over 0.41-0.45"
    print("  stable at 15 of 16 throughout 0.41 to 0.45 on the best-of-four endpoint, "
          "and at 13 of 16 on the mean-of-four endpoint")


if __name__ == "__main__":
    import sys
    main()
    if "--sensitivity" in sys.argv:
        sensitivity()
