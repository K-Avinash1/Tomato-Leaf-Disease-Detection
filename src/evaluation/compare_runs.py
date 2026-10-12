import argparse

import pandas as pd
from scipy.stats import binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="val_predictions.csv of the reference run")
    ap.add_argument("--b", required=True, help="val_predictions.csv of the run being compared")
    a = ap.parse_args()

    A = pd.read_csv(a.a).set_index("rel_path")
    B = pd.read_csv(a.b).set_index("rel_path")
    assert list(A.index) == list(B.index), "the two files must cover the same validation images"
    ok_a = A["true_idx"] == A["pred_idx"]
    ok_b = B["true_idx"] == B["pred_idx"]
    a_only = int((ok_a & ~ok_b).sum())     # reference right, other wrong
    b_only = int((~ok_a & ok_b).sum())     # other right, reference wrong
    both_wrong = int((~ok_a & ~ok_b).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    print(f"Errors: reference {int((~ok_a).sum())} | other {int((~ok_b).sum())}")
    print(f"Only reference right: {a_only} | only other right: {b_only} | both wrong: {both_wrong}")
    print(f"Exact McNemar p-value (two-sided): {p:.4f}")
    diff = pd.DataFrame({"true": A["class"], "pred_reference": A["pred_class"], "pred_other": B["pred_class"]})
    print("\nImages where the two runs differ in correctness:")
    print(diff[ok_a != ok_b].to_string())


if __name__ == "__main__":
    main()