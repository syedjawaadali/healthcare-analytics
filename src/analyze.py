"""Analyze trial outcomes: efficacy, safety, and a t-test between arms."""
from pathlib import Path
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data" / "trial.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python src/generate_data.py first")
    df = pd.read_csv(DATA)
    df["change"] = df["followup"] - df["baseline"]

    print("=== Enrollment ===")
    print(df["arm"].value_counts().to_string())

    print("\n=== Mean change from baseline (lower = better) ===")
    print(df.groupby("arm")["change"].agg(["mean", "std", "count"]).round(2))

    print("\n=== Adverse-event rate by arm ===")
    print((df.groupby("arm")["adverse_event"].mean() * 100).round(1)
          .astype(str).add(" %").to_string())

    t = df[df.arm == "Treatment"]["change"]
    p = df[df.arm == "Placebo"]["change"]
    stat, pval = stats.ttest_ind(t, p, equal_var=False)
    print("\n=== Efficacy test (Welch t-test on change) ===")
    print(f"t = {stat:.2f},  p-value = {pval:.2e}")
    print("Result:", "statistically significant (p < 0.05)"
          if pval < 0.05 else "not significant")


if __name__ == "__main__":
    main()
