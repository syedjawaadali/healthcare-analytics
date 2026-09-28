"""Generate a synthetic clinical-trial dataset (two treatment arms)."""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(2026)
DATA = Path(__file__).resolve().parents[1] / "data"


def generate(n: int = 1200) -> pd.DataFrame:
    arm = RNG.choice(["Treatment", "Placebo"], n)
    age = RNG.normal(55, 12, n).clip(18, 90).round(0)
    sex = RNG.choice(["F", "M"], n)
    baseline = RNG.normal(150, 20, n)          # e.g. baseline biomarker
    effect = np.where(arm == "Treatment", -18, -4)
    followup = baseline + effect + RNG.normal(0, 12, n)
    adverse = RNG.uniform(size=n) < np.where(arm == "Treatment", 0.14, 0.10)
    return pd.DataFrame({
        "patient_id": range(1, n + 1), "arm": arm, "age": age, "sex": sex,
        "baseline": baseline.round(1), "followup": followup.round(1),
        "adverse_event": adverse.astype(int)})


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    df = generate()
    out = DATA / "trial.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} patients -> {out}")
