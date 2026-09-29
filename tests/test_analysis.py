"""Invariant + statistical-sanity tests for the trial analysis."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate_data  # noqa: E402
import analyze  # noqa: E402


@pytest.fixture(scope="module")
def df():
    raw = generate_data.generate(n=1500)
    raw["change"] = raw["followup"] - raw["baseline"]
    return raw


def test_schema(df):
    expected = {"patient_id", "arm", "age", "sex",
                "baseline", "followup", "adverse_event"}
    assert expected.issubset(df.columns)


def test_arms_and_balance(df):
    assert set(df["arm"].unique()) == {"Treatment", "Placebo"}
    # both arms should be non-trivially populated
    counts = df["arm"].value_counts()
    assert counts.min() > 0.3 * len(df)


def test_age_bounds(df):
    assert df["age"].between(18, 90).all()


def test_adverse_event_binary(df):
    assert set(df["adverse_event"].unique()).issubset({0, 1})


def test_efficacy_direction_and_significance(df):
    eff = analyze.efficacy(df)
    # treatment must reduce the biomarker more than placebo
    assert eff["treatment_mean_change"] < eff["placebo_mean_change"]
    assert eff["difference"] < 0
    # a large seeded effect must clear significance
    assert eff["significant"]
    assert eff["p_value"] < 0.05
    # CI must exclude zero and bracket the point estimate
    assert eff["ci95_low"] < eff["difference"] < eff["ci95_high"]
    assert eff["ci95_high"] < 0


def test_effect_size_is_large(df):
    eff = analyze.efficacy(df)
    assert abs(eff["cohens_d"]) >= 0.8  # Cohen's convention for "large"


def test_safety_rates_bounded(df):
    saf = analyze.safety(df)
    for key in ("ae_rate_treatment_pct", "ae_rate_placebo_pct"):
        assert 0 <= saf[key] <= 100
    assert saf["relative_risk"] > 0


def test_kpis_shape(df):
    k = analyze.kpis(df)
    assert k["n_patients"] == len(df)
    assert k["n_treatment"] + k["n_placebo"] == len(df)
    assert 0 <= k["pct_female"] <= 100


def test_insights_render(df):
    k = analyze.kpis(df)
    lines = analyze.insights(k)
    assert len(lines) == 4
    assert all(isinstance(s, str) and s for s in lines)
