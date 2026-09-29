"""Efficacy + safety analysis for a synthetic two-arm clinical trial.

Reproduces the clinical-research reporting pattern I deliver: a governed set of
outcome statistics (effect size, confidence interval, hypothesis tests) plus a
one-page trial-results dashboard. Run after ``generate_data.py`` — outputs land
in ``outputs/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trial.csv"
OUT = ROOT / "outputs"

# ---- House style -----------------------------------------------------------
INK = "#0f172a"
GRID = "#e2e8f0"
TREAT = "#1E90FF"
CTRL = "#7C5CFF"
ACCENT_2 = "#00C2A8"
WARN = "#F2647C"
PALETTE = ["#1E90FF", "#00C2A8", "#F5A524", "#7C5CFF", "#F2647C"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["change"] = df["followup"] - df["baseline"]
    return df


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def efficacy(df: pd.DataFrame) -> dict:
    """Primary endpoint: change from baseline, Treatment vs Placebo.

    Lower change = greater biomarker reduction = better. Uses Welch's t-test
    (does not assume equal variances) plus effect size and a 95% CI on the
    between-arm difference.
    """
    t = df.loc[df["arm"] == "Treatment", "change"].to_numpy()
    p = df.loc[df["arm"] == "Placebo", "change"].to_numpy()
    stat, pval = stats.ttest_ind(t, p, equal_var=False)

    diff = t.mean() - p.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + p.var(ddof=1) / len(p))
    # Welch–Satterthwaite degrees of freedom for the CI critical value
    num = (t.var(ddof=1) / len(t) + p.var(ddof=1) / len(p)) ** 2
    den = ((t.var(ddof=1) / len(t)) ** 2 / (len(t) - 1)
           + (p.var(ddof=1) / len(p)) ** 2 / (len(p) - 1))
    dof = num / den
    crit = stats.t.ppf(0.975, dof)
    return {
        "treatment_mean_change": round(float(t.mean()), 2),
        "placebo_mean_change": round(float(p.mean()), 2),
        "treatment_std": round(float(t.std(ddof=1)), 2),
        "placebo_std": round(float(p.std(ddof=1)), 2),
        "difference": round(float(diff), 2),
        "ci95_low": round(float(diff - crit * se), 2),
        "ci95_high": round(float(diff + crit * se), 2),
        "cohens_d": round(_cohens_d(t, p), 2),
        "t_stat": round(float(stat), 2),
        "p_value": float(pval),
        "significant": bool(pval < 0.05),
        "n_treatment": int(len(t)),
        "n_placebo": int(len(p)),
    }


def safety(df: pd.DataFrame) -> dict:
    """Secondary endpoint: adverse-event rate. Relative risk + chi-square test."""
    tab = pd.crosstab(df["arm"], df["adverse_event"])
    # columns: 0 = no event, 1 = event
    ae_t = int(tab.loc["Treatment", 1])
    ae_p = int(tab.loc["Placebo", 1])
    n_t = int(tab.loc["Treatment"].sum())
    n_p = int(tab.loc["Placebo"].sum())
    rate_t = ae_t / n_t
    rate_p = ae_p / n_p
    chi2, pval, _, _ = stats.chi2_contingency(tab.to_numpy())
    return {
        "ae_rate_treatment_pct": round(100 * rate_t, 1),
        "ae_rate_placebo_pct": round(100 * rate_p, 1),
        "ae_count_treatment": ae_t,
        "ae_count_placebo": ae_p,
        "relative_risk": round(rate_t / rate_p, 2),
        "risk_difference_pct": round(100 * (rate_t - rate_p), 1),
        "chi2": round(float(chi2), 2),
        "p_value": float(pval),
        "significant": bool(pval < 0.05),
    }


def kpis(df: pd.DataFrame) -> dict:
    """Trial-level KPI snapshot — single source of truth for headline numbers."""
    eff = efficacy(df)
    saf = safety(df)
    return {
        "n_patients": int(len(df)),
        "n_treatment": eff["n_treatment"],
        "n_placebo": eff["n_placebo"],
        "mean_age": round(float(df["age"].mean()), 1),
        "pct_female": round(100 * (df["sex"] == "F").mean(), 1),
        "efficacy": eff,
        "safety": saf,
    }


def insights(k: dict) -> list[str]:
    eff, saf = k["efficacy"], k["safety"]
    return [
        f"Treatment reduced the biomarker by **{abs(eff['difference'])} points more** "
        f"than placebo ({eff['treatment_mean_change']} vs {eff['placebo_mean_change']} "
        f"mean change), 95% CI [{eff['ci95_low']}, {eff['ci95_high']}].",
        f"The effect is **large** (Cohen's d = {eff['cohens_d']}) and highly "
        f"significant (Welch t = {eff['t_stat']}, p = {eff['p_value']:.1e}).",
        f"Safety signal is modest: adverse-event rate {saf['ae_rate_treatment_pct']}% "
        f"vs {saf['ae_rate_placebo_pct']}% (relative risk {saf['relative_risk']}), "
        f"{'significant' if saf['significant'] else 'not significant'} "
        f"(chi-square p = {saf['p_value']:.2f}).",
        f"Arms are balanced: {k['n_treatment']} treatment / {k['n_placebo']} placebo, "
        f"mean age {k['mean_age']}, {k['pct_female']}% female.",
    ]


def dashboard(df: pd.DataFrame, k: dict) -> None:
    """One-page trial-results dashboard (4 panels) + standalone hero charts."""
    OUT.mkdir(exist_ok=True)
    eff, saf = k["efficacy"], k["safety"]
    t = df.loc[df["arm"] == "Treatment", "change"]
    p = df.loc[df["arm"] == "Placebo", "change"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Clinical Trial — Efficacy & Safety Dashboard", fontsize=16,
                 fontweight="bold", color=INK, x=0.5, y=0.98)

    # 1) Change-from-baseline distribution by arm
    ax = axes[0, 0]
    bins = np.linspace(min(t.min(), p.min()), max(t.max(), p.max()), 30)
    ax.hist(p, bins=bins, alpha=0.6, color=CTRL, label="Placebo")
    ax.hist(t, bins=bins, alpha=0.6, color=TREAT, label="Treatment")
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_title("Change From Baseline (lower = better)")
    ax.set_xlabel("Δ biomarker")
    ax.legend(frameon=False)

    # 2) Mean change with 95% CI error bars
    ax = axes[0, 1]
    means = [eff["placebo_mean_change"], eff["treatment_mean_change"]]
    sems = [p.std(ddof=1) / np.sqrt(len(p)), t.std(ddof=1) / np.sqrt(len(t))]
    ci = [1.96 * s for s in sems]
    ax.bar(["Placebo", "Treatment"], means, color=[CTRL, TREAT],
           yerr=ci, capsize=8, ecolor=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"Mean Change ± 95% CI  (d = {eff['cohens_d']})")
    ax.set_ylabel("Δ biomarker")

    # 3) Adverse-event rate by arm
    ax = axes[1, 0]
    rates = [saf["ae_rate_placebo_pct"], saf["ae_rate_treatment_pct"]]
    bars = ax.bar(["Placebo", "Treatment"], rates, color=[CTRL, WARN])
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.2, f"{r}%",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Adverse-Event Rate  (RR = {saf['relative_risk']})")
    ax.set_ylabel("% of arm")

    # 4) Effect plot — between-arm difference with CI (forest-style)
    ax = axes[1, 1]
    ax.errorbar([eff["difference"]], [0],
                xerr=[[eff["difference"] - eff["ci95_low"]],
                      [eff["ci95_high"] - eff["difference"]]],
                fmt="o", color=TREAT, ecolor=TREAT, capsize=10, ms=12, lw=2.5)
    ax.axvline(0, color=WARN, lw=1.5, ls="--", label="no effect")
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_title("Treatment Effect (Δ vs placebo) ± 95% CI")
    ax.set_xlabel("Between-arm difference in change")
    ax.text(eff["difference"], 0.25,
            f"{eff['difference']}  [{eff['ci95_low']}, {eff['ci95_high']}]",
            ha="center", fontweight="bold", color=INK)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "dashboard.png", dpi=130)
    plt.close(fig)

    # Standalone hero: change-from-baseline distribution
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(p, bins=bins, alpha=0.6, color=CTRL, label="Placebo")
    ax.hist(t, bins=bins, alpha=0.6, color=TREAT, label="Treatment")
    ax.axvline(t.mean(), color=TREAT, lw=2, ls="--")
    ax.axvline(p.mean(), color=CTRL, lw=2, ls="--")
    ax.set_title("Change From Baseline by Arm — Treatment shifts left (greater reduction)")
    ax.set_xlabel("Δ biomarker (followup − baseline)")
    ax.set_ylabel("patients")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "change_by_arm.png", dpi=120)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python src/generate_data.py first")
    df = load()
    k = kpis(df)
    findings = insights(k)
    dashboard(df, k)

    (OUT / "kpis.json").write_text(json.dumps(k, indent=2))

    eff, saf = k["efficacy"], k["safety"]
    print("=== Enrollment ===")
    print(f"{k['n_patients']:,} patients — {k['n_treatment']} treatment / "
          f"{k['n_placebo']} placebo; mean age {k['mean_age']}, {k['pct_female']}% female")

    print("\n=== Efficacy (primary endpoint: change from baseline) ===")
    print(f"Treatment: {eff['treatment_mean_change']} ± {eff['treatment_std']}")
    print(f"Placebo:   {eff['placebo_mean_change']} ± {eff['placebo_std']}")
    print(f"Difference: {eff['difference']}  95% CI [{eff['ci95_low']}, {eff['ci95_high']}]")
    print(f"Cohen's d: {eff['cohens_d']}  |  Welch t = {eff['t_stat']}, "
          f"p = {eff['p_value']:.2e}  ->  "
          f"{'SIGNIFICANT' if eff['significant'] else 'not significant'}")

    print("\n=== Safety (adverse events) ===")
    print(f"Treatment {saf['ae_rate_treatment_pct']}% vs Placebo "
          f"{saf['ae_rate_placebo_pct']}%  (RR {saf['relative_risk']}, "
          f"chi-square p = {saf['p_value']:.3f})")

    print("\n=== Insights ===")
    for line in findings:
        print(" •", line.replace("**", ""))
    print(f"\nDashboard + charts + kpis.json written to {OUT}/")


if __name__ == "__main__":
    main()
