from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bdc_coverage_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
BLUE = "#08457E"
ORANGE = "#F28E00"


def esc(value):
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def rtf_row(cells, widths, *, bold=False, top=False, bottom=False, mid=False):
    out = [r"\trowd\trgaph80\trleft0"]
    if top: out.append(r"\trbrdrt\brdrdb\brdrw14")
    if mid: out.append(r"\trbrdrb\brdrs\brdrw8")
    if bottom: out.append(r"\trbrdrb\brdrdb\brdrw14")
    end = 0
    for width in widths:
        end += width
        out.append(fr"\cellx{end}")
    for i, value in enumerate(cells):
        out.append((r"\intbl\ql " if i == 0 else r"\intbl\qc ") + (r"\b " if bold else "") + esc(value) + (r"\b0" if bold else "") + r"\cell")
    out.append(r"\row")
    return "".join(out)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    robust = pd.read_csv(DATA / "h3_robustness_scenarios.csv")
    years = pd.read_csv(DATA / "h3_year_summary.csv")
    panel = pd.read_csv(DATA / "h3_analysis_panel.csv")
    placebo = pd.read_csv(DATA / "h3_market_placebo_and_nav_sensitivity.csv")
    linkage = pd.read_csv(DATA / "h2_h3_linkage_summary.csv")
    headers = ["", "(1)", "(2)", "(3)", "(4)", "(5)"]
    rows = [
        headers,
        ["Mean buffer compression (pp)", *[f"{x:.3f}" for x in robust.mean_buffer_shrink_pp]],
        ["Median buffer compression (pp)", *[f"{x:.3f}" for x in robust.median_buffer_shrink_pp]],
        ["LRMES measure", "Monthly mean 50%", "Empirical p01", "Month-end 50%", "Beta-implied 50%", "Positive-loss 50%"],
        ["Positive compression share", *[f"{x:.3f}" for x in robust.positive_shrink_share]],
        ["Within 10 pp after stress", *[str(int(x)) for x in robust.within_10pp_after]],
        ["Within 25 pp after stress", *[str(int(x)) for x in robust.within_25pp_after]],
        ["Legal breaches after stress", *[str(int(x)) for x in robust.breach_observations_after]],
        ["Observations", *[str(int(x)) for x in robust.observations]],
        ["BDC clusters", *[str(int(x)) for x in robust.firms]],
    ]
    note = "This table reports calibrated reductions in the distance between each BDC's reported asset-coverage ratio and its applicable 150% or 200% statutory threshold. Column (1) uses the maintained 50% six-month climate-factor decline; Column (2) uses the sample's 34.1% empirical first-percentile decline. Because positive beta mechanically implies positive compression, the table emphasizes magnitudes and threshold proximity rather than significance stars."
    body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs20\landscape\paperw15840\paperh12240\margl700\margr700\margt750\margb750", r"\pard\sa40\b\fs24 Table 5: Climate Stress and BDC Asset-Coverage Capacity.\b0\par", r"\pard\sa110\fs18 " + esc(note) + r"\par"]
    for i, cells in enumerate(rows):
        body.append(rtf_row(cells, [3300] + [1800] * 5, bold=i == 0, top=i == 0, mid=i == 0, bottom=i == len(rows) - 1))
    body += [r"\pard\sa120\fs18 " + esc("Notes: " + note) + r"\par", "}"]
    (RESULTS / "Table_5_BDC_Asset_Coverage_Stress.rtf").write_text("\n".join(body), encoding="ascii", errors="ignore")

    order = [
        "Climate shock, market equity mapping",
        "Empirical climate 1pct tail (34.1pct decline)",
        "Empirical market 1pct tail (16.8pct decline)",
        "50 percent broad-market shock placebo",
        "Climate shock, NAV mapping",
    ]
    placebo = placebo.set_index("scenario").loc[order].reset_index()
    compare_rows = [
        ["", "Climate 50%", "Climate p01", "Market p01", "Market 50%", "Climate / NAV"],
        ["Mean buffer compression (pp)", *[f"{x:.3f}" for x in placebo.mean_buffer_shrink_pp]],
        ["Median buffer compression (pp)", *[f"{x:.3f}" for x in placebo.median_buffer_shrink_pp]],
        ["Mean of firm-level buffer shares (%)", *[f"{x:.2f}" for x in placebo.mean_buffer_consumed_pct]],
        ["Ratio of mean compression to mean buffer (%)", *[f"{x:.2f}" for x in placebo.ratio_of_mean_compression_to_mean_buffer_pct]],
        ["Legal breaches", *[str(int(x)) for x in placebo.breach_observations_after]],
        ["Observations", *[str(int(x)) for x in placebo.observations]],
    ]
    matched_ratio = placebo.loc[placebo.scenario.str.startswith("Empirical climate"), "mean_buffer_shrink_pp"].iloc[0] / placebo.loc[placebo.scenario.str.startswith("Empirical market"), "mean_buffer_shrink_pp"].iloc[0]
    compare_note = "Climate p01 and Market p01 use each factor's empirical first-percentile six-month return over 2010-2025: -34.1% and -16.8%, respectively. Their mean-compression ratio is " + f"{matched_ratio:.3f}. " + "The mean of firm-level ratios is distinct from the ratio of sample means; both are reported. The NAV column replaces market equity only for the maintained 50% climate shock."
    compare_body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs20\landscape\paperw15840\paperh12240\margl620\margr620\margt720\margb720", r"\pard\sa40\b\fs24 Table 6: Matched-Tail Market Comparison, NAV Sensitivity, and the H2-H3 Link.\b0\par", r"\pard\sa110\fs18 " + esc(compare_note) + r"\par"]
    for i, cells in enumerate(compare_rows):
        compare_body.append(rtf_row(cells, [3900] + [1950] * 5, bold=i == 0, top=i == 0, mid=i == 0, bottom=i == len(compare_rows) - 1))
    link = linkage.iloc[0]
    compare_body += [r"\pard\sa120\fs18 " + esc("H2-to-H3 linkage: the H2 point estimate implies a zero-to-maximum carbon-intensive investment-share beta increment of " + f"{link.predicted_beta_increment_zero_to_max:.4f}" + " and a representative incremental buffer compression of " + f"{link.predicted_buffer_compression_zero_to_max_pp:.3f}" + " percentage points. This is a mechanical mapping of an imprecise coefficient, not a causal estimate.") + r"\par", "}"]
    (RESULTS / "Table_6_Matched_Tail_NAV_and_H2_H3_Linkage.rtf").write_text("\n".join(compare_body), encoding="ascii", errors="ignore")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(years.year, years.mean_baseline_coverage_pct, color=BLUE, lw=2.3, marker="o", label="Reported coverage")
    ax.plot(years.year, years.mean_stressed_coverage_pct, color=ORANGE, lw=2.3, marker="o", label="After climate stress")
    ax.fill_between(years.year, years.mean_stressed_coverage_pct, years.mean_baseline_coverage_pct, color=ORANGE, alpha=.12)
    ax.set_ylabel("Mean asset coverage (%)")
    ax.set_xlabel("Year")
    ax.set_xticks(years.year)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_6_Asset_Coverage_Before_After.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    primary = panel[panel.actual_asset_coverage_pct.notna()].copy()
    cutoffs = np.array([5, 10, 15, 20, 25, 30])
    before = [(primary.baseline_buffer_pp <= c).sum() for c in cutoffs]
    after = [(primary.stressed_buffer_pp <= c).sum() for c in cutoffs]
    x = np.arange(len(cutoffs))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = .38
    ax.bar(x - width / 2, before, width, color=BLUE, label="Before stress")
    ax.bar(x + width / 2, after, width, color=ORANGE, label="After stress")
    ax.set_xticks(x, [f"≤{c} pp" for c in cutoffs])
    ax.set_ylabel("Firm-quarter observations")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_7_Threshold_Proximity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
