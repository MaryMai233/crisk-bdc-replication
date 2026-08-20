from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bdc_results_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
BLUE = "#08457E"
ORANGE = "#F28E00"


def esc(value):
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def rtf_row(cells, widths, *, bold=False, top=False, bottom=False, mid=False):
    out = [r"\trowd\trgaph70\trleft0"]
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
    models = pd.read_csv(DATA / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    panel = pd.read_csv(DATA / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv")
    ids = ["H2_1", "H2_2", "H2_3", "H2_4", "H2_5", "H2_6"]
    m = models.loc[ids]
    coef = []
    for _, x in m.iterrows():
        star = "***" if x.p_two_sided < .01 else "**" if x.p_two_sided < .05 else "*" if x.p_two_sided < .10 else ""
        coef.append(f"{x.coefficient_exposure:.3f}{star}")
    rows = [
        ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
        ["Broad carbon-intensive investment share", *coef],
        ["", *[f"({x:.3f})" for x in m.standard_error]],
        ["Outcome", "Equity beta", "Asset beta", "Equity beta", "Asset beta", "Equity beta", "Asset beta"],
        ["Financial controls", "No", "No", "Yes", "Yes", "No", "No"],
        ["BDC fixed effects", "No", "No", "No", "No", "Yes", "Yes"],
        ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
        ["Observations", *[str(int(x)) for x in m.n]],
        ["R-squared", *[f"{x:.3f}" for x in m.r_squared]],
    ]
    note = "The table relates standardized disclosed broad carbon-intensive investment shares to standardized BDC climate beta. Standard errors are clustered by BDC and appear in parentheses. Financial controls are size, leverage, ROA, book-to-market, and market beta. ***, **, and * denote two-sided significance at 1%, 5%, and 10%."
    body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs22\landscape\paperw15840\paperh12240\margl720\margr720\margt900\margb900", r"\pard\sa40\b\fs24 Table 2: BDC Investment Exposure and Climate Beta.\b0\par"]
    for i, cells in enumerate(rows):
        body.append(rtf_row(cells, [3200] + [1800] * 6, bold=i == 0, top=i == 0, mid=i == 0, bottom=i == len(rows) - 1))
    body += [r"\pard\sa120\fs18 " + esc("Notes: " + note) + r"\par", "}"]
    body[1] = r"\pard\sa40\b\fs24 Table 3: BDC Investment Exposure and Climate Beta.\b0\par"
    (RESULTS / "Table_3_BDC_Investment_Exposure.rtf").write_text("\n".join(body), encoding="ascii", errors="ignore")

    credit_path = DATA / "h2_credit_return_robustness_models.csv"
    if credit_path.exists():
        credit = pd.read_csv(credit_path).set_index("model_id").loc[["C1", "C2", "C3", "C4", "C5", "C6"]]
        credit_coef = []
        for _, item in credit.iterrows():
            star = "***" if item.p_two_sided < .01 else "**" if item.p_two_sided < .05 else "*" if item.p_two_sided < .10 else ""
            credit_coef.append(f"{item.coefficient_exposure:.3f}{star}")
        credit_rows = [
            ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
            ["Broad carbon-intensive investment share", *credit_coef],
            ["", *[f"({x:.3f})" for x in credit.standard_error]],
            ["Outcome", "Equity beta", "Equity beta", "Asset beta", "Asset beta", "Equity beta", "Equity beta"],
            ["High-yield factor", "No", "Yes", "No", "Yes", "Yes", "Yes"],
            ["Financial controls", "No", "No", "No", "No", "Yes", "No"],
            ["BDC fixed effects", "No", "No", "No", "No", "No", "Yes"],
            ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Observations", *[str(int(x)) for x in credit.n]],
            ["R-squared", *[f"{x:.3f}" for x in credit.r_squared]],
        ]
        credit_note = "Credit-adjusted columns re-estimate dynamic climate beta after adding the average HYG/JNK return less SHY. All models use the same 380 BDC-quarters and BDC-clustered standard errors. ***, **, and * denote two-sided significance at 1%, 5%, and 10%."
        credit_body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs22\landscape\paperw15840\paperh12240\margl720\margr720\margt900\margb900", r"\pard\sa40\b\fs24 Table 3: High-Yield Credit-Factor Robustness of BDC Climate Beta.\b0\par"]
        for i, cells in enumerate(credit_rows):
            credit_body.append(rtf_row(cells, [3200] + [1800] * 6, bold=i == 0, top=i == 0, mid=i == 0, bottom=i == len(credit_rows) - 1))
        credit_body += [r"\pard\sa120\fs18 " + esc("Notes: " + credit_note) + r"\par", "}"]
        credit_body[1] = r"\pard\sa40\b\fs24 Table S1: High-Yield Credit-Factor Robustness of BDC Climate Beta.\b0\par"

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    x = panel["brown_share_broad_dynamic_pct"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharex=True)
    for ax, ycol, label, color in [
        (axes[0], "beta_climate_equity_report_month", "Equity climate beta", BLUE),
        (axes[1], "beta_climate_asset_report_month", "Asset climate beta", ORANGE),
    ]:
        y = panel[ycol].to_numpy(float)
        ax.scatter(x, y, s=14, alpha=.35, color=color, edgecolor="none")
        fit = np.polyfit(x, y, 1)
        grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(grid, np.polyval(fit, grid), color="black", lw=1.5)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Broad carbon-intensive investment share (%)")
    axes[0].set_ylabel("Climate beta")
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_3_Investment_Exposure_and_Climate_Beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    panel["quarter"] = pd.PeriodIndex(panel["fiscal_quarter"], freq="Q").to_timestamp("Q") if "fiscal_quarter" in panel else pd.to_datetime(panel["datadate"])
    trend = panel.groupby("quarter")["brown_share_broad_dynamic_pct"].agg(["mean", "median"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.plot(trend.quarter, trend["mean"], color=BLUE, lw=2.2, label="Mean")
    ax.plot(trend.quarter, trend["median"], color=ORANGE, lw=2.2, label="Median")
    ax.set_ylabel("Broad carbon-intensive investment share (%)")
    ax.set_xlabel("")
    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))
    ax.set_xticks([pd.Timestamp(f"{year}-07-01") for year in range(2021, 2026)])
    ax.set_xticklabels([str(year) for year in range(2021, 2026)])
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_4_Investment_Exposure_Trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
