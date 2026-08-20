from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bdc_ff49_results")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
BLUE = "#17365D"
ORANGE = "#A64B00"
GRAY = "#6B7280"


def esc(value: object) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def rtf_row(cells, widths, *, bold=False, top=False, bottom=False, mid=False):
    out = [r"\trowd\trgaph80\trleft0"]
    if top:
        out.append(r"\trbrdrt\brdrdb\brdrw14")
    if mid:
        out.append(r"\trbrdrb\brdrs\brdrw8")
    if bottom:
        out.append(r"\trbrdrb\brdrdb\brdrw14")
    endpoint = 0
    for width in widths:
        endpoint += width
        out.append(fr"\cellx{endpoint}")
    for index, value in enumerate(cells):
        alignment = r"\intbl\ql " if index == 0 else r"\intbl\qc "
        out.append(alignment + (r"\b " if bold else "") + esc(value) + (r"\b0" if bold else "") + r"\cell")
    out.append(r"\row")
    return "".join(out)


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def coefficient(item: pd.Series) -> str:
    return f"{item.coefficient_exposure:.3f}{stars(item.p_two_sided)}"


def write_table() -> None:
    primary = pd.read_csv(DATA / "h2_ff49_dcc_portfolio_mechanism_models.csv").set_index("model_id")
    robust = pd.read_csv(DATA / "h2_ff49_dcc_mechanism_robustness_models.csv").set_index("model_id")
    ff49_ols = pd.read_csv(DATA / "h2_ff49_portfolio_mechanism_models.csv").set_index("model_id")
    ff12 = pd.read_csv(DATA / "h2_portfolio_beta_mechanism_models.csv").set_index("model_id")
    brown = pd.read_csv(DATA / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    wild = pd.read_csv(DATA / "h2_ff49_dcc_wild_cluster_bootstrap.csv")

    ids = ["DCC49_4", "DCC49_5", "DCC49_6", "DCC49_1", "DCC49_2", "DCC49_3"]
    main = primary.loc[ids]
    rows: list[list[str]] = [
        ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
        ["Panel A: DCC-FF49 portfolio climate beta", "", "", "", "", "", ""],
        ["Portfolio climate beta", *[coefficient(row) for _, row in main.iterrows()]],
        ["", *[f"({row.standard_error:.3f})" for _, row in main.iterrows()]],
        ["Outcome", "Equity", "Equity", "Equity", "Asset", "Asset", "Asset"],
        ["Financial controls", "No", "Yes", "No", "No", "Yes", "No"],
        ["BDC fixed effects", "No", "No", "Yes", "No", "No", "Yes"],
        ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
        ["Observations", *[str(int(row.n)) for _, row in main.iterrows()]],
        ["R-squared", *[f"{row.r_squared:.3f}" for _, row in main.iterrows()]],
        ["Panel B: Measurement and sample robustness", "", "", "", "", "", ""],
        ["", "Brown share", "FF12 beta", "FF49 OLS", "DCC-FF49", "High/medium", "Post-2021"],
    ]
    compare = [
        brown.loc["H2_5"],
        ff12.loc["M1"],
        ff49_ols.loc["FF49_6"],
        robust.loc["FULL_EQ"],
        robust.loc["HIMED_EQ"],
        robust.loc["POST21_EQ"],
    ]
    rows.extend(
        [
            ["Equity-beta coefficient", *[coefficient(row) for row in compare]],
            ["", *[f"({row.standard_error:.3f})" for row in compare]],
            ["BDC fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Observations", *[str(int(row.n)) for row in compare]],
            ["R-squared", *[f"{row.r_squared:.3f}" for row in compare]],
        ]
    )
    wild_equity = wild.loc[wild["outcome"].str.contains("equity")].iloc[0]
    note = (
        "The dependent variable and portfolio measures are standardized. FF49 industry returns are "
        "value weighted from CRSP common stocks; industry and BDC betas use the same median scalar-DCC "
        "parameter vector. Standard errors clustered by BDC appear in parentheses. High/medium excludes "
        "low-confidence label mappings. Post-2021 drops the DCC warm-up year. The full-sample equity "
        f"wild-cluster bootstrap p-value is {wild_equity.wild_cluster_p_two_sided:.3f} (9,999 draws), so "
        "the 10-percent conventional result is not robust to small-cluster bootstrap inference. ***, **, "
        "and * denote two-sided significance at 1, 5, and 10 percent."
    )
    body = [
        r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs22\landscape\paperw15840\paperh12240\margl720\margr720\margt720\margb720",
        r"\pard\sa40\b\fs24 Table 3: BDC Portfolio Climate Beta and Market Climate Beta.\b0\par",
        r"\pard\sa80\fs19 This table tests whether the climate beta of a BDC's disclosed investment industries is reflected in the BDC's traded climate beta.\par",
    ]
    widths = [3300] + [1700] * 6
    for index, cells in enumerate(rows):
        section = str(cells[0]).startswith("Panel")
        body.append(
            rtf_row(
                cells,
                widths,
                bold=index == 0 or section,
                top=index == 0,
                mid=index == 0 or section,
                bottom=index == len(rows) - 1,
            )
        )
    body.extend([r"\pard\sa80\fs18 " + esc("Notes: " + note) + r"\par", "}"])
    (RESULTS / "Table_3_BDC_FF49_DCC_Portfolio_Mechanism.rtf").write_text(
        "\n".join(body), encoding="ascii", errors="ignore"
    )


def plot_results() -> None:
    primary = pd.read_csv(DATA / "h2_ff49_dcc_portfolio_mechanism_models.csv").set_index("model_id")
    robust = pd.read_csv(DATA / "h2_ff49_dcc_mechanism_robustness_models.csv").set_index("model_id")
    ff49_ols = pd.read_csv(DATA / "h2_ff49_portfolio_mechanism_models.csv").set_index("model_id")
    ff12 = pd.read_csv(DATA / "h2_portfolio_beta_mechanism_models.csv").set_index("model_id")
    brown = pd.read_csv(DATA / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    z90 = 1.6448536269514722

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1), sharex=True)
    comparisons = [
        ("Brown share", brown.loc["H2_5"]),
        ("FF12 portfolio", ff12.loc["M1"]),
        ("FF49 rolling OLS", ff49_ols.loc["FF49_6"]),
        ("FF49 aligned DCC", primary.loc["DCC49_6"]),
    ]
    robustness = [
        ("Full sample", robust.loc["FULL_EQ"]),
        ("High/medium maps", robust.loc["HIMED_EQ"]),
        ("Post-2021", robust.loc["POST21_EQ"]),
        ("Drop imputations", robust.loc["NOIMP_EQ"]),
        ("Coverage >= 80%", robust.loc["COV80_EQ"]),
    ]
    for ax, items, title in [
        (axes[0], comparisons, "Measurement resolution"),
        (axes[1], robustness, "DCC-FF49 robustness"),
    ]:
        y = np.arange(len(items))[::-1]
        estimates = np.array([row.coefficient_exposure for _, row in items])
        errors = z90 * np.array([row.standard_error for _, row in items])
        colors = [ORANGE if row.p_two_sided < 0.10 else BLUE for _, row in items]
        ax.errorbar(estimates, y, xerr=errors, fmt="none", ecolor=GRAY, elinewidth=1.3, capsize=3)
        ax.scatter(estimates, y, c=colors, s=36, zorder=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _ in items])
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Standardized coefficient (90% CI)")
    axes[0].set_xlim(-0.28, 0.40)
    axes[1].set_xlim(-0.28, 0.40)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(RESULTS / "Figure_3_FF49_DCC_Portfolio_Mechanism.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_table()
    plot_results()


if __name__ == "__main__":
    main()
