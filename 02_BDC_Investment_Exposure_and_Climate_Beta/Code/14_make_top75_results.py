from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_top75_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
MODULE1 = PACKAGE / "01_Bank_CRISK_Replication"
BLUE = "#08457E"
ORANGE = "#F28E00"
GRAY = "#666666"


def rtf_escape(value: object) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def write_rtf(specifications: pd.DataFrame) -> None:
    labels = specifications["label"].str.replace("\n", " ", regex=False).tolist()
    widths = [3100] + [2400] * len(labels)
    rows = [
        ["", *labels],
        ["Equity portfolio beta", *[f"{row.equity:.3f}{significance(row.equity_p)}" for row in specifications.itertuples()]],
        ["", *[f"({row.equity_se:.3f})" for row in specifications.itertuples()]],
        ["Asset portfolio beta", *[f"{row.asset:.3f}{significance(row.asset_p)}" for row in specifications.itertuples()]],
        ["", *[f"({row.asset_se:.3f})" for row in specifications.itertuples()]],
        ["KOL tracking correlation", *[f"{value:.3f}" for value in specifications["tracking"]]],
        ["Observations", "380", "380", "380"],
    ]
    body = [
        r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs22\landscape\paperw15840\paperh12240\margl900\margr900\margt900\margb900",
        r"\pard\sa60\b\fs24 Table S2: KOL Basket Breadth and BDC Portfolio Climate Beta.\b0\par",
    ]
    for row_number, row in enumerate(rows):
        body.append(r"\trowd\trgaph80\trleft0")
        endpoint = 0
        for width in widths:
            endpoint += width
            body.append(fr"\cellx{endpoint}")
        for column, value in enumerate(row):
            align = r"\intbl\ql " if column == 0 else r"\intbl\qc "
            bold = r"\b " if row_number == 0 else ""
            bold_end = r"\b0" if row_number == 0 else ""
            body.append(align + bold + rtf_escape(value) + bold_end + r"\cell")
        body.append(r"\row")
    note = (
        "Notes: The cumulative-75-percent basket contains 15 securities and covers 77.1 percent of "
        "KOL common-stock market value in the official 30 September 2020 N-PORT schedule. All models "
        "include BDC and quarter fixed effects with BDC-clustered standard errors. ***, **, and * denote "
        "two-sided significance at 1, 5, and 10 percent."
    )
    body.extend([r"\pard\sa80\fs18 " + rtf_escape(note) + r"\par", "}"])
    (RESULTS / "Table_S2_KOL_Top75_Robustness.rtf").write_text(
        "\n".join(body), encoding="ascii", errors="ignore"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    top5 = pd.read_csv(PROCESSED / "h2_ff49_dcc_portfolio_mechanism_models.csv").set_index("model_id")
    top75 = pd.read_csv(PROCESSED / "h2_ff49_dcc_kol_top75_models.csv").set_index("model_id")
    top75_weekly = pd.read_csv(PROCESSED / "h2_ff49_dcc_kol_top75_weekly_models.csv").set_index("model_id")
    tracking_top75 = pd.read_csv(MODULE1 / "Data" / "Processed" / "kol_top75_tracking_diagnostics.csv")
    tracking_top5 = pd.read_csv(MODULE1 / "Data" / "Processed" / "coal_proxy_diagnostics.csv")
    maintained_top5 = tracking_top5.loc[tracking_top5["maintained_factor"].eq(1)].iloc[0]

    tracking = pd.DataFrame(
        [
            {"basket": "Top five", "frequency": "Daily", "correlation": maintained_top5["correlation_with_kol"]},
            {"basket": "Top five", "frequency": "Weekly", "correlation": maintained_top5["weekly_correlation_with_kol"]},
            {"basket": "Cumulative 75%", "frequency": "Daily", "correlation": tracking_top75.loc[tracking_top75["frequency"].eq("Daily"), "correlation_with_kol"].iloc[0]},
            {"basket": "Cumulative 75%", "frequency": "Weekly", "correlation": tracking_top75.loc[tracking_top75["frequency"].eq("Weekly"), "correlation_with_kol"].iloc[0]},
        ]
    )
    specifications = pd.DataFrame(
        [
            {
                "label": "Top five\ndaily", "tracking": maintained_top5["correlation_with_kol"],
                "equity": top5.loc["DCC49_6", "coefficient_exposure"],
                "equity_se": top5.loc["DCC49_6", "standard_error"],
                "equity_p": top5.loc["DCC49_6", "p_two_sided"],
                "asset": top5.loc["DCC49_3", "coefficient_exposure"],
                "asset_se": top5.loc["DCC49_3", "standard_error"],
                "asset_p": top5.loc["DCC49_3", "p_two_sided"],
            },
            {
                "label": "Cumulative 75%\ndaily", "tracking": tracking_top75.loc[tracking_top75["frequency"].eq("Daily"), "correlation_with_kol"].iloc[0],
                "equity": top75.loc["T75_6", "coefficient_exposure"],
                "equity_se": top75.loc["T75_6", "standard_error"],
                "equity_p": top75.loc["T75_6", "p_two_sided"],
                "asset": top75.loc["T75_3", "coefficient_exposure"],
                "asset_se": top75.loc["T75_3", "standard_error"],
                "asset_p": top75.loc["T75_3", "p_two_sided"],
            },
            {
                "label": "Cumulative 75%\nweekly", "tracking": tracking_top75.loc[tracking_top75["frequency"].eq("Weekly"), "correlation_with_kol"].iloc[0],
                "equity": top75_weekly.loc["T75W_6", "coefficient_exposure"],
                "equity_se": top75_weekly.loc["T75W_6", "standard_error"],
                "equity_p": top75_weekly.loc["T75W_6", "p_two_sided"],
                "asset": top75_weekly.loc["T75W_3", "coefficient_exposure"],
                "asset_se": top75_weekly.loc["T75W_3", "standard_error"],
                "asset_p": top75_weekly.loc["T75W_3", "p_two_sided"],
            },
        ]
    )
    specifications.to_csv(PROCESSED / "h2_top5_top75_daily_weekly_summary.csv", index=False)
    write_rtf(specifications)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [0.85, 1.15]})
    ax = axes[0]
    x = np.arange(2)
    width = 0.34
    daily = tracking.loc[tracking["frequency"].eq("Daily")].set_index("basket").loc[["Top five", "Cumulative 75%"]]
    weekly = tracking.loc[tracking["frequency"].eq("Weekly")].set_index("basket").loc[["Top five", "Cumulative 75%"]]
    bars1 = ax.bar(x - width / 2, daily["correlation"], width, color=BLUE, label="Daily")
    bars2 = ax.bar(x + width / 2, weekly["correlation"], width, color=ORANGE, label="Weekly")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["Top five", "Cumulative 75%"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Correlation with KOL")
    ax.set_title("A. Pre-liquidation tracking")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(specifications))
    shift = 0.09
    ax.errorbar(
        x - shift, specifications["equity"], yerr=1.645 * specifications["equity_se"],
        fmt="o", color=BLUE, ecolor=BLUE, capsize=3, label="Equity beta",
    )
    ax.errorbar(
        x + shift, specifications["asset"], yerr=1.645 * specifications["asset_se"],
        fmt="s", color=ORANGE, ecolor=ORANGE, capsize=3, label="Asset beta",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, specifications["label"])
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("B. BDC portfolio mechanism")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, row in specifications.iterrows():
        if row["equity_p"] < 0.10:
            ax.text(idx - shift, row["equity"] + 1.645 * row["equity_se"] + 0.02, "*", ha="center", color=BLUE, fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_5_KOL_Basket_Breadth_and_BDC_Mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(tracking.to_string(index=False))
    print(specifications.to_string(index=False))


if __name__ == "__main__":
    main()
