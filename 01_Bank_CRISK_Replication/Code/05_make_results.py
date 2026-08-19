from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bank_results_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
BLUE = "#08457E"
ORANGE = "#F28E00"


def esc(value: object) -> str:
    text = str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return text


def row(cells, widths, *, bold=False, top=False, bottom=False, mid=False, align=None):
    align = align or ["l"] + ["c"] * (len(cells) - 1)
    out = [r"\trowd\trgaph80\trleft0"]
    if top:
        out.append(r"\trbrdrt\brdrdb\brdrw14")
    if mid:
        out.append(r"\trbrdrb\brdrs\brdrw8")
    if bottom:
        out.append(r"\trbrdrb\brdrdb\brdrw14")
    total = 0
    for width in widths:
        total += width
        out.append(fr"\cellx{total}")
    for value, side in zip(cells, align):
        out.append(fr"\intbl\q{side} " + (r"\b " if bold else "") + esc(value) + (r"\b0" if bold else "") + r"\cell")
    out.append(r"\row")
    return "".join(out)


def write_table(path: Path, title: str, note: str, sections):
    body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs22\paperw12240\paperh15840\margl900\margr900\margt900\margb900"]
    body += [r"\pard\sa100\b\fs24 " + esc(title) + r"\b0\par"]
    for section in sections:
        body.append(r"\pard\sa40\b " + esc(section["label"]) + r"\b0\par")
        rows = section["rows"]
        for index, cells in enumerate(rows):
            body.append(row(cells, section["widths"], bold=index == 0, top=index == 0, mid=index == 0, bottom=index == len(rows) - 1, align=section.get("align")))
        body.append(r"\pard\sa100\par")
    body += [r"\pard\fs18 " + esc("Notes: " + note) + r"\par", "}"]
    path.write_text("\n".join(body), encoding="ascii", errors="ignore")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tests = pd.read_csv(DATA / "h1_statistical_tests.csv").set_index("test_id")
    bench = pd.read_csv(DATA / "h1_benchmark_comparison.csv").set_index("criterion_id")
    annual = pd.read_csv(DATA / "dcb_crisk_annual_summary.csv")
    daily = pd.read_csv(DATA / "bank_h1_daily_2010_2025.csv", parse_dates=["date"], low_memory=False)
    top4_detail = pd.read_csv(DATA / "top4_end2020_detail.csv").sort_values("current_ticker")
    decomposition = pd.read_csv(DATA / "top4_mcrisk_scale_decomposition.csv").iloc[0]
    peaks = pd.read_csv(DATA / "bank_beta_peak_timing.csv", parse_dates=["peak_date"])
    bdc_paired = pd.read_csv(DATA / "bdc_beta_2019_2020_paired_test.csv").iloc[0]

    def stars(p_value):
        if p_value < 0.01:
            return "***"
        if p_value < 0.05:
            return "**"
        if p_value < 0.10:
            return "*"
        return ""

    section_a = [
        ["", "Annual beta", "Daily beta", "Signed CRISK", "Positive CRISK"],
        ["2020 increase", *[
            f"{tests.loc[x, 'estimate']:.3f}{stars(tests.loc[x, 'p_value_two_sided'])}"
            for x in ["T1", "T3", "T4", "T5"]
        ]],
        ["", *[f"({tests.loc[x, 'std_error']:.3f})" for x in ["T1", "T3", "T4", "T5"]]],
        ["Observations", *[str(int(tests.loc[x, "n"])) for x in ["T1", "T3", "T4", "T5"]]],
    ]
    section_b = [
        ["Published comparison", "Replicated", "Published", "Ratio", "Qualification"],
        ["Annual-average peak year", str(int(bench.loc['T1','replicated_value'])), "2020", "--", "2020-2021 plateau"],
        ["Raw daily peak", f"{peaks.loc[peaks.series.eq('Cross-bank raw daily mean'),'peak_value'].iloc[0]:.3f}", "--", "--", "Vaccine/value rotation"],
        ["Maximum 127-day beta (WFC)", f"{bench.loc['S1','replicated_value']:.4f}", "<0.500", f"{bench.loc['S1','ratio_to_published']:.2f}", "Smoothing not stated"],
        ["Top-four mCRISK (USD bn)", f"{bench.loc['M1','replicated_value']:.1f}", "260.0", f"{bench.loc['M1','ratio_to_published']:.2f}", "85.3%"],
        ["Top-four CRISK increase (USD bn)", f"{bench.loc['M2','replicated_value']:.1f}", "430.9", f"{bench.loc['M2','ratio_to_published']:.2f}", "86.5%"],
    ]
    section_c = [["Institution", "Mean beta (reference)", "Mean E x LRMES", "Mean mCRISK", "Identity error"]]
    for item in top4_detail.itertuples(index=False):
        section_c.append([
            item.current_ticker,
            f"{item.beta_climate_ma127:.4f}",
            f"{item.mean_equity_lrmes_usd_bn:.2f}",
            f"{item.mcrisk_usd_bn:.1f}",
            f"{item.mean_identity_error_usd_bn:.2e}",
        ])
    section_c.append([
        "Total", "--", f"{top4_detail.mean_equity_lrmes_usd_bn.sum():.2f}",
        f"{top4_detail.mcrisk_usd_bn.sum():.1f}", f"{top4_detail.mean_identity_error_usd_bn.abs().max():.2e}",
    ])
    section_d = [
        ["Sample", "Mean beta 2019", "Mean beta 2020", "Mean change", "Positive changes"],
        [
            "20 BDCs", f"{bdc_paired.mean_beta_2019:.3f}", f"{bdc_paired.mean_beta_2020:.3f}",
            f"{bdc_paired.mean_change:.3f}{stars(bdc_paired.paired_t_p_two_sided)}",
            f"{int(bdc_paired.positive_changes)}/{int(bdc_paired.observations)}",
        ],
        ["", "", "", f"({bdc_paired.mean_change_std_error:.3f})", ""],
        [
            "19-BDC H2 subsample", f"{bdc_paired.h2_subsample_mean_beta_2019:.3f}",
            f"{bdc_paired.h2_subsample_mean_beta_2020:.3f}",
            f"{bdc_paired.h2_subsample_mean_change:.3f}{stars(bdc_paired.h2_paired_t_p_two_sided)}",
            f"{int(bdc_paired.h2_subsample_positive_changes)}/{int(bdc_paired.h2_subsample_observations)}",
        ],
        ["", "", "", f"({bdc_paired.h2_mean_change_se:.3f})", ""],
    ]
    note = (
        "Daily diagnostics use HAC(203); lag sensitivity is archived. Because institutions share one factor realization, paired statistics measure cross-institution consistency rather than independent event replications. "
        "Panel A's signed CRISK is the difference between calendar-year daily means; Panel B's benchmark-aligned increase uses December-average beta with year-end debt and equity. The raw daily peak follows the 9 November 2020 Pfizer announcement, not the March oil-price break. "
        "Panel C applies mCRISK = 0.92 x E x LRMES daily and then averages the products; mean beta is reported only for reference. "
        f"The replicated-to-published mCRISK ratio of {decomposition.mcrisk_ratio_replicated_to_published:.3f} equals an equity-scale ratio of {decomposition.equity_scale_ratio:.3f} times a climate-loss-rate ratio of {decomposition.loss_rate_ratio:.3f}. "
        "The published text rounds the increase to USD 425 billion; its Table 2 sums to USD 430.89 billion. P-values are two-sided. ***, **, and * denote p<0.01, p<0.05, and p<0.10."
    )
    write_table(RESULTS / "Table_1_Bank_CRISK_Replication.rtf", "Table 1: Bank Replication of the 2020 Climate-Risk Shock.", note, [
        {"label": "Panel A: Statistical validation", "rows": section_a, "widths": [2600, 1700, 1700, 1700, 1700]},
        {"label": "Panel B: Published-magnitude comparison", "rows": section_b, "widths": [3600, 1450, 1450, 1450, 1450]},
        {"label": "Panel C: End-2020 top-four 127-day mean construction", "rows": section_c, "widths": [2300, 1600, 2300, 2000, 1800]},
        {"label": "Panel D: BDC time-series validation", "rows": section_d, "widths": [2500, 1900, 1900, 1800, 2200]},
    ])

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    grouped = annual.groupby(["group", "year"], as_index=False).agg(
        beta_climate_mean=("beta_climate_mean", "mean"),
        institutions=("current_ticker", "nunique"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.3), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax = axes[0]
    for group, color in [("Bank", BLUE), ("BDC", ORANGE)]:
        g = grouped[grouped.group.eq(group)]
        ax.plot(g.year, g.beta_climate_mean, lw=2.3, color=color, label="Banks" if group == "Bank" else "BDCs")
    ax.axvline(2020, color="0.55", ls="--", lw=1)
    ax.set_ylabel("Mean climate beta")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    count_ax = axes[1]
    for group, color in [("Bank", BLUE), ("BDC", ORANGE)]:
        g = grouped[grouped.group.eq(group)]
        count_ax.step(g.year, g.institutions, where="mid", lw=1.8, color=color)
    count_ax.set_ylabel("N")
    count_ax.set_xlabel("Year")
    count_ax.set_xticks(range(2010, 2026, 2))
    count_ax.set_ylim(0, max(grouped.institutions) + 2)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_1_Aggregate_Climate_Beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top = daily[daily.current_ticker.isin(["BAC", "C", "JPM", "WFC"]) & daily.date.between("2019-01-01", "2020-12-31")]
    agg = top.groupby("date", as_index=False).agg(raw_signed=("crisk_8pct_mn", "sum"), raw_positive=("crisk_8pct_positive_mn", "sum"))
    agg[["raw_signed", "raw_positive"]] /= 1000
    agg["smooth_signed"] = agg.raw_signed.rolling(127, min_periods=127).mean()
    agg["smooth_positive"] = agg.raw_positive.rolling(127, min_periods=127).mean()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.1), sharex=True)
    axes[0].plot(agg.date, agg.raw_signed, color=BLUE, lw=1.35, label="Signed CRISK")
    axes[0].plot(agg.date, agg.raw_positive, color=ORANGE, lw=1.35, label="Positive CRISK")
    axes[0].set_title("Raw daily series", loc="left", fontsize=11)
    axes[1].plot(agg.date, agg.smooth_signed, color=BLUE, lw=2.1, label="Signed CRISK")
    axes[1].plot(agg.date, agg.smooth_positive, color=ORANGE, lw=2.1, label="Positive CRISK")
    axes[1].set_title("Trailing 127-trading-day means", loc="left", fontsize=11)
    events = [
        (pd.Timestamp("2020-03-09"), "Oil", "#8B0000", "--"),
        (pd.Timestamp("2020-03-16"), "COVID", "0.45", ":"),
        (pd.Timestamp("2020-11-09"), "Pfizer", "#5B2C6F", "-."),
    ]
    for ax in axes:
        for event_date, event_label, event_color, event_style in events:
            ax.axvline(event_date, color=event_color, ls=event_style, lw=0.9)
        ax.set_ylabel("USD billion")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    ymin, ymax = axes[0].get_ylim()
    event_positions = [0.89, 0.67, 0.82]
    for (event_date, event_label, event_color, _), fraction in zip(events, event_positions):
        axes[0].annotate(event_label, xy=(event_date, ymin + fraction * (ymax - ymin)), xytext=(-3, 0),
                         textcoords="offset points", color=event_color, rotation=90,
                         va="center", ha="right", fontsize=8, clip_on=True)
    axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_2_Top_Four_Bank_CRISK.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
