from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bdc_descriptive_matplotlib")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
H3_DATA = PACKAGE / "03_BDC_Asset_Coverage_Stress_Test" / "Data" / "Processed"


def load_helper():
    path = Path(__file__).with_name("01_industry_beta_helpers.py")
    spec = importlib.util.spec_from_file_location("h2_desc_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def esc(value: object) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def row(cells, widths, *, bold=False, top=False, bottom=False):
    out = [r"\trowd\trgaph70\trleft0"]
    if top:
        out.append(r"\trbrdrt\brdrdb\brdrw14\trbrdrb\brdrs\brdrw8")
    if bottom:
        out.append(r"\trbrdrb\brdrdb\brdrw14")
    endpoint = 0
    for width in widths:
        endpoint += width
        out.append(fr"\cellx{endpoint}")
    for index, value in enumerate(cells):
        out.append((r"\intbl\ql " if index == 0 else r"\intbl\qc ") + (r"\b " if bold else "") + esc(value) + (r"\b0" if bold else "") + r"\cell")
    out.append(r"\row")
    return "".join(out)


def variance_decomposition(frame: pd.DataFrame, column: str) -> dict[str, float]:
    data = frame[["ticker", column]].dropna().copy()
    grand = data[column].mean()
    firm_mean = data.groupby("ticker")[column].transform("mean")
    total_ss = float(((data[column] - grand) ** 2).sum())
    between_ss = float(((firm_mean - grand) ** 2).sum())
    within_ss = float(((data[column] - firm_mean) ** 2).sum())
    return {
        "variable": column,
        "observations": len(data),
        "firms": data["ticker"].nunique(),
        "total_sum_squares": total_ss,
        "between_sum_squares": between_ss,
        "within_sum_squares": within_ss,
        "between_share_of_total_variation": between_ss / total_ss,
        "within_share_of_total_variation": within_ss / total_ss,
        "decomposition_error": total_ss - between_ss - within_ss,
    }


def main() -> None:
    panel = pd.read_csv(PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv", parse_dates=["datadate"])
    h3_path = H3_DATA / "h3_analysis_panel.csv"
    if h3_path.exists():
        h3 = pd.read_csv(h3_path, parse_dates=["datadate"])
        extra = h3[["ticker", "datadate", "baseline_coverage_pct", "price_to_nav"]].drop_duplicates(["ticker", "datadate"])
        panel = panel.merge(extra, on=["ticker", "datadate"], how="left", validate="one_to_one")
    else:
        panel["baseline_coverage_pct"] = np.nan
        panel["price_to_nav"] = np.nan

    labels = {
        "beta_climate_equity_report_month": "Equity climate beta",
        "beta_climate_asset_report_month": "Asset climate beta",
        "brown_share_broad_dynamic_pct": "Broad carbon-intensive share (%)",
        "brown_share_narrow_dynamic_pct": "Narrow carbon-intensive share (%)",
        "assets_total_mn": "Total assets (USD mn)",
        "debt_to_assets": "Debt to assets",
        "roa_quarter": "Quarterly ROA",
        "book_to_market": "Book to market",
        "beta_market_report_month": "Market beta",
        "baseline_coverage_pct": "Reported asset coverage (%)",
        "price_to_nav": "Market to NAV",
    }
    rows = []
    for variable, label in labels.items():
        values = pd.to_numeric(panel[variable], errors="coerce").dropna()
        rows.append({
            "variable": variable,
            "label": label,
            "n": len(values),
            "mean": values.mean(),
            "standard_deviation": values.std(ddof=1),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "minimum": values.min(),
            "maximum": values.max(),
        })
    descriptive = pd.DataFrame(rows)
    descriptive.to_csv(PROCESSED / "h2_descriptive_statistics_full.csv", index=False)
    HELPER.export_dta(descriptive, PROCESSED / "h2_descriptive_statistics_full.dta")
    reciprocal = pd.DataFrame([{
        "observations": int(panel[["book_to_market", "price_to_nav"]].dropna().shape[0]),
        "mean_book_to_market": panel["book_to_market"].mean(),
        "mean_market_to_nav": panel["price_to_nav"].mean(),
        "max_abs_product_minus_one": (
            panel["book_to_market"] * panel["price_to_nav"] - 1.0
        ).abs().max(),
        "definition": "Market-to-NAV is the exact observation-level reciprocal of book-to-market in the maintained BDC panel.",
    }])
    reciprocal.to_csv(PROCESSED / "book_to_market_reciprocal_audit.csv", index=False)
    HELPER.export_dta(reciprocal, PROCESSED / "book_to_market_reciprocal_audit.dta")
    panel.to_csv(PROCESSED / "bdc19_descriptive_panel_2021_2025.csv", index=False)
    HELPER.export_dta(panel, PROCESSED / "bdc19_descriptive_panel_2021_2025.dta")

    decomposition = pd.DataFrame([
        variance_decomposition(panel, "brown_share_broad_dynamic_pct"),
        variance_decomposition(panel, "brown_share_narrow_dynamic_pct"),
        variance_decomposition(panel, "beta_climate_equity_report_month"),
        variance_decomposition(panel, "beta_climate_asset_report_month"),
    ])
    decomposition.to_csv(PROCESSED / "h2_within_between_variance_decomposition.csv", index=False)
    HELPER.export_dta(decomposition, PROCESSED / "h2_within_between_variance_decomposition.dta")

    table_rows = [["Variable", "N", "Mean", "SD", "Median", "Min", "Max"]]
    for item in descriptive.itertuples(index=False):
        precision = 4 if item.variable in {"book_to_market", "price_to_nav"} else 3
        table_rows.append([
            item.label,
            str(int(item.n)),
            f"{item.mean:.{precision}f}",
            f"{item.standard_deviation:.{precision}f}",
            f"{item.median:.{precision}f}",
            f"{item.minimum:.{precision}f}",
            f"{item.maximum:.{precision}f}",
        ])
    broad = decomposition.loc[decomposition["variable"].eq("brown_share_broad_dynamic_pct")].iloc[0]
    note = (
        "The balanced panel contains 19 BDCs and 20 quarters from 2021Q1 through 2025Q4. Climate betas are report-month averages. "
        "Portfolio shares are percentages of reported fair value. Asset coverage and market-to-NAV are available for 376 observations. "
        f"Within-BDC movements account for {100*broad.within_share_of_total_variation:.1f}% of total variation in the broad carbon-intensive share; "
        "the remainder is between BDCs. Book-to-market and market-to-NAV use the same book equity measure and are exact observation-level reciprocals; their close sample means are not a duplicated row."
    )
    body = [
        r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs21\paperw12240\paperh15840\margl700\margr700\margt760\margb760",
        r"\pard\sa35\b\fs24 Table 2: BDC Panel Descriptive Statistics.\b0\par",
        r"\pard\sa100\fs18 " + esc(note) + r"\par",
    ]
    for index, cells in enumerate(table_rows):
        body.append(row(cells, [3550] + [1250] * 6, bold=index == 0, top=index == 0, bottom=index == len(table_rows) - 1))
    body.extend([r"\pard\sa90\fs17 " + esc("Notes: " + note) + r"\par", "}"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "Table_2_BDC_Descriptive_Statistics.rtf").write_text("\n".join(body), encoding="ascii", errors="ignore")
    print(descriptive[["label", "n", "mean", "standard_deviation", "median", "minimum", "maximum"]].to_string(index=False))
    print("\nVariance decomposition:\n", decomposition.to_string(index=False))


if __name__ == "__main__":
    main()
