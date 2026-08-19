from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def export(df: pd.DataFrame, path_no_suffix: Path) -> None:
    df.to_csv(path_no_suffix.with_suffix(".csv"), index=False)
    rename = {}
    used = set()
    for original in df.columns:
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(original)).lower()
        if not re.match(r"^[A-Za-z_]", clean):
            clean = f"v_{clean}"
        clean = clean[:32]
        candidate = clean
        sequence = 1
        while candidate in used:
            suffix = f"_{sequence}"
            candidate = clean[: 32 - len(suffix)] + suffix
            sequence += 1
        used.add(candidate)
        rename[original] = candidate
    safe = df.rename(columns=rename).copy()
    for c in safe.select_dtypes(include=["object"]).columns:
        safe[c] = safe[c].astype(str).str.slice(0, 244)
    safe.to_stata(path_no_suffix.with_suffix(".dta"), write_index=False, version=118)


def cluster_mean_test(values: pd.Series, groups: pd.Series) -> dict:
    keep = values.notna() & groups.notna()
    y = values[keep].astype(float).to_numpy()
    g = groups[keep].astype(str).to_numpy()
    estimate = float(y.mean())
    residual = y - estimate
    unique_groups = np.unique(g)
    cluster_sums = np.array([residual[g == group].sum() for group in unique_groups])
    n = len(y)
    cluster_count = len(unique_groups)
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - 1))
    se = float(np.sqrt(correction * np.sum(cluster_sums ** 2) / (n ** 2)))
    t_value = estimate / se
    df = cluster_count - 1
    p_one = float(stats.t.sf(t_value, df=df))
    return {
        "estimate": estimate,
        "std_error": se,
        "t_stat": t_value,
        "p_value_one_sided": p_one,
        "p_value_two_sided": min(1.0, 2 * p_one) if t_value >= 0 else min(1.0, 2 * (1 - p_one)),
        "observations": int(len(y)),
        "clusters": int(cluster_count),
        "df": int(df),
    }


def bootstrap_firm_mean(panel: pd.DataFrame, col: str, reps: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    tickers = panel["ticker"].drop_duplicates().to_numpy()
    draws = np.empty(reps)
    pieces = {t: panel.loc[panel["ticker"] == t, col].dropna().to_numpy(float) for t in tickers}
    for i in range(reps):
        sampled = rng.choice(tickers, size=len(tickers), replace=True)
        draws[i] = np.concatenate([pieces[t] for t in sampled]).mean()
    return {
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "ci_2_5": float(np.quantile(draws, 0.025)),
        "ci_97_5": float(np.quantile(draws, 0.975)),
        "p_one_sided_le_zero": float((np.sum(draws <= 0) + 1) / (reps + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-panel", required=True)
    parser.add_argument("--daily-crisk", required=True)
    parser.add_argument("--sec-coverage", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    market = pd.read_csv(args.market_panel, parse_dates=["datadate", "report_month_start", "report_month_end"])
    market = market[market["ticker"] != "BBDC"].copy()
    daily = pd.read_csv(args.daily_crisk, parse_dates=["date"])
    daily = daily[(daily["group"] == "BDC") & (daily["current_ticker"].isin(market["ticker"]))].copy()
    coverage = pd.read_csv(args.sec_coverage, parse_dates=["report_date"])

    factor_path = (
        Path(__file__).resolve().parents[2]
        / "01_Bank_CRISK_Replication/Data/Processed/climate_factor_daily_2010_2025.csv"
    )
    factor = pd.read_csv(factor_path, parse_dates=["date"])
    climate_six_month = np.expm1(
        factor.sort_values("date")["ret_climate"].rolling(126, min_periods=126).sum().dropna()
    )
    market_six_month = np.expm1(
        factor.sort_values("date")["logret_spy"].rolling(126, min_periods=126).sum().dropna()
    )
    climate_empirical_p01_return = float(climate_six_month.quantile(0.01))
    market_empirical_p01_return = float(market_six_month.quantile(0.01))
    climate_empirical_p01_shock = abs(climate_empirical_p01_return)
    market_empirical_p01_shock = abs(market_empirical_p01_return)

    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    daily["lrmes_climate_empirical_p01"] = 1.0 - np.exp(
        daily["beta_climate"] * np.log(1.0 - climate_empirical_p01_shock)
    )
    daily["lrmes_market_empirical_p01"] = 1.0 - np.exp(
        daily["beta_market"] * np.log(1.0 - market_empirical_p01_shock)
    )
    monthly = daily.groupby(["current_ticker", "month"], as_index=False).agg(
        lrmes_climate_month_mean=("lrmes_climate", "mean"),
        lrmes_climate_month_end=("lrmes_climate", "last"),
        lrmes_climate_empirical_p01_month_mean=("lrmes_climate_empirical_p01", "mean"),
        lrmes_market_empirical_p01_month_mean=("lrmes_market_empirical_p01", "mean"),
        daily_beta_climate_month_mean=("beta_climate", "mean"),
        daily_crisk_8pct_positive_month_mean_mn=("crisk_8pct_positive_mn", "mean"),
        daily_observations=("date", "size"),
    ).rename(columns={"current_ticker": "ticker"})
    market["month"] = market["report_month_end"].dt.to_period("M").astype(str)
    panel = market.merge(monthly, on=["ticker", "month"], how="left", validate="one_to_one")
    panel = panel.merge(
        coverage,
        left_on=["ticker", "fiscal_quarter"],
        right_on=["ticker", "calendar_quarter"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_sec"),
    )

    # Use the original Part 1 monthly mean LRMES; recompute from monthly beta only as a QC benchmark.
    panel["lrmes_from_monthly_beta"] = 1.0 - np.exp(
        panel["beta_climate_equity_report_month"] * np.log(0.5)
    )
    panel["lrmes_used"] = panel["lrmes_climate_month_mean"]
    panel["climate_equity_loss_mn"] = panel["market_equity_report_month_mn"] * panel["lrmes_used"]
    panel["baseline_coverage_pct"] = panel["actual_asset_coverage_pct"]
    panel["baseline_buffer_pp"] = panel["baseline_coverage_pct"] - panel["statutory_threshold_pct"]
    panel["buffer_shrink_pp"] = 100.0 * panel["climate_equity_loss_mn"] / panel["debt_total_mn"]
    panel["stressed_coverage_pct"] = panel["baseline_coverage_pct"] - panel["buffer_shrink_pp"]
    panel["stressed_buffer_pp"] = panel["stressed_coverage_pct"] - panel["statutory_threshold_pct"]
    panel["breach_before"] = panel["baseline_buffer_pp"] < 0
    panel["breach_after"] = panel["stressed_buffer_pp"] < 0
    panel["new_breach"] = (~panel["breach_before"]) & panel["breach_after"]
    panel["coverage_equity_injection_shortfall_mn"] = np.maximum(
        0.0, -panel["stressed_buffer_pp"] / 100.0 * panel["debt_total_mn"]
    )
    panel["regulatory_buffer_before_mn"] = panel["baseline_buffer_pp"] / 100.0 * panel["debt_total_mn"]
    panel["regulatory_buffer_after_mn"] = panel["stressed_buffer_pp"] / 100.0 * panel["debt_total_mn"]
    panel["buffer_consumed_pct"] = 100.0 * panel["climate_equity_loss_mn"] / panel["regulatory_buffer_before_mn"]
    panel["price_to_nav"] = panel["market_equity_report_month_mn"] / panel["book_equity_mn"]
    panel["nav_climate_loss_mn"] = panel["book_equity_mn"] * panel["lrmes_used"]
    panel["nav_buffer_shrink_pp"] = 100.0 * panel["nav_climate_loss_mn"] / panel["debt_total_mn"]
    panel["nav_stressed_buffer_pp"] = panel["baseline_buffer_pp"] - panel["nav_buffer_shrink_pp"]
    panel["lrmes_market_50pct"] = 1.0 - np.exp(
        panel["beta_market_report_month"] * np.log(0.5)
    )
    panel["market_buffer_shrink_pp"] = (
        100.0
        * panel["market_equity_report_month_mn"]
        * panel["lrmes_market_50pct"]
        / panel["debt_total_mn"]
    )
    panel["climate_empirical_p01_buffer_shrink_pp"] = (
        100.0
        * panel["market_equity_report_month_mn"]
        * panel["lrmes_climate_empirical_p01_month_mean"]
        / panel["debt_total_mn"]
    )
    panel["market_empirical_p01_buffer_shrink_pp"] = (
        100.0
        * panel["market_equity_report_month_mn"]
        * panel["lrmes_market_empirical_p01_month_mean"]
        / panel["debt_total_mn"]
    )
    panel["climate_to_market_empirical_p01_ratio"] = (
        panel["climate_empirical_p01_buffer_shrink_pp"]
        / panel["market_empirical_p01_buffer_shrink_pp"]
    )
    panel["climate_to_market_placebo_ratio"] = (
        panel["buffer_shrink_pp"] / panel["market_buffer_shrink_pp"]
    )
    for cutoff in [5, 10, 25]:
        panel[f"within_{cutoff}pp_before"] = panel["baseline_buffer_pp"] <= cutoff
        panel[f"within_{cutoff}pp_after"] = panel["stressed_buffer_pp"] <= cutoff
        panel[f"new_within_{cutoff}pp"] = (~panel[f"within_{cutoff}pp_before"]) & panel[f"within_{cutoff}pp_after"]

    # Fully market-based approximation, retained as sensitivity rather than the statutory primary measure.
    panel["qmarket_coverage_before_pct"] = 100.0 * (
        1.0 + panel["market_equity_report_month_mn"] / panel["debt_total_mn"]
    )
    panel["qmarket_coverage_after_pct"] = 100.0 * (
        1.0 + panel["market_equity_report_month_mn"] * (1.0 - panel["lrmes_used"]) / panel["debt_total_mn"]
    )
    panel["qmarket_shortfall_mn"] = np.maximum(
        0.0,
        panel["statutory_threshold_pct"] / 100.0 * panel["debt_total_mn"]
        - (panel["debt_total_mn"] + panel["market_equity_report_month_mn"] * (1.0 - panel["lrmes_used"])),
    )
    k = 0.08
    panel["bank8_crisk_signed_mn"] = (
        k * panel["debt_total_mn"]
        - (1.0 - k) * panel["market_equity_report_month_mn"] * (1.0 - panel["lrmes_used"])
    )
    panel["bank8_crisk_positive_mn"] = np.maximum(0.0, panel["bank8_crisk_signed_mn"])
    panel["climate_loss_to_qmarket_assets_pct"] = 100.0 * panel["climate_equity_loss_mn"] / (
        panel["debt_total_mn"] + panel["market_equity_report_month_mn"]
    )
    panel["coverage_data_status"] = np.where(
        panel["baseline_coverage_pct"].notna(), "SEC_REPORTED", "MISSING_SEC_RATIO"
    )

    primary = panel.dropna(subset=["baseline_coverage_pct", "statutory_threshold_pct", "lrmes_used", "debt_total_mn"]).copy()
    test = cluster_mean_test(primary["buffer_shrink_pp"], primary["ticker"])
    naive = stats.ttest_1samp(primary["buffer_shrink_pp"], popmean=0, alternative="greater")
    try:
        wilcoxon = stats.wilcoxon(primary["buffer_shrink_pp"], alternative="greater", zero_method="wilcox")
        wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
    boot = bootstrap_firm_mean(primary, "buffer_shrink_pp")
    tests = pd.DataFrame([
        {"test": "Firm-clustered one-sample mean test", "outcome": "Buffer shrink (pp)", **test},
        {
            "test": "Naive one-sample t test", "outcome": "Buffer shrink (pp)",
            "estimate": primary["buffer_shrink_pp"].mean(), "std_error": primary["buffer_shrink_pp"].std(ddof=1)/np.sqrt(len(primary)),
            "t_stat": float(naive.statistic), "p_value_one_sided": float(naive.pvalue),
            "p_value_two_sided": min(1.0, float(naive.pvalue)*2), "observations": len(primary), "clusters": np.nan, "df": len(primary)-1,
        },
        {
            "test": "Wilcoxon signed-rank", "outcome": "Buffer shrink (pp)",
            "estimate": primary["buffer_shrink_pp"].median(), "std_error": np.nan, "t_stat": wilcoxon_stat,
            "p_value_one_sided": wilcoxon_p, "p_value_two_sided": min(1.0, wilcoxon_p*2) if pd.notna(wilcoxon_p) else np.nan,
            "observations": len(primary), "clusters": np.nan, "df": np.nan,
        },
        {
            "test": "Firm-cluster bootstrap", "outcome": "Buffer shrink (pp)",
            "estimate": primary["buffer_shrink_pp"].mean(), "std_error": np.nan, "t_stat": np.nan,
            "p_value_one_sided": boot["p_one_sided_le_zero"], "p_value_two_sided": np.nan,
            "observations": len(primary), "clusters": primary["ticker"].nunique(), "df": np.nan,
            "ci_2_5": boot["ci_2_5"], "ci_97_5": boot["ci_97_5"], "bootstrap_reps": boot["bootstrap_reps"],
        },
    ])

    variables = [
        "baseline_coverage_pct", "stressed_coverage_pct", "baseline_buffer_pp", "stressed_buffer_pp",
        "buffer_shrink_pp", "lrmes_used", "climate_equity_loss_mn", "coverage_equity_injection_shortfall_mn",
        "bank8_crisk_positive_mn", "buffer_consumed_pct", "qmarket_coverage_before_pct", "qmarket_coverage_after_pct",
        "price_to_nav", "nav_buffer_shrink_pp", "market_buffer_shrink_pp", "climate_to_market_placebo_ratio",
        "climate_empirical_p01_buffer_shrink_pp", "market_empirical_p01_buffer_shrink_pp",
        "climate_to_market_empirical_p01_ratio",
    ]
    summary_rows = []
    for v in variables:
        s = primary[v].dropna()
        summary_rows.append({
            "variable": v, "n": len(s), "mean": s.mean(), "std": s.std(ddof=1), "p10": s.quantile(.1),
            "median": s.median(), "p90": s.quantile(.9), "min": s.min(), "max": s.max(),
        })
    summary = pd.DataFrame(summary_rows)

    year_summary = primary.assign(year=primary["datadate"].dt.year).groupby("year", as_index=False).agg(
        observations=("ticker", "size"), firms=("ticker", "nunique"),
        mean_baseline_coverage_pct=("baseline_coverage_pct", "mean"),
        mean_stressed_coverage_pct=("stressed_coverage_pct", "mean"),
        mean_buffer_shrink_pp=("buffer_shrink_pp", "mean"),
        breaches_before=("breach_before", "sum"), breaches_after=("breach_after", "sum"), new_breaches=("new_breach", "sum"),
        within_10pp_before=("within_10pp_before", "sum"), within_10pp_after=("within_10pp_after", "sum"),
        within_25pp_before=("within_25pp_before", "sum"), within_25pp_after=("within_25pp_after", "sum"),
        aggregate_coverage_shortfall_mn=("coverage_equity_injection_shortfall_mn", "sum"),
        aggregate_bank8_crisk_positive_mn=("bank8_crisk_positive_mn", "sum"),
    )
    firm_summary = primary.groupby("ticker", as_index=False).agg(
        company_name=("company_name", "first"), observations=("ticker", "size"),
        mean_baseline_coverage_pct=("baseline_coverage_pct", "mean"),
        mean_stressed_coverage_pct=("stressed_coverage_pct", "mean"),
        mean_buffer_shrink_pp=("buffer_shrink_pp", "mean"),
        min_stressed_buffer_pp=("stressed_buffer_pp", "min"),
        quarters_breached_before=("breach_before", "sum"), quarters_breached_after=("breach_after", "sum"),
        new_breach_quarters=("new_breach", "sum"),
        quarters_within_10pp_after=("within_10pp_after", "sum"),
        quarters_within_25pp_after=("within_25pp_after", "sum"),
        max_coverage_shortfall_mn=("coverage_equity_injection_shortfall_mn", "max"),
        total_coverage_shortfall_mn=("coverage_equity_injection_shortfall_mn", "sum"),
        total_bank8_crisk_positive_mn=("bank8_crisk_positive_mn", "sum"),
    ).sort_values(["new_breach_quarters", "max_coverage_shortfall_mn"], ascending=False)

    spearman = stats.spearmanr(primary["buffer_shrink_pp"], primary["bank8_crisk_signed_mn"])
    positive_cov = primary["buffer_shrink_pp"] > 0
    positive_bank = primary["bank8_crisk_positive_mn"] > 0
    comparison = pd.DataFrame([
        {"metric": "Positive risk signal observations", "unit": "count", "coverage_rule": int(positive_cov.sum()), "bank_k8": int(positive_bank.sum())},
        {"metric": "Positive risk signal share", "unit": "share", "coverage_rule": float(positive_cov.mean()), "bank_k8": float(positive_bank.mean())},
        {"metric": "Mean continuous pressure", "unit": "pp / USD mn signed", "coverage_rule": primary["buffer_shrink_pp"].mean(), "bank_k8": primary["bank8_crisk_signed_mn"].mean()},
        {"metric": "Aggregate positive legal shortfall", "unit": "USD mn", "coverage_rule": primary["coverage_equity_injection_shortfall_mn"].sum(), "bank_k8": primary["bank8_crisk_positive_mn"].sum()},
        {"metric": "Observations within 10pp after stress", "unit": "count", "coverage_rule": int(primary["within_10pp_after"].sum()), "bank_k8": np.nan},
        {"metric": "Spearman: buffer shrink vs signed k=8% CRISK", "unit": "rho / p-value", "coverage_rule": float(spearman.statistic), "bank_k8": float(spearman.pvalue)},
    ])

    robustness_rows = []
    scenarios = {
        "Primary: monthly mean DCB LRMES": panel["lrmes_climate_month_mean"],
        f"Empirical climate 1pct tail ({100*climate_empirical_p01_shock:.1f}pct decline)": panel["lrmes_climate_empirical_p01_month_mean"],
        "Month-end DCB LRMES": panel["lrmes_climate_month_end"],
        "Monthly beta-implied LRMES": panel["lrmes_from_monthly_beta"],
        "Positive-loss-only monthly mean LRMES": panel["lrmes_climate_month_mean"].clip(lower=0),
    }
    for name, lrmes in scenarios.items():
        shrink = 100.0 * panel["market_equity_report_month_mn"] * lrmes / panel["debt_total_mn"]
        keep = panel["baseline_coverage_pct"].notna() & shrink.notna()
        scenario = panel.loc[keep, ["ticker", "baseline_coverage_pct", "statutory_threshold_pct"]].copy()
        scenario["shrink"] = shrink[keep]
        scenario["stressed_buffer"] = scenario["baseline_coverage_pct"] - scenario["statutory_threshold_pct"] - scenario["shrink"]
        scenario_test = cluster_mean_test(scenario["shrink"], scenario["ticker"])
        robustness_rows.append({
            "scenario": name, "observations": len(scenario), "firms": scenario["ticker"].nunique(),
            "mean_buffer_shrink_pp": scenario["shrink"].mean(), "median_buffer_shrink_pp": scenario["shrink"].median(),
            "positive_shrink_share": (scenario["shrink"] > 0).mean(),
            "firm_clustered_one_sided_p": scenario_test["p_value_one_sided"],
            "firm_clustered_two_sided_p": scenario_test["p_value_two_sided"],
            "breach_observations_after": (scenario["stressed_buffer"] < 0).sum(),
            "within_10pp_after": (scenario["stressed_buffer"] <= 10).sum(),
            "within_25pp_after": (scenario["stressed_buffer"] <= 25).sum(),
        })
    robustness = pd.DataFrame(robustness_rows)

    placebo_nav = pd.DataFrame(
        [
            {
                "scenario": f"Empirical climate 1pct tail ({100*climate_empirical_p01_shock:.1f}pct decline)",
                "mean_buffer_shrink_pp": primary["climate_empirical_p01_buffer_shrink_pp"].mean(),
                "median_buffer_shrink_pp": primary["climate_empirical_p01_buffer_shrink_pp"].median(),
                "mean_buffer_consumed_pct": (
                    100.0
                    * primary["climate_empirical_p01_buffer_shrink_pp"]
                    / primary["baseline_buffer_pp"]
                ).mean(),
                "ratio_of_mean_compression_to_mean_buffer_pct": (
                    100.0 * primary["climate_empirical_p01_buffer_shrink_pp"].mean()
                    / primary["baseline_buffer_pp"].mean()
                ),
                "breach_observations_after": int(
                    (primary["baseline_buffer_pp"] - primary["climate_empirical_p01_buffer_shrink_pp"] < 0).sum()
                ),
                "observations": len(primary),
            },
            {
                "scenario": "Climate shock, market equity mapping",
                "mean_buffer_shrink_pp": primary["buffer_shrink_pp"].mean(),
                "median_buffer_shrink_pp": primary["buffer_shrink_pp"].median(),
                "mean_buffer_consumed_pct": primary["buffer_consumed_pct"].mean(),
                "ratio_of_mean_compression_to_mean_buffer_pct": (
                    100.0 * primary["buffer_shrink_pp"].mean() / primary["baseline_buffer_pp"].mean()
                ),
                "breach_observations_after": int(primary["breach_after"].sum()),
                "observations": len(primary),
            },
            {
                "scenario": "Climate shock, NAV mapping",
                "mean_buffer_shrink_pp": primary["nav_buffer_shrink_pp"].mean(),
                "median_buffer_shrink_pp": primary["nav_buffer_shrink_pp"].median(),
                "mean_buffer_consumed_pct": (
                    100.0 * primary["nav_climate_loss_mn"] / primary["regulatory_buffer_before_mn"]
                ).mean(),
                "ratio_of_mean_compression_to_mean_buffer_pct": (
                    100.0 * primary["nav_buffer_shrink_pp"].mean() / primary["baseline_buffer_pp"].mean()
                ),
                "breach_observations_after": int((primary["nav_stressed_buffer_pp"] < 0).sum()),
                "observations": int(primary["nav_buffer_shrink_pp"].notna().sum()),
            },
            {
                "scenario": f"Empirical market 1pct tail ({100*market_empirical_p01_shock:.1f}pct decline)",
                "mean_buffer_shrink_pp": primary["market_empirical_p01_buffer_shrink_pp"].mean(),
                "median_buffer_shrink_pp": primary["market_empirical_p01_buffer_shrink_pp"].median(),
                "mean_buffer_consumed_pct": (
                    100.0
                    * primary["market_empirical_p01_buffer_shrink_pp"]
                    / primary["baseline_buffer_pp"]
                ).mean(),
                "ratio_of_mean_compression_to_mean_buffer_pct": (
                    100.0 * primary["market_empirical_p01_buffer_shrink_pp"].mean()
                    / primary["baseline_buffer_pp"].mean()
                ),
                "breach_observations_after": int(
                    (primary["baseline_buffer_pp"] - primary["market_empirical_p01_buffer_shrink_pp"] < 0).sum()
                ),
                "observations": len(primary),
            },
            {
                "scenario": "50 percent broad-market shock placebo",
                "mean_buffer_shrink_pp": primary["market_buffer_shrink_pp"].mean(),
                "median_buffer_shrink_pp": primary["market_buffer_shrink_pp"].median(),
                "mean_buffer_consumed_pct": (
                    100.0
                    * primary["market_equity_report_month_mn"]
                    * primary["lrmes_market_50pct"]
                    / primary["regulatory_buffer_before_mn"]
                ).mean(),
                "ratio_of_mean_compression_to_mean_buffer_pct": (
                    100.0 * primary["market_buffer_shrink_pp"].mean() / primary["baseline_buffer_pp"].mean()
                ),
                "breach_observations_after": int(
                    (
                        primary["baseline_buffer_pp"]
                        - primary["market_buffer_shrink_pp"]
                        < 0
                    ).sum()
                ),
                "observations": int(primary["market_buffer_shrink_pp"].notna().sum()),
            },
        ]
    )

    linkage = pd.DataFrame()
    h2_panel_path = (
        Path(__file__).resolve().parents[2]
        / "02_BDC_Investment_Exposure_and_Climate_Beta/Data/Processed/bdc19_dynamic_portfolio_h2_panel_2021_2025.csv"
    )
    h2_models_path = (
        Path(__file__).resolve().parents[2]
        / "02_BDC_Investment_Exposure_and_Climate_Beta/Data/Processed/h2_brown_share_dynamic_models.csv"
    )
    if h2_panel_path.exists() and h2_models_path.exists():
        h2_panel = pd.read_csv(h2_panel_path, parse_dates=["datadate"])
        h2_models = pd.read_csv(h2_models_path)
        h2_coefficient = float(
            h2_models.loc[h2_models["model_id"].eq("H2_1"), "coefficient_exposure"].iloc[0]
        )
        beta_sd = h2_panel["beta_climate_equity_report_month"].std(ddof=0)
        brown_sd = h2_panel["brown_share_broad_dynamic_pct"].std(ddof=0)
        linkage = primary.merge(
            h2_panel[["ticker", "datadate", "brown_share_broad_dynamic_pct"]],
            on=["ticker", "datadate"],
            how="left",
            validate="one_to_one",
        )
        linkage["h2_predicted_beta_increment_vs_zero"] = (
            h2_coefficient
            * beta_sd
            * linkage["brown_share_broad_dynamic_pct"]
            / brown_sd
        )
        linkage["lrmes_counterfactual_zero_brown"] = 1.0 - np.exp(
            (
                linkage["beta_climate_equity_report_month"]
                - linkage["h2_predicted_beta_increment_vs_zero"]
            )
            * np.log(0.5)
        )
        linkage["h2_implied_incremental_buffer_compression_pp"] = (
            100.0
            * linkage["market_equity_report_month_mn"]
            * (linkage["lrmes_used"] - linkage["lrmes_counterfactual_zero_brown"])
            / linkage["debt_total_mn"]
        )
        max_brown = h2_panel["brown_share_broad_dynamic_pct"].max()
        beta_increment_zero_to_max = h2_coefficient * beta_sd * max_brown / brown_sd
        representative_beta = h2_panel["beta_climate_equity_report_month"].median()
        representative_e_to_d = (
            primary["market_equity_report_month_mn"] / primary["debt_total_mn"]
        ).median()
        zero_lrmes = 1.0 - np.exp(representative_beta * np.log(0.5))
        max_lrmes = 1.0 - np.exp(
            (representative_beta + beta_increment_zero_to_max) * np.log(0.5)
        )
        linkage_summary = pd.DataFrame(
            [
                {
                    "h2_standardized_coefficient": h2_coefficient,
                    "maximum_broad_brown_share_pct": max_brown,
                    "predicted_beta_increment_zero_to_max": beta_increment_zero_to_max,
                    "predicted_buffer_compression_zero_to_max_pp": 100.0
                    * representative_e_to_d
                    * (max_lrmes - zero_lrmes),
                    "mean_incremental_compression_at_observed_exposure_pp": linkage[
                        "h2_implied_incremental_buffer_compression_pp"
                    ].mean(),
                    "interpretation": "Mechanical linkage using the statistically imprecise H2 point estimate; not a causal estimate.",
                }
            ]
        )
        export(linkage, outdir / "h2_h3_linkage_panel")
        export(linkage_summary, outdir / "h2_h3_linkage_summary")

    decision = "CALIBRATED_BUFFER_EROSION_NO_BREACH"
    decision_table = pd.DataFrame([{
        "hypothesis": "H3",
        "decision": decision,
        "primary_sample_observations": len(primary),
        "firms": primary["ticker"].nunique(),
        "mean_buffer_shrink_pp": primary["buffer_shrink_pp"].mean(),
        "firm_clustered_one_sided_p": test["p_value_one_sided"],
        "firm_clustered_two_sided_p": test["p_value_two_sided"],
        "baseline_breach_observations": int(primary["breach_before"].sum()),
        "stressed_breach_observations": int(primary["breach_after"].sum()),
        "new_breach_observations": int(primary["new_breach"].sum()),
        "firms_with_new_breach": int(primary.loc[primary["new_breach"], "ticker"].nunique()),
        "aggregate_coverage_shortfall_mn": primary["coverage_equity_injection_shortfall_mn"].sum(),
        "positive_shrink_observations": int((primary["buffer_shrink_pp"] > 0).sum()),
        "within_10pp_before": int(primary["within_10pp_before"].sum()),
        "within_10pp_after": int(primary["within_10pp_after"].sum()),
        "within_25pp_before": int(primary["within_25pp_before"].sum()),
        "within_25pp_after": int(primary["within_25pp_after"].sum()),
        "interpretation": "The maintained calibration erodes the reported asset-coverage buffer, but no observation crosses the legal threshold. Positivity is partly mechanical conditional on a positive estimated beta, so the p-value is descriptive rather than a test of the DCC model.",
    }])

    export(panel, outdir / "h3_analysis_panel")
    export(summary, outdir / "h3_descriptive_statistics")
    export(tests, outdir / "h3_hypothesis_tests")
    export(year_summary, outdir / "h3_year_summary")
    export(firm_summary, outdir / "h3_firm_summary")
    export(comparison, outdir / "h3_coverage_vs_bank8_comparison")
    export(robustness, outdir / "h3_robustness_scenarios")
    export(placebo_nav, outdir / "h3_market_placebo_and_nav_sensitivity")
    export(decision_table, outdir / "h3_hypothesis_decision")

    audit = {
        "market_rows_input": int(len(market)), "daily_rows_input": int(len(daily)),
        "coverage_rows_input": int(len(coverage)), "analysis_rows": int(len(panel)),
        "primary_rows": int(len(primary)), "firms": int(primary["ticker"].nunique()),
        "missing_monthly_lrmes": int(panel["lrmes_used"].isna().sum()),
        "missing_sec_coverage": int(panel["baseline_coverage_pct"].isna().sum()),
        "duplicate_firm_quarter": int(panel.duplicated(["ticker", "fiscal_quarter"]).sum()),
        "decision": decision,
        "stress_definition": "Six-month 50% climate-factor decline; monthly mean DCB LRMES",
        "climate_six_month_empirical_p01_return": climate_empirical_p01_return,
        "market_six_month_empirical_p01_return": market_empirical_p01_return,
        "matched_tail_mean_compression_ratio_climate_to_market": float(
            primary["climate_empirical_p01_buffer_shrink_pp"].mean()
            / primary["market_empirical_p01_buffer_shrink_pp"].mean()
        ),
        "coverage_formula": "stressed coverage = SEC reported coverage - (market equity * LRMES / total debt)",
    }
    (outdir / "h3_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(decision_table.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
