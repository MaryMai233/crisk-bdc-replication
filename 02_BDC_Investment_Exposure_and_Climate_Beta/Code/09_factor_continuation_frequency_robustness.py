from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crisk_bdc_v21_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = PROCESSED / "Audit"
MODULE1 = PACKAGE / "01_Bank_CRISK_Replication"
BLUE = "#08457E"
ORANGE = "#F28E00"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DCC = load_module(MODULE1 / "Code" / "02_estimate_dcb_crisk.py", "crisk_dcc_v21")
H2 = load_module(Path(__file__).with_name("01_industry_beta_helpers.py"), "crisk_h2_helper_v21")


def esc(value: object) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def rtf_row(cells, widths, *, bold=False, top=False, bottom=False):
    out = [r"\trowd\trgaph60\trleft0"]
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


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    H2.export_dta(frame, path)


def reestimate_weekly_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "dcb_crisk_daily_2010_2025.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    baseline = pd.read_csv(
        PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv",
        parse_dates=["datadate"],
    )
    tickers = sorted(baseline["ticker"].unique())
    daily = daily[daily["group"].eq("BDC") & daily["current_ticker"].isin(tickers)].copy()
    daily["institution_log_return"] = np.log1p(daily["ret"].clip(lower=-0.999999))

    first_pass = {}
    parameter_rows = []
    for ticker, frame in daily.groupby("current_ticker"):
        weekly = (
            frame.set_index("date")[["institution_log_return", "logret_spy", "ret_climate"]]
            .resample("W-FRI")
            .agg(["sum", "count"])
        )
        weekly.columns = ["_".join(item) for item in weekly.columns]
        weekly = weekly[
            weekly["institution_log_return_count"].ge(3)
            & weekly["logret_spy_count"].ge(3)
            & weekly["ret_climate_count"].ge(3)
        ].copy()
        raw = weekly[["institution_log_return_sum", "logret_spy_sum", "ret_climate_sum"]].to_numpy(float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        gjr_fits = [DCC.fit_gjr(centered[:, index]) for index in range(3)]
        h = np.column_stack([fit.variance for fit in gjr_fits])
        z = np.column_stack([fit.standardized for fit in gjr_fits])
        dcc_parameters, dcc_success, objective = DCC.fit_dcc(z)
        first_pass[ticker] = {"weekly": weekly, "h": h, "z": z}
        parameter_rows.append(
            {
                "ticker": ticker,
                "weekly_observations": len(weekly),
                "dcc_alpha_first_pass": dcc_parameters[0],
                "dcc_beta_first_pass": dcc_parameters[1],
                "dcc_success": dcc_success,
                "dcc_objective": objective,
                "gjr_all_success": all(item.success for item in gjr_fits),
            }
        )
        print(f"Weekly DCC {ticker}: T={len(weekly)}, success={dcc_success}", flush=True)

    parameters = pd.DataFrame(parameter_rows)
    median_dcc = parameters.loc[
        parameters["dcc_success"], ["dcc_alpha_first_pass", "dcc_beta_first_pass"]
    ].median().to_numpy()
    if median_dcc.sum() >= 0.999:
        raise RuntimeError("Weekly median DCC persistence is outside the maintained stationary region")

    weekly_pieces = []
    for ticker, item in first_pass.items():
        _, correlations = DCC.dcc_path(item["z"], median_dcc, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(item["h"], correlations)
        output = item["weekly"].reset_index()[["date"]].copy()
        output["ticker"] = ticker
        output["beta_market_weekly"] = beta_market
        output["beta_climate_equity_weekly"] = beta_climate
        output["dcc_alpha_median"] = median_dcc[0]
        output["dcc_beta_median"] = median_dcc[1]
        weekly_pieces.append(output)
    weekly_betas = pd.concat(weekly_pieces, ignore_index=True).sort_values(["ticker", "date"])
    weekly_betas["quarter"] = weekly_betas["date"].dt.to_period("Q").astype(str)
    quarter = weekly_betas.groupby(["ticker", "quarter"], as_index=False).agg(
        beta_climate_equity_weekly_qmean=("beta_climate_equity_weekly", "mean"),
        beta_market_weekly_qmean=("beta_market_weekly", "mean"),
        weekly_observations=("date", "size"),
    )
    panel = baseline.merge(quarter, on=["ticker", "quarter"], how="left", validate="one_to_one")
    panel["beta_climate_asset_weekly_qmean"] = (
        panel["beta_climate_equity_weekly_qmean"]
        * panel["market_equity_report_month_mn"]
        / (panel["market_equity_report_month_mn"] + panel["debt_total_mn"])
    )
    for column in ["beta_climate_equity_weekly_qmean", "beta_climate_asset_weekly_qmean"]:
        standard_deviation = panel[column].std(ddof=0)
        panel[f"z_{column}"] = (panel[column] - panel[column].mean()) / standard_deviation
    parameters["dcc_alpha_median"] = median_dcc[0]
    parameters["dcc_beta_median"] = median_dcc[1]
    parameters["dcc_persistence_median"] = median_dcc.sum()
    return weekly_betas, panel, parameters


def estimate_variant(panel: pd.DataFrame, outcome: str, model_id: str, label: str) -> dict:
    return H2.fit_ols(
        panel,
        outcome,
        "z_brown_share_broad_dynamic_pct",
        model_id,
        label,
        fixed_effects=["quarter"],
        cluster="ticker",
    )


def horizon_correlations() -> pd.DataFrame:
    top5 = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "climate_factor_daily_2010_2025.csv",
        parse_dates=["date"],
    ).set_index("date")
    broad = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "climate_factor_daily_us_coal_alternative_2010_2025.csv",
        parse_dates=["date"],
    ).set_index("date")

    rows = []
    definitions = [
        ("Published top-five continuation", top5, "top5_usd_logret"),
        ("U.S. coal value-weighted continuation", broad, "logret_coal_vw"),
    ]
    windows = [
        ("Full pre-liquidation overlap", "2010-01-01", "2020-12-14"),
        ("Common recent overlap", "2019-01-01", "2020-12-14"),
    ]
    for definition, frame, proxy in definitions:
        for window, start, end in windows:
            pair = frame.loc[start:end, ["logret_kol", proxy]].dropna()
            for frequency, rule in [("Daily", None), ("Weekly", "W-FRI"), ("Monthly", "ME")]:
                sample = pair if rule is None else pair.resample(rule).sum(min_count=1).dropna()
                rows.append(
                    {
                        "continuation": definition,
                        "overlap_window": window,
                        "frequency": frequency,
                        "observations": len(sample),
                        "correlation_with_kol": sample["logret_kol"].corr(sample[proxy]),
                    }
                )
    return pd.DataFrame(rows)


def build_rtf(models: pd.DataFrame) -> None:
    def starred(value: float, p_value: float) -> str:
        mark = "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""
        return f"{value:.3f}{mark}"

    indexed = models.set_index("model_id")
    panel_a_ids = ["F1", "F2", "F3", "F4", "F5", "F6"]
    selected = indexed.loc[panel_a_ids]
    rows_a = [
        ["", "Top-five daily", "Top-five daily", "U.S. coal daily", "U.S. coal daily", "Top-five weekly", "Top-five weekly"],
        ["Broad investment share", *[
            starred(value, p_value)
            for value, p_value in zip(selected["coefficient_exposure"], selected["p_two_sided"])
        ]],
        ["", *[f"({x:.3f})" for x in selected["standard_error"]]],
        ["Outcome", "Equity", "Asset", "Equity", "Asset", "Equity", "Asset"],
        ["Frequency", "Daily", "Daily", "Daily", "Daily", "Weekly", "Weekly"],
        ["Observations", *[str(int(x)) for x in selected["n"]]],
    ]

    main_models = pd.read_csv(PROCESSED / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    inference = pd.read_csv(PROCESSED / "h2_inference_robustness_models.csv").set_index("model_id")
    wild = pd.read_csv(PROCESSED / "h2_wild_cluster_bootstrap.csv")
    power = pd.read_csv(PROCESSED / "h2_power_diagnostics.csv").set_index("model_id")
    equity_items = [main_models.loc["H2_7"], inference.loc["R1"], inference.loc["R2"], inference.loc["R5"], inference.loc["R7"]]
    asset_items = [main_models.loc["H2_8"], inference.loc["R3"], inference.loc["R4"], inference.loc["R6"], inference.loc["R8"]]
    rows_b = [
        ["", "Narrow", "Lower", "Upper", "Lagged", "Post-2021", "MDE80"],
        ["Equity beta", *[
            starred(x["coefficient_exposure"], x["p_two_sided"]) for x in equity_items
        ], f"{power.loc['H2_1','minimum_detectable_effect_80pct_power']:.3f}"],
        ["", *[f"({x['standard_error']:.3f})" for x in equity_items], ""],
        ["Asset beta", *[
            starred(x["coefficient_exposure"], x["p_two_sided"]) for x in asset_items
        ], f"{power.loc['H2_2','minimum_detectable_effect_80pct_power']:.3f}"],
        ["", *[f"({x['standard_error']:.3f})" for x in asset_items], ""],
        ["Observations", *[str(int(x['n'])) for x in equity_items], "380"],
    ]

    credit = pd.read_csv(PROCESSED / "h2_credit_return_robustness_models.csv").set_index("model_id")
    credit_items = credit.loc[["C1", "C2", "C3", "C4", "C5", "C6"]]
    rows_c = [
        ["", "Baseline equity", "HY equity", "Baseline asset", "HY asset", "HY + controls", "HY + BDC FE"],
        ["Broad investment share", *[
            starred(value, p_value)
            for value, p_value in zip(credit_items["coefficient_exposure"], credit_items["p_two_sided"])
        ]],
        ["", *[f"({x:.3f})" for x in credit_items["standard_error"]]],
        ["Observations", *[str(int(x)) for x in credit_items["n"]]],
    ]

    note = (
        "Panel A compares the published top-five continuation at daily frequency, the earlier U.S. coal value-weighted continuation, and a weekly DCC estimate of the published continuation. "
        "All outcomes and exposures are standardized within the corresponding 19-BDC panel; every regression includes quarter fixed effects and clusters standard errors by BDC. "
        "Panel B reports classification, timing, and power checks. Panel C reuses the archived Table 3 outcomes and compares them with a separately re-estimated high-yield-return-adjusted DCC. Standard errors appear in parentheses; ***, **, and * denote two-sided significance at 1%, 5%, and 10%. No coefficient reaches conventional significance."
    )
    body = [
        r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs19\landscape\paperw15840\paperh12240\margl520\margr520\margt650\margb650",
        r"\pard\sa30\b\fs23 Table 4: Factor Continuation, Frequency, and H2 Robustness.\b0\par",
        r"\pard\sa20\b Panel A: Factor continuation and return frequency\b0\par",
    ]
    for index, cells in enumerate(rows_a):
        body.append(rtf_row(cells, [2750] + [1750] * 6, bold=index == 0, top=index == 0, bottom=index == len(rows_a) - 1))
    body.append(r"\pard\sa45\b Panel B: Classification, timing, and inference\b0\par")
    for index, cells in enumerate(rows_b):
        body.append(rtf_row(cells, [2450] + [1700] * 6, bold=index == 0, top=index == 0, bottom=index == len(rows_b) - 1))
    body.append(r"\pard\sa45\b Panel C: High-yield credit-return factor\b0\par")
    for index, cells in enumerate(rows_c):
        body.append(rtf_row(cells, [2750] + [1750] * 6, bold=index == 0, top=index == 0, bottom=index == len(rows_c) - 1))
    body.extend([r"\pard\sa70\fs16 " + esc("Notes: " + note) + r"\par", "}"])
    (RESULTS / "Table_4_Factor_Continuation_and_H2_Robustness.rtf").write_text("\n".join(body), encoding="ascii", errors="ignore")


def build_figure(models: pd.DataFrame, attenuation: pd.DataFrame) -> None:
    selected = models.set_index("model_id").loc[["F1", "F2", "F3", "F4", "F5", "F6"]].copy()
    selected["label"] = [
        "Top-five daily: equity",
        "Top-five daily: asset",
        "U.S. coal daily: equity",
        "U.S. coal daily: asset",
        "Top-five weekly: equity",
        "Top-five weekly: asset",
    ]
    y = np.arange(len(selected))[::-1]
    coef = selected["coefficient_exposure"].to_numpy()
    error = 1.96 * selected["standard_error"].to_numpy()
    colors = ["#08457E", "#F28E00"] * 3
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.errorbar(coef, y, xerr=error, fmt="none", ecolor="#666666", elinewidth=1.2, capsize=3)
    ax.scatter(coef, y, c=colors, s=48, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9)
    pooled_mde = pd.read_csv(PROCESSED / "h2_power_diagnostics.csv").set_index("model_id").loc[
        "H2_1", "minimum_detectable_effect_80pct_power"
    ]
    ax.axvline(pooled_mde, color="#8B0000", linestyle="--", linewidth=0.9)
    ax.axvline(-pooled_mde, color="#8B0000", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Standardized H2 coefficient with 95% confidence interval")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax = axes[1]
    observed = attenuation.loc[attenuation["point_type"].eq("Observed")].copy()
    summary = attenuation.loc[attenuation["point_type"].eq("Naive extrapolation")].iloc[0]
    grid = np.linspace(observed["tracking_correlation"].min(), 1.0, 100)
    top_five = observed[observed["specification"].str.startswith("Top-five")].sort_values("tracking_correlation")
    us_coal = observed[observed["specification"].eq("U.S. coal daily")]
    ax.plot(top_five["tracking_correlation"], top_five["h2_equity_coefficient"], color=BLUE, linewidth=1.8,
            marker="o", label="Top-five basket: frequency only", zorder=3)
    ax.scatter(us_coal["tracking_correlation"], us_coal["h2_equity_coefficient"], color=ORANGE, s=58,
               marker="s", label="U.S. coal: content also changes", zorder=4)
    ax.plot(grid, summary["intercept"] + summary["slope"] * grid, color="0.35", linestyle="--", linewidth=1.2)
    ax.scatter([1.0], [summary["h2_equity_coefficient"]], color=ORANGE, s=58, marker="D", zorder=4)
    ax.axhline(pooled_mde, color="#8B0000", linestyle=":", linewidth=0.9)
    for item in observed.itertuples(index=False):
        ax.annotate(item.specification, (item.tracking_correlation, item.h2_equity_coefficient),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0.4, 1.03)
    ax.set_xlabel("Correlation with KOL")
    ax.set_ylabel("Standardized equity-beta coefficient")
    ax.set_title("Tracking and frequency diagnostics", fontsize=11)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "Figure_5_Factor_Continuation_Sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reestimate-weekly", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    weekly_beta_path = PROCESSED / "h2_top5_weekly_betas_2010_2025.csv"
    weekly_panel_path = PROCESSED / "bdc19_top5_weekly_h2_panel_2021_2025.csv"
    parameter_path = PROCESSED / "h2_top5_weekly_dcc_parameters.csv"
    if args.reestimate_weekly or not weekly_panel_path.exists():
        weekly_betas, weekly_panel, parameters = reestimate_weekly_panel()
        weekly_betas.to_csv(weekly_beta_path, index=False)
        export_dta(weekly_betas, weekly_beta_path.with_suffix(".dta"))
        weekly_panel.to_csv(weekly_panel_path, index=False)
        export_dta(weekly_panel, weekly_panel_path.with_suffix(".dta"))
        parameters.to_csv(parameter_path, index=False)
        export_dta(parameters, parameter_path.with_suffix(".dta"))
    else:
        weekly_panel = pd.read_csv(weekly_panel_path)
        parameters = pd.read_csv(parameter_path)

    top5_daily = pd.read_csv(PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv")
    us_coal_daily = pd.read_csv(PROCESSED / "bdc19_us_coal_daily_h2_panel_2021_2025.csv")
    specifications = [
        (top5_daily, "z_beta_climate_equity_report_month", "F1", "Published top-five continuation, daily equity beta"),
        (top5_daily, "z_beta_climate_asset_report_month", "F2", "Published top-five continuation, daily asset beta"),
        (us_coal_daily, "z_beta_climate_equity_report_month", "F3", "U.S. coal value-weighted continuation, daily equity beta"),
        (us_coal_daily, "z_beta_climate_asset_report_month", "F4", "U.S. coal value-weighted continuation, daily asset beta"),
        (weekly_panel, "z_beta_climate_equity_weekly_qmean", "F5", "Published top-five continuation, weekly equity beta"),
        (weekly_panel, "z_beta_climate_asset_weekly_qmean", "F6", "Published top-five continuation, weekly asset beta"),
    ]
    models = pd.DataFrame([estimate_variant(*item) for item in specifications])
    models.to_csv(PROCESSED / "h2_factor_continuation_frequency_models.csv", index=False)
    export_dta(models, PROCESSED / "h2_factor_continuation_frequency_models.dta")

    decision_path = PROCESSED / "h2_dynamic_decision.csv"
    decision = pd.read_csv(decision_path)
    indexed_models = models.set_index("model_id")
    decision["decision_code"] = "NOT_SUPPORTED_FACTOR_AND_FREQUENCY_SENSITIVE"
    decision["conclusion_en"] = (
        "H2 is not statistically supported and its sign is unresolved. Parsimonious estimates are positive and rise with tracking, whereas controlled and fixed-effects estimates can be negative and are too imprecise to interpret by sign. "
        "Holding the top-five basket fixed, weekly aggregation raises the coefficient, a pattern consistent with attenuation from non-synchronous international closes. The U.S. coal alternative also changes economic content and cannot be interpreted as a tracking-only comparison."
    )
    decision["conclusion_cn"] = (
        "H2 未获得统计支持，而且符号没有被数据决定。简约模型为正并随追踪度提高，控制变量与固定效应模型可转为负值，但都过于不精确。"
        "固定 top-five 篮子后，周频系数高于日频，符合非同步交易导致衰减的解释；美国煤炭替代组合同时改变经济内容，不能视为单纯的追踪度比较。"
    )
    decision["us_coal_daily_equity_coefficient"] = indexed_models.loc["F3", "coefficient_exposure"]
    decision["us_coal_daily_asset_coefficient"] = indexed_models.loc["F4", "coefficient_exposure"]
    decision["top5_weekly_equity_coefficient"] = indexed_models.loc["F5", "coefficient_exposure"]
    decision["top5_weekly_asset_coefficient"] = indexed_models.loc["F6", "coefficient_exposure"]
    decision["top5_weekly_equity_p_two_sided"] = indexed_models.loc["F5", "p_two_sided"]
    decision["top5_weekly_asset_p_two_sided"] = indexed_models.loc["F6", "p_two_sided"]
    decision.to_csv(decision_path, index=False)
    export_dta(decision, PROCESSED / "h2_dynamic_decision.dta")

    correlations = horizon_correlations()
    correlations.to_csv(PROCESSED / "h2_factor_tracking_correlations.csv", index=False)
    export_dta(correlations, PROCESSED / "h2_factor_tracking_correlations.dta")
    recent = correlations.loc[correlations["overlap_window"].eq("Common recent overlap")]
    points = pd.DataFrame([
        {
            "point_type": "Observed", "specification": "Top-five daily",
            "comparison_role": "same_basket_frequency",
            "tracking_correlation": recent.loc[
                recent["continuation"].eq("Published top-five continuation") & recent["frequency"].eq("Daily"),
                "correlation_with_kol",
            ].iloc[0],
            "h2_equity_coefficient": indexed_models.loc["F1", "coefficient_exposure"],
        },
        {
            "point_type": "Observed", "specification": "U.S. coal daily",
            "comparison_role": "content_and_synchronization_change",
            "tracking_correlation": recent.loc[
                recent["continuation"].eq("U.S. coal value-weighted continuation") & recent["frequency"].eq("Daily"),
                "correlation_with_kol",
            ].iloc[0],
            "h2_equity_coefficient": indexed_models.loc["F3", "coefficient_exposure"],
        },
        {
            "point_type": "Observed", "specification": "Top-five weekly",
            "comparison_role": "same_basket_frequency",
            "tracking_correlation": recent.loc[
                recent["continuation"].eq("Published top-five continuation") & recent["frequency"].eq("Weekly"),
                "correlation_with_kol",
            ].iloc[0],
            "h2_equity_coefficient": indexed_models.loc["F5", "coefficient_exposure"],
        },
    ])
    slope, intercept = np.polyfit(points["tracking_correlation"], points["h2_equity_coefficient"], 1)
    extrapolated = pd.DataFrame([{
        "point_type": "Naive extrapolation", "specification": "Correlation equals one",
        "tracking_correlation": 1.0, "h2_equity_coefficient": intercept + slope,
        "comparison_role": "descriptive_three_point_extrapolation",
    }])
    attenuation = pd.concat([points, extrapolated], ignore_index=True)
    attenuation["slope"] = slope
    attenuation["intercept"] = intercept
    attenuation["formal_errors_in_variables_correction"] = 0
    attenuation.to_csv(PROCESSED / "h2_tracking_coefficient_attenuation_diagnostic.csv", index=False)
    export_dta(attenuation, PROCESSED / "h2_tracking_coefficient_attenuation_diagnostic.dta")
    weekly_audit = pd.DataFrame([{
        "firms": parameters["ticker"].nunique(),
        "dcc_success_count": int(parameters["dcc_success"].astype(bool).sum()),
        "gjr_success_count": int(parameters["gjr_all_success"].astype(bool).sum()),
        "weekly_obs_min": int(parameters["weekly_observations"].min()),
        "weekly_obs_median": float(parameters["weekly_observations"].median()),
        "weekly_obs_max": int(parameters["weekly_observations"].max()),
        "median_dcc_alpha": float(parameters["dcc_alpha_median"].iloc[0]),
        "median_dcc_beta": float(parameters["dcc_beta_median"].iloc[0]),
        "median_dcc_persistence": float(parameters["dcc_persistence_median"].iloc[0]),
    }])
    weekly_audit.to_csv(PROCESSED / "h2_weekly_estimation_audit.csv", index=False)
    export_dta(weekly_audit, PROCESSED / "h2_weekly_estimation_audit.dta")
    build_rtf(models)
    build_figure(models, attenuation)

    f = models.set_index("model_id")
    audit = {
        "status": "PASS",
        "daily_top5_equity_coefficient": float(f.loc["F1", "coefficient_exposure"]),
        "daily_us_coal_equity_coefficient": float(f.loc["F3", "coefficient_exposure"]),
        "weekly_top5_equity_coefficient": float(f.loc["F5", "coefficient_exposure"]),
        "daily_top5_asset_coefficient": float(f.loc["F2", "coefficient_exposure"]),
        "daily_us_coal_asset_coefficient": float(f.loc["F4", "coefficient_exposure"]),
        "weekly_top5_asset_coefficient": float(f.loc["F6", "coefficient_exposure"]),
        "naive_correlation_one_extrapolation": float(intercept + slope),
        "weekly_dcc_success_count": int(weekly_audit.loc[0, "dcc_success_count"]),
        "weekly_gjr_success_count": int(weekly_audit.loc[0, "gjr_success_count"]),
        "interpretation": (
            "The same-basket daily-to-weekly comparison is the primary frequency diagnostic. Weekly estimates are positive and larger, "
            "but remain statistically imprecise. The U.S. coal point changes economic content as well as synchronization. "
            "H2 is not statistically supported and its sign is unresolved."
        ),
        "same_window_warning": (
            "The often-quoted 0.740 U.S.-coal correlation uses the full 2010-2020 overlap, whereas 0.454 for the top-five basket uses 2019-2020. "
            "The audit therefore reports both continuations over both identical windows."
        ),
    }
    (AUDIT / "h2_factor_continuation_frequency_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
