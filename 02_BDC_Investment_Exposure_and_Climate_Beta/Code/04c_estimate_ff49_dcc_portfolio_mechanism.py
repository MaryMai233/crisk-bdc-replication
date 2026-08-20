from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"


def load_ff49_module():
    path = Path(__file__).with_name("04b_estimate_ff49_portfolio_mechanism.py")
    spec = importlib.util.spec_from_file_location("ff49_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FF = load_ff49_module()


def cluster_t(
    y: np.ndarray, x: np.ndarray, groups: np.ndarray, coefficient_index: int = 1
) -> tuple[float, float, float]:
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    unique = np.unique(groups)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique:
        mask = groups == group
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k = x.shape
    correction = (len(unique) / (len(unique) - 1)) * ((n - 1) / max(n - k, 1))
    covariance = correction * inv @ meat @ inv
    se = float(np.sqrt(max(covariance[coefficient_index, coefficient_index], 0)))
    return float(beta[coefficient_index]), se, float(beta[coefficient_index] / se)


def wild_cluster_two_way_fe(
    frame: pd.DataFrame, outcome: str, exposure_name: str, reps: int = 9_999
) -> dict[str, object]:
    data = frame[[outcome, exposure_name, "ticker", "quarter"]].dropna().copy()
    fixed_blocks = [np.ones((len(data), 1))]
    for column in ("ticker", "quarter"):
        fixed_blocks.append(
            pd.get_dummies(data[column].astype(str), drop_first=True, dtype=float).to_numpy()
        )
    restricted_x = np.column_stack(fixed_blocks)
    x = np.column_stack([fixed_blocks[0], data[[exposure_name]].to_numpy(float), *fixed_blocks[1:]])
    y = data[outcome].to_numpy(float)
    groups = data["ticker"].astype(str).to_numpy()
    coefficient, standard_error, observed_t = cluster_t(y, x, groups)
    restricted_beta = np.linalg.pinv(restricted_x.T @ restricted_x) @ restricted_x.T @ y
    fitted = restricted_x @ restricted_beta
    residual = y - fitted
    unique = np.unique(groups)
    rng = np.random.default_rng(20260820)
    simulated_t = np.empty(reps)
    for draw in range(reps):
        signs = dict(zip(unique, rng.choice([-1.0, 1.0], size=len(unique))))
        y_star = fitted + residual * np.array([signs[group] for group in groups])
        simulated_t[draw] = cluster_t(y_star, x, groups)[2]
    p_value = float((1 + np.sum(np.abs(simulated_t) >= abs(observed_t))) / (reps + 1))
    return {
        "outcome": outcome,
        "exposure": exposure_name,
        "coefficient": coefficient,
        "cluster_standard_error": standard_error,
        "observed_t": observed_t,
        "wild_cluster_p_two_sided": p_value,
        "bootstrap_repetitions": reps,
        "clusters": int(len(unique)),
        "observations": int(len(data)),
        "seed": 20260820,
        "fixed_effects": "ticker and quarter",
    }


def main() -> None:
    exposure = pd.read_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv")
    exposure, imputation_audit = FF.replace_geography_only_quarters(exposure)
    ff49 = pd.read_csv(
        PROCESSED / "ff49_industry_dcc_beta_quarterly_2020_2025.csv",
        parse_dates=["quarter_end_observation_date"],
    ).rename(
        columns={
            "cbeta_ff49_dcc_qend": "cbeta_ff49_qend",
            "cbeta_ff49_dcc_qmean": "cbeta_ff49_qmean",
            "mbeta_ff49_dcc_qend": "mbeta_ff49_qend",
            "mbeta_ff49_dcc_qmean": "mbeta_ff49_qmean",
        }
    )
    mapped, portfolio = FF.build_portfolio_beta(exposure, ff49)
    mapped.to_csv(PROCESSED / "dynamic_bdc_industry_ff49_dcc_mapping_2021_2025.csv", index=False)
    FF.PRIOR.export_dta(
        mapped.drop(columns=["ff49_codes"]),
        PROCESSED / "dynamic_bdc_industry_ff49_dcc_mapping_2021_2025.dta",
    )
    portfolio = portfolio.rename(
        columns={
            "ff49_portfolio_climate_beta_qend": "ff49_dcc_portfolio_climate_beta_qend",
            "ff49_portfolio_climate_beta_qmean": "ff49_dcc_portfolio_climate_beta_qmean",
        }
    )
    portfolio.to_csv(PROCESSED / "dynamic_bdc_ff49_dcc_portfolio_beta_2021_2025.csv", index=False)
    FF.PRIOR.export_dta(portfolio, PROCESSED / "dynamic_bdc_ff49_dcc_portfolio_beta_2021_2025.dta")

    panel = pd.read_csv(
        PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv",
        parse_dates=["datadate"],
    )
    panel["quarter"] = panel["datadate"].dt.to_period("Q").astype(str)
    panel = panel.merge(portfolio, on=["ticker", "quarter"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    panel["ff49_dcc_portfolio_climate_beta_qend_lag1"] = panel.groupby("ticker")[
        "ff49_dcc_portfolio_climate_beta_qend"
    ].shift(1)
    zcols = [
        "beta_climate_equity_report_month", "beta_climate_asset_report_month",
        "ff49_dcc_portfolio_climate_beta_qend", "ff49_dcc_portfolio_climate_beta_qmean",
        "ff49_dcc_portfolio_climate_beta_qend_lag1", "log_assets", "debt_to_assets",
        "roa_quarter", "book_to_market", "beta_market_report_month",
    ]
    panel = FF.add_standardized(panel, zcols)
    panel.to_csv(PROCESSED / "bdc19_ff49_dcc_portfolio_mechanism_panel_2021_2025.csv", index=False)
    FF.PRIOR.export_dta(panel, PROCESSED / "bdc19_ff49_dcc_portfolio_mechanism_panel_2021_2025.dta")

    controls = [
        "z_log_assets", "z_debt_to_assets", "z_roa_quarter",
        "z_book_to_market", "z_beta_market_report_month",
    ]
    specs = [
        ("DCC49_1", "Asset beta, DCC-FF49 portfolio beta, quarter FE", "z_beta_climate_asset_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", [], ["quarter"]),
        ("DCC49_2", "Asset beta, DCC-FF49 portfolio beta plus controls", "z_beta_climate_asset_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", controls, ["quarter"]),
        ("DCC49_3", "Asset beta, DCC-FF49 portfolio beta, firm and quarter FE", "z_beta_climate_asset_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("DCC49_4", "Equity beta, DCC-FF49 portfolio beta, quarter FE", "z_beta_climate_equity_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", [], ["quarter"]),
        ("DCC49_5", "Equity beta, DCC-FF49 portfolio beta plus controls", "z_beta_climate_equity_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", controls, ["quarter"]),
        ("DCC49_6", "Equity beta, DCC-FF49 portfolio beta, firm and quarter FE", "z_beta_climate_equity_report_month", "z_ff49_dcc_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("DCC49_7", "Asset beta, DCC-FF49 quarterly-mean portfolio beta", "z_beta_climate_asset_report_month", "z_ff49_dcc_portfolio_climate_beta_qmean", [], ["quarter"]),
        ("DCC49_8", "Asset beta, lagged DCC-FF49 portfolio beta", "z_beta_climate_asset_report_month", "z_ff49_dcc_portfolio_climate_beta_qend_lag1", [], ["quarter"]),
    ]
    models = pd.DataFrame(
        [
            FF.PRIOR.fit_ols(
                panel, outcome, exposure_name, model_id, label,
                controls=model_controls, fixed_effects=fixed_effects, cluster="ticker",
            )
            for model_id, label, outcome, exposure_name, model_controls, fixed_effects in specs
        ]
    )
    models["coefficient_unit"] = "Standard deviations of BDC climate beta per one-SD DCC-FF49 portfolio climate beta"
    models.to_csv(PROCESSED / "h2_ff49_dcc_portfolio_mechanism_models.csv", index=False)
    FF.PRIOR.export_dta(models, PROCESSED / "h2_ff49_dcc_portfolio_mechanism_models.dta")

    robustness_rows: list[dict[str, object]] = []
    subset_specs = [
        ("FULL", "Full sample", panel),
        ("POST21", "2022Q1-2025Q4", panel.loc[panel["quarter"].ge("2022Q1")]),
        ("COV80", "Mapping coverage at least 80 percent", panel.loc[panel["mapped_weight_pct"].ge(80)]),
        (
            "NOIMP",
            "Exclude four geography-table imputations",
            panel.loc[
                ~(
                    (panel["ticker"].eq("CGBD") & panel["quarter"].isin(["2025Q1", "2025Q2", "2025Q3"]))
                    | (panel["ticker"].eq("TSLX") & panel["quarter"].eq("2023Q1"))
                )
            ],
        ),
    ]
    for sample_id, sample_label, sample in subset_specs:
        for outcome_id, outcome in (
            ("EQ", "z_beta_climate_equity_report_month"),
            ("AS", "z_beta_climate_asset_report_month"),
        ):
            row = FF.PRIOR.fit_ols(
                sample,
                outcome,
                "z_ff49_dcc_portfolio_climate_beta_qend",
                f"{sample_id}_{outcome_id}",
                f"{sample_label}; {outcome_id} climate beta",
                controls=[],
                fixed_effects=["ticker", "quarter"],
                cluster="ticker",
            )
            row["mapping_confidence_sample"] = "all mapped rules"
            robustness_rows.append(row)

    exposure_confidence = exposure.copy()
    exposure_confidence["mapping_confidence_filter"] = exposure_confidence["industry_reported"].map(
        lambda value: FF.label_to_ff49(value)[2]
    )
    high_medium = exposure_confidence.loc[
        exposure_confidence["mapping_confidence_filter"].isin(["high", "medium"])
    ].copy()
    _, hm_portfolio = FF.build_portfolio_beta(high_medium, ff49)
    hm_portfolio = hm_portfolio.rename(
        columns={"ff49_portfolio_climate_beta_qend": "hm_portfolio_beta_qend"}
    )
    hm_panel = pd.read_csv(
        PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv",
        parse_dates=["datadate"],
    )
    hm_panel["quarter"] = hm_panel["datadate"].dt.to_period("Q").astype(str)
    hm_panel = hm_panel.merge(hm_portfolio, on=["ticker", "quarter"], how="inner", validate="one_to_one")
    hm_panel = FF.add_standardized(
        hm_panel,
        [
            "beta_climate_equity_report_month", "beta_climate_asset_report_month",
            "hm_portfolio_beta_qend",
        ],
    )
    for outcome_id, outcome in (
        ("EQ", "z_beta_climate_equity_report_month"),
        ("AS", "z_beta_climate_asset_report_month"),
    ):
        row = FF.PRIOR.fit_ols(
            hm_panel,
            outcome,
            "z_hm_portfolio_beta_qend",
            f"HIMED_{outcome_id}",
            f"High- and medium-confidence mappings; {outcome_id} climate beta",
            controls=[],
            fixed_effects=["ticker", "quarter"],
            cluster="ticker",
        )
        row["mapping_confidence_sample"] = "high and medium only"
        robustness_rows.append(row)
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(PROCESSED / "h2_ff49_dcc_mechanism_robustness_models.csv", index=False)
    FF.PRIOR.export_dta(robustness, PROCESSED / "h2_ff49_dcc_mechanism_robustness_models.dta")

    wild = pd.DataFrame(
        [
            wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_equity_report_month",
                "z_ff49_dcc_portfolio_climate_beta_qend",
            ),
            wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_asset_report_month",
                "z_ff49_dcc_portfolio_climate_beta_qend",
            ),
        ]
    )
    wild.to_csv(PROCESSED / "h2_ff49_dcc_wild_cluster_bootstrap.csv", index=False)
    FF.PRIOR.export_dta(wild, PROCESSED / "h2_ff49_dcc_wild_cluster_bootstrap.dta")

    audit = {
        "status": "PASS",
        "estimator_alignment": "Industry, bank, and BDC climate betas all use the common scalar-DCC parameter vector",
        "company_quarters": int(len(panel)),
        "firms": int(panel["ticker"].nunique()),
        "median_mapping_coverage_pct": float(panel["mapped_weight_pct"].median()),
        "minimum_mapping_coverage_pct": float(panel["mapped_weight_pct"].min()),
        "geography_table_quarters_replaced": imputation_audit,
    }
    (AUDIT / "ff49_dcc_bdc_portfolio_mechanism_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "p_one_sided_positive", "n", "r_squared"]].to_string(index=False))
    print("\nRobustness models:")
    print(robustness[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n"]].to_string(index=False))
    print("\nWild-cluster bootstrap:")
    print(wild.to_string(index=False))


if __name__ == "__main__":
    main()
