from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"


def load_helper():
    path = Path(__file__).with_name("01_industry_beta_helpers.py")
    spec = importlib.util.spec_from_file_location("h2_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def standardize(frame: pd.DataFrame, column: str) -> None:
    sd = frame[column].std(ddof=0)
    frame[f"z_{column}"] = (frame[column] - frame[column].mean()) / sd


def design(frame: pd.DataFrame, outcome: str, exposure: str, include_exposure: bool):
    needed = [outcome, exposure, "quarter", "ticker"]
    data = frame[needed].dropna().copy()
    blocks = [np.ones((len(data), 1))]
    if include_exposure:
        blocks.append(data[[exposure]].to_numpy(float))
    dummies = pd.get_dummies(data["quarter"].astype(str), drop_first=True, dtype=float)
    blocks.append(dummies.to_numpy(float))
    return data, np.column_stack(blocks), data[outcome].to_numpy(float)


def cluster_t(y: np.ndarray, x: np.ndarray, groups: np.ndarray, coefficient_index: int = 1):
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


def wild_cluster_p(frame: pd.DataFrame, outcome: str, exposure: str, reps: int = 4999):
    data, x, y = design(frame, outcome, exposure, True)
    groups = data["ticker"].astype(str).to_numpy()
    coef, se, observed_t = cluster_t(y, x, groups)
    _, restricted_x, restricted_y = design(frame, outcome, exposure, False)
    restricted_beta = np.linalg.pinv(restricted_x.T @ restricted_x) @ restricted_x.T @ restricted_y
    fitted = restricted_x @ restricted_beta
    residual = restricted_y - fitted
    unique = np.unique(groups)
    rng = np.random.default_rng(20250818)
    simulated = np.empty(reps)
    for draw in range(reps):
        signs = dict(zip(unique, rng.choice([-1.0, 1.0], size=len(unique))))
        y_star = fitted + residual * np.array([signs[group] for group in groups])
        simulated[draw] = cluster_t(y_star, x, groups)[2]
    p_two = float((1 + np.sum(np.abs(simulated) >= abs(observed_t))) / (reps + 1))
    return {
        "outcome": outcome,
        "coefficient": coef,
        "cluster_standard_error": se,
        "observed_t": observed_t,
        "wild_cluster_p_two_sided": p_two,
        "bootstrap_repetitions": reps,
        "clusters": len(unique),
        "observations": len(data),
        "seed": 20250818,
    }


def esc(value: object) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def rtf_row(cells, widths, bold=False, top=False, bottom=False):
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


def main() -> None:
    panel = pd.read_csv(
        PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv",
        parse_dates=["datadate"],
    )
    rows = pd.read_csv(PROCESSED / "dynamic_bdc_industry_ff12_mapping_2021_2025.csv")
    rows["portfolio_fair_value_pct"] = pd.to_numeric(rows["portfolio_fair_value_pct"], errors="coerce")
    rows["low_weight"] = np.where(rows["mapping_confidence"].eq("low"), rows["portfolio_fair_value_pct"], 0.0)
    rows["low_brown_weight"] = np.where(
        rows["mapping_confidence"].eq("low"),
        rows["portfolio_fair_value_pct"] * rows["brown_broad"],
        0.0,
    )
    bounds = (
        rows.groupby(["ticker", "quarter"], as_index=False)
        .agg(low_weight=("low_weight", "sum"), low_brown_weight=("low_brown_weight", "sum"))
    )
    panel = panel.merge(bounds, on=["ticker", "quarter"], how="left", validate="one_to_one")
    residual = (100.0 - panel["mapped_weight_pct"]).clip(lower=0)
    panel["brown_share_lower_pct"] = (
        panel["brown_share_broad_dynamic_pct"] - panel["low_brown_weight"]
    ).clip(lower=0)
    panel["brown_share_upper_pct"] = (
        panel["brown_share_broad_dynamic_pct"]
        + (panel["low_weight"] - panel["low_brown_weight"]).clip(lower=0)
        + residual
    ).clip(upper=100)
    panel["brown_share_broad_lag1_pct"] = panel.groupby("ticker")[
        "brown_share_broad_dynamic_pct"
    ].shift(1)
    for column in [
        "brown_share_lower_pct",
        "brown_share_upper_pct",
        "brown_share_broad_lag1_pct",
    ]:
        standardize(panel, column)

    y_eq = "z_beta_climate_equity_report_month"
    y_as = "z_beta_climate_asset_report_month"
    specifications = [
        ("R1", "Lower classification bound, equity beta", y_eq, "z_brown_share_lower_pct", panel),
        ("R2", "Upper classification bound, equity beta", y_eq, "z_brown_share_upper_pct", panel),
        ("R3", "Lower classification bound, asset beta", y_as, "z_brown_share_lower_pct", panel),
        ("R4", "Upper classification bound, asset beta", y_as, "z_brown_share_upper_pct", panel),
        ("R5", "Lagged broad exposure, equity beta", y_eq, "z_brown_share_broad_lag1_pct", panel),
        ("R6", "Lagged broad exposure, asset beta", y_as, "z_brown_share_broad_lag1_pct", panel),
        ("R7", "Post-2021 broad exposure, equity beta", y_eq, "z_brown_share_broad_dynamic_pct", panel[panel["datadate"].dt.year >= 2022]),
        ("R8", "Post-2021 broad exposure, asset beta", y_as, "z_brown_share_broad_dynamic_pct", panel[panel["datadate"].dt.year >= 2022]),
    ]
    models = pd.DataFrame(
        [
            HELPER.fit_ols(
                sample,
                outcome,
                exposure,
                model_id,
                label,
                fixed_effects=["quarter"],
                cluster="ticker",
            )
            for model_id, label, outcome, exposure, sample in specifications
        ]
    )
    models.to_csv(PROCESSED / "h2_inference_robustness_models.csv", index=False)
    HELPER.export_dta(models, PROCESSED / "h2_inference_robustness_models.dta")

    wild = pd.DataFrame(
        [
            wild_cluster_p(panel, y_eq, "z_brown_share_broad_dynamic_pct"),
            wild_cluster_p(panel, y_as, "z_brown_share_broad_dynamic_pct"),
        ]
    )
    wild.to_csv(PROCESSED / "h2_wild_cluster_bootstrap.csv", index=False)
    HELPER.export_dta(wild, PROCESSED / "h2_wild_cluster_bootstrap.dta")

    main_models = pd.read_csv(PROCESSED / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    power_rows = []
    for model_id in ["H2_1", "H2_2", "H2_5", "H2_6"]:
        item = main_models.loc[model_id]
        tcrit = stats.t.ppf(0.975, df=item["df_inference"])
        power_rows.append(
            {
                "model_id": model_id,
                "observed_standardized_effect": item["coefficient_exposure"],
                "cluster_standard_error": item["standard_error"],
                "minimum_detectable_effect_80pct_power": (tcrit + stats.norm.ppf(0.80)) * item["standard_error"],
                "minimum_detectable_effect_90pct_power": (tcrit + stats.norm.ppf(0.90)) * item["standard_error"],
                "clusters": int(item["df_inference"] + 1),
                "fixed_effects": item["fixed_effects"],
            }
        )
    power = pd.DataFrame(power_rows)
    power.to_csv(PROCESSED / "h2_power_diagnostics.csv", index=False)
    HELPER.export_dta(power, PROCESSED / "h2_power_diagnostics.dta")

    membership = (
        panel.assign(year=panel["datadate"].dt.year)
        .groupby(["year", "ticker"], as_index=False)
        .agg(quarters=("quarter", "nunique"))
    )
    membership.to_csv(PROCESSED / "h2_sample_membership_by_year.csv", index=False)
    HELPER.export_dta(membership, PROCESSED / "h2_sample_membership_by_year.dta")
    panel.to_csv(PROCESSED / "h2_classification_bounds_panel.csv", index=False)
    HELPER.export_dta(panel, PROCESSED / "h2_classification_bounds_panel.dta")

    def starred(value: float, p_value: float) -> str:
        mark = "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""
        return f"{value:.3f}{mark}"

    selected = models.set_index("model_id").loc[["R1", "R2", "R5", "R7", "R3", "R4", "R6", "R8"]]
    rows_out = [
        ["", "Lower", "Upper", "Lagged", "Post-2021", "Lower", "Upper", "Lagged", "Post-2021"],
        ["Brown investment share", *[
            starred(value, p_value)
            for value, p_value in zip(selected["coefficient_exposure"], selected["p_two_sided"])
        ]],
        ["", *[f"({x:.3f})" for x in selected["standard_error"]]],
        ["Outcome", "Equity", "Equity", "Equity", "Equity", "Asset", "Asset", "Asset", "Asset"],
        ["Observations", *[str(int(x)) for x in selected["n"]]],
    ]
    note = (
        "Lower and upper bounds treat every low-confidence or unmapped portfolio weight as, respectively, non-brown or brown. "
        "Lagged specifications use the prior quarter's exposure. Post-2021 specifications omit 2021. All models include quarter fixed effects and cluster standard errors by BDC. "
        "***, **, and * denote two-sided significance at 1%, 5%, and 10%."
    )
    body = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\fs20\landscape\paperw15840\paperh12240\margl600\margr600\margt800\margb800", r"\pard\sa40\b\fs24 Table 2A: Classification, Timing, and COVID-Period Robustness.\b0\par"]
    for index, cells in enumerate(rows_out):
        body.append(rtf_row(cells, [3000] + [1450] * 8, bold=index == 0, top=index == 0, bottom=index == len(rows_out) - 1))
    body += [r"\pard\sa100\fs17 " + esc("Notes: " + note) + r"\par", "}"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    body[1] = r"\pard\sa40\b\fs24 Table S2: Classification, Timing, and COVID-Period Robustness.\b0\par"
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n"]].to_string(index=False))
    print("\nWild-cluster bootstrap:\n", wild.to_string(index=False))
    print("\nPower diagnostics:\n", power.to_string(index=False))


if __name__ == "__main__":
    main()
