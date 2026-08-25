from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
P1 = ROOT / "01_Bank_CRISK_Replication" / "Data" / "Processed"
P2 = ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Data" / "Processed"


def main() -> None:
    top4 = pd.read_csv(P1 / "top4_end2020_detail.csv")
    assert top4["mean_identity_error_usd_bn"].abs().max() < 1e-10

    paired = pd.read_csv(P1 / "bdc_beta_2019_2020_paired_test.csv").iloc[0]
    assert paired["observations"] == 20
    assert paired["positive_changes"] == 19
    assert paired["h2_subsample_observations"] == 19
    assert paired["h2_subsample_positive_changes"] == 18

    hac = pd.read_csv(P1 / "h1_hac_lag_sensitivity.csv")
    assert set(hac["max_lag"]) == {21, 63, 126, 203}
    beta_hac = hac[hac["outcome"].eq("Daily cross-bank mean climate beta")].set_index("max_lag")
    assert beta_hac.loc[203, "std_error"] > beta_hac.loc[21, "std_error"]

    panel = pd.read_csv(P2 / "bdc19_descriptive_panel_2021_2025.csv")
    reciprocal_error = np.abs(panel["book_to_market"] * panel["price_to_nav"] - 1.0).max()
    assert reciprocal_error < 1e-12

    primary = pd.read_csv(P2 / "h2_brown_share_dynamic_models.csv").set_index("model_id")
    credit = pd.read_csv(P2 / "h2_credit_return_robustness_models.csv").set_index("model_id")
    assert abs(primary.loc["H2_1", "coefficient_exposure"] - credit.loc["C1", "coefficient_exposure"]) < 1e-12
    assert abs(primary.loc["H2_2", "coefficient_exposure"] - credit.loc["C3", "coefficient_exposure"]) < 1e-12

    attenuation = pd.read_csv(P2 / "h2_tracking_coefficient_attenuation_diagnostic.csv")
    extrapolated = attenuation.loc[
        attenuation["point_type"].eq("Naive extrapolation"), "h2_equity_coefficient"
    ].iloc[0]
    pooled_mde = pd.read_csv(P2 / "h2_power_diagnostics.csv").set_index("model_id").loc[
        "H2_1", "minimum_detectable_effect_80pct_power"
    ]
    assert 0.20 < extrapolated < 0.23
    assert extrapolated < pooled_mde

    weekly = pd.read_csv(P2 / "h2_weekly_estimation_audit.csv").iloc[0]
    assert weekly["dcc_success_count"] == 19
    assert weekly["gjr_success_count"] == 19

    top75_tracking = pd.read_csv(
        ROOT / "01_Bank_CRISK_Replication" / "Data" / "Processed" / "kol_top75_tracking_diagnostics.csv"
    ).set_index("frequency")
    assert top75_tracking.loc["Daily", "correlation_with_kol"] > 0.78
    assert top75_tracking.loc["Weekly", "correlation_with_kol"] > 0.94
    top75_models = pd.read_csv(P2 / "h2_ff49_dcc_kol_top75_models.csv").set_index("model_id")
    top75_weekly_models = pd.read_csv(
        P2 / "h2_ff49_dcc_kol_top75_weekly_models.csv"
    ).set_index("model_id")
    assert 0.17 < top75_models.loc["T75_6", "coefficient_exposure"] < 0.20
    assert top75_models.loc["T75_6", "p_two_sided"] > 0.10
    assert 0.16 < top75_weekly_models.loc["T75W_6", "coefficient_exposure"] < 0.19

    audit_dir = P2 / "Audit"
    ff49_build = json.loads((audit_dir / "ff49_industry_climate_beta_audit.json").read_text())
    assert ff49_build["ff49_groups"] == 49
    assert min(row["classification_rate_after_common_stock_filter"] for row in ff49_build["annual_files"]) > 0.998

    ff49_dcc = json.loads((audit_dir / "ff49_industry_dcc_beta_audit.json").read_text())
    assert ff49_dcc["ff49_industries"] == 49
    assert ff49_dcc["industry_gjr_success_count"] == 49

    mechanism_audit = json.loads((audit_dir / "ff49_dcc_bdc_portfolio_mechanism_audit.json").read_text())
    assert mechanism_audit["company_quarters"] == 380
    assert mechanism_audit["median_mapping_coverage_pct"] > 97.0
    mapped_rows = pd.read_csv(P2 / "dynamic_bdc_industry_ff49_dcc_mapping_2021_2025.csv")
    mapped_overall = 100 * mapped_rows["mapped_weight"].sum() / mapped_rows["portfolio_fair_value_pct"].sum()
    assert mapped_overall > 95.0

    dcc_models = pd.read_csv(P2 / "h2_ff49_dcc_portfolio_mechanism_models.csv").set_index("model_id")
    dcc_robust = pd.read_csv(P2 / "h2_ff49_dcc_mechanism_robustness_models.csv").set_index("model_id")
    assert abs(dcc_models.loc["DCC49_6", "coefficient_exposure"] - 0.151745) < 1e-6
    assert abs(dcc_models.loc["DCC49_6", "coefficient_exposure"] - dcc_robust.loc["FULL_EQ", "coefficient_exposure"]) < 1e-12
    assert dcc_models.loc["DCC49_6", "p_two_sided"] < 0.10

    wild_dcc = pd.read_csv(P2 / "h2_ff49_dcc_wild_cluster_bootstrap.csv")
    wild_equity = wild_dcc[wild_dcc["outcome"].eq("z_beta_climate_equity_report_month")].iloc[0]
    assert wild_equity["bootstrap_repetitions"] == 9999
    assert wild_equity["wild_cluster_p_two_sided"] > 0.10

    assert (ROOT / "Paper" / "Climate_Risk_and_BDCs.pdf").exists()
    assert (ROOT / "Paper" / "Climate_Risk_and_BDCs_Word.docx").exists()
    required_results = [
        ROOT / "01_Bank_CRISK_Replication" / "Results" / "Figure03_Bank_Level_Beta_Changes.png",
        ROOT / "01_Bank_CRISK_Replication" / "Results" / "Figure04_Published_vs_Replicated.png",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_3_BDC_Investment_Exposure.rtf",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_3_BDC_FF49_DCC_Portfolio_Mechanism.rtf",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_4_Factor_Continuation_and_H2_Robustness.rtf",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Figure_5_Factor_Continuation_Sensitivity.png",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Figure_3_FF49_DCC_Portfolio_Mechanism.png",
        ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Figure_5_KOL_Basket_Breadth_and_BDC_Mechanism.png",
        ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Table_3_BDC_Asset_Coverage_Stress.rtf",
        ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Figure_6_Asset_Coverage_Before_After.png",
        ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Figure_7_Threshold_Proximity.png",
    ]
    assert all(path.exists() for path in required_results)
    assert not list(ROOT.glob("**/Results/*.xlsx"))
    print("PASS: release numerical and file assertions")


if __name__ == "__main__":
    main()
