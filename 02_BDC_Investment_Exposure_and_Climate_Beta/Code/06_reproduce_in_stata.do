version 17.0
clear all
set more off

* Run this do-file from the 02_BDC_Investment_Exposure_and_Climate_Beta directory.
local root = subinstr("`c(pwd)'", char(92), "/", .)
capture confirm file "`root'/Data/Processed/bdc19_dynamic_portfolio_h2_panel_2021_2025.dta"
if _rc {
    display as error "Set the working directory to 02_BDC_Investment_Exposure_and_Climate_Beta and rerun."
    exit 601
}

capture which esttab
if _rc ssc install estout, replace
capture which reghdfe
if _rc {
    ssc install ftools, replace
    ssc install reghdfe, replace
}

use "`root'/Data/Processed/bdc19_dynamic_portfolio_h2_panel_2021_2025.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq

local equity z_beta_climate_equity_report_mon
local asset  z_beta_climate_asset_report_mont
local brown  z_brown_share_broad_dynamic_pct
local controls z_log_assets z_debt_to_assets z_roa_quarter z_book_to_market z_beta_market_report_month

reghdfe `equity' `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store h2_1
reghdfe `asset' `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store h2_2
reghdfe `equity' `brown' `controls', absorb(quarter_id) vce(cluster bdc_id)
estimates store h2_3
reghdfe `asset' `brown' `controls', absorb(quarter_id) vce(cluster bdc_id)
estimates store h2_4
reghdfe `equity' `brown', absorb(bdc_id quarter_id) vce(cluster bdc_id)
estimates store h2_5
reghdfe `asset' `brown', absorb(bdc_id quarter_id) vce(cluster bdc_id)
estimates store h2_6

esttab h2_1 h2_2 h2_3 h2_4 h2_5 h2_6 using ///
    "`root'/Results/Table_3_Stata_Reproduction.rtf", replace rtf ///
    keep(`brown') b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) label compress ///
    mtitles("Equity beta" "Asset beta" "Equity beta" "Asset beta" "Equity beta" "Asset beta") ///
    stats(N r2, fmt(0 3) labels("Observations" "R-squared")) ///
    title("BDC Investment Exposure and Climate Beta") ///
    addnotes("Standard errors are clustered by BDC and appear in parentheses." ///
             "*, **, and *** denote two-sided significance at 10%, 5%, and 1%.")

display as text "Stata table written to Results/Table_3_Stata_Reproduction.rtf"

* High-yield excess-return robustness, using the full 2010--2025 DCC sample.
use "`root'/Data/Processed/h2_credit_return_robustness_panel.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq

local brown z_brown_share_broad_dynamic_pct
local controls z_log_assets z_debt_to_assets z_roa_quarter z_book_to_market z_beta_market_report_month

reghdfe z_beta_climate_full2f `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store credit_1
reghdfe z_beta_climate_credit3f `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store credit_2
reghdfe z_beta_asset_full2f `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store credit_3
reghdfe z_beta_asset_credit3f `brown', absorb(quarter_id) vce(cluster bdc_id)
estimates store credit_4
reghdfe z_beta_climate_credit3f `brown' `controls', absorb(quarter_id) vce(cluster bdc_id)
estimates store credit_5
reghdfe z_beta_climate_credit3f `brown', absorb(bdc_id quarter_id) vce(cluster bdc_id)
estimates store credit_6

esttab credit_1 credit_2 credit_3 credit_4 credit_5 credit_6 using ///
    "`root'/Results/Table_S1_Stata_Reproduction.rtf", replace rtf ///
    keep(`brown') b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) label compress ///
    mtitles("Equity 2F" "Equity + credit" "Asset 2F" "Asset + credit" "Equity + controls" "Equity + BDC FE") ///
    stats(N r2, fmt(0 3) labels("Observations" "R-squared")) ///
    title("High-Yield Credit-Factor Robustness of BDC Climate Beta") ///
    addnotes("Both DCC specifications are estimated over the identical 2010--2025 daily sample." ///
             "Standard errors are clustered by BDC; inference is two-sided.")

display as text "Stata table written to Results/Table_S1_Stata_Reproduction.rtf"

* Factor-continuation and frequency sensitivity.
use "`root'/Data/Processed/bdc19_dynamic_portfolio_h2_panel_2021_2025.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq
reghdfe z_beta_climate_equity_report_mon z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_1
reghdfe z_beta_climate_asset_report_mont z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_2

use "`root'/Data/Processed/bdc19_us_coal_daily_h2_panel_2021_2025.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq
reghdfe z_beta_climate_equity_report_mon z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_3
reghdfe z_beta_climate_asset_report_mont z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_4

use "`root'/Data/Processed/bdc19_top5_weekly_h2_panel_2021_2025.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq
reghdfe z_beta_climate_equity_weekly_qme z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_5
reghdfe z_beta_climate_asset_weekly_qmea z_brown_share_broad_dynamic_pct, absorb(quarter_id) vce(cluster bdc_id)
estimates store factor_6

esttab factor_1 factor_2 factor_3 factor_4 factor_5 factor_6 using ///
    "`root'/Results/Table_4_Stata_Reproduction.rtf", replace rtf ///
    keep(z_brown_share_broad_dynamic_pct) b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) label compress ///
    mtitles("Top-five daily E" "Top-five daily A" "U.S. coal daily E" "U.S. coal daily A" "Top-five weekly E" "Top-five weekly A") ///
    stats(N r2, fmt(0 3) labels("Observations" "R-squared")) ///
    title("Factor Continuation and Frequency Sensitivity") ///
    addnotes("All specifications include quarter fixed effects and cluster standard errors by BDC." ///
             "The weekly beta is averaged within each calendar quarter.")

display as text "Stata table written to Results/Table_4_Stata_Reproduction.rtf"

* Granular portfolio mechanism: FF49 industry betas estimated with the same DCC system.
use "`root'/Data/Processed/bdc19_ff49_dcc_portfolio_mechanism_panel_2021_2025.dta", clear
encode ticker, gen(bdc_id)
gen quarter_id = quarterly(fiscal_quarter, "YQ")
format quarter_id %tq

local equity z_beta_climate_equity_report_mon
local asset  z_beta_climate_asset_report_mont
local port   z_ff49_dcc_portfolio_climate_bet
local controls z_log_assets z_debt_to_assets z_roa_quarter z_book_to_market z_beta_market_report_month

reghdfe `equity' `port', absorb(quarter_id) vce(cluster bdc_id)
estimates store dcc49_1
reghdfe `equity' `port' `controls', absorb(quarter_id) vce(cluster bdc_id)
estimates store dcc49_2
reghdfe `equity' `port', absorb(bdc_id quarter_id) vce(cluster bdc_id)
estimates store dcc49_3
reghdfe `asset' `port', absorb(quarter_id) vce(cluster bdc_id)
estimates store dcc49_4
reghdfe `asset' `port' `controls', absorb(quarter_id) vce(cluster bdc_id)
estimates store dcc49_5
reghdfe `asset' `port', absorb(bdc_id quarter_id) vce(cluster bdc_id)
estimates store dcc49_6

esttab dcc49_1 dcc49_2 dcc49_3 dcc49_4 dcc49_5 dcc49_6 using ///
    "`root'/Results/Table_3_FF49_DCC_Stata_Reproduction.rtf", replace rtf ///
    keep(`port') b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) label compress ///
    mtitles("Equity" "Equity + controls" "Equity + BDC FE" "Asset" "Asset + controls" "Asset + BDC FE") ///
    stats(N r2, fmt(0 3) labels("Observations" "R-squared")) ///
    title("BDC Portfolio Climate Beta and Market Climate Beta") ///
    addnotes("FF49 industry and BDC climate betas use the same median scalar-DCC parameters." ///
             "Standard errors are clustered by BDC; *, **, and *** denote two-sided significance at 10%, 5%, and 1%.")

display as text "Stata table written to Results/Table_3_FF49_DCC_Stata_Reproduction.rtf"
