version 17.0
clear all
set more off

* Run this do-file from the 01_Bank_CRISK_Replication directory.
local root = subinstr("`c(pwd)'", char(92), "/", .)
capture confirm file "`root'/Data/Processed/bank_level_2019_2020_changes.dta"
if _rc {
    display as error "Set the working directory to 01_Bank_CRISK_Replication and rerun."
    exit 601
}

capture which esttab
if _rc ssc install estout, replace

use "`root'/Data/Processed/bank_level_2019_2020_changes.dta", clear
reg beta_mean_change, vce(robust)
estimates store annual_beta

use "`root'/Data/Processed/bank_h1_daily_2010_2025.dta", clear
keep if inrange(year, 2019, 2020)
collapse (mean) beta_climate, by(date)
gen post2020 = year(date) == 2020
tsset date
newey beta_climate post2020, lag(203)
estimates store daily_beta

esttab annual_beta daily_beta using ///
    "`root'/Results/Table_1_Stata_Reproduction.rtf", replace rtf ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) label compress ///
    mtitles("Annual beta" "Daily beta") ///
    stats(N, fmt(0) labels("Observations")) ///
    title("Bank Replication of the 2020 Climate-Risk Shock") ///
    addnotes("Annual beta is a bank-level paired-change regression." ///
             "The daily specification uses Newey-West standard errors with 203 lags." ///
             "*, **, and *** denote two-sided significance at 10%, 5%, and 1%.")

display as text "Stata table written to Results/Table_1_Stata_Reproduction.rtf"
