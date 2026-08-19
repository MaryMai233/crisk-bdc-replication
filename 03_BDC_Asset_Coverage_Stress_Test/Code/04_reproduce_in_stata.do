version 17.0
clear all
set more off

* Run this do-file from the 03_BDC_Asset_Coverage_Stress_Test directory.
local root = subinstr("`c(pwd)'", char(92), "/", .)
capture confirm file "`root'/Data/Processed/h3_analysis_panel.dta"
if _rc {
    display as error "Set the working directory to 03_BDC_Asset_Coverage_Stress_Test and rerun."
    exit 601
}

capture which esttab
if _rc ssc install estout, replace

use "`root'/Data/Processed/h3_analysis_panel.dta", clear
keep if coverage_data_status == "SEC_REPORTED"
encode ticker, gen(bdc_id)
reg buffer_shrink_pp, vce(cluster bdc_id)
estimates store h3_primary

esttab h3_primary using ///
    "`root'/Results/Table_3_Stata_Reproduction.rtf", replace rtf ///
    keep(_cons) b(3) se(3) nostar label compress ///
    coeflabels(_cons "Coverage-buffer compression (pp)") ///
    stats(N, fmt(0) labels("Observations")) ///
    title("Climate Stress and BDC Asset-Coverage Capacity") ///
    addnotes("Standard errors are clustered by BDC and appear in parentheses for reproducibility." ///
             "The manuscript treats this as a calibration because the positive sign is partly mechanical and first-stage DCC uncertainty is not propagated.")

display as text "Stata table written to Results/Table_3_Stata_Reproduction.rtf"
