# Table crosswalk to Jung, Engle, and Berner (2025)

This release deliberately mirrors the original paper's table logic rather than combining unrelated robustness checks into dense multi-panel tables.

**v4.2 formatting rule:** every table number/title is placed **above** the table; table notes remain below; and every main table/figure is embedded in the relevant narrative section near its first substantive discussion rather than collected at the end.

**Climate-factor scope:** the original paper constructs four climate transition factors---Stranded Asset, Emission, Brown Minus Green (BMG), and CEP. This archive currently implements **one climate factor, the Stranded Asset factor**, as the replication baseline. The top-five and cumulative-75% KOL baskets are two continuation rules for that same factor, not two different climate factors. FF12/FF49/brown-share variables are exposure mappings, and HYG/JNK-minus-SHY is a control-factor robustness, not an additional climate factor.

| Original paper | This package | Relationship |
|---|---|---|
| Table B.3: Bank-level Data Summary Statistics | Table 2: BDC Summary Statistics | Direct formatting analogue: Mean, St.Dev., 25th percentile, 75th percentile, Count. |
| Table 1: Bank Climate Beta and Loan Portfolio Climate Beta | Table 3: BDC Climate Beta and Portfolio Climate Beta | Direct specification analogue: (1) exposure only; (2) + controls; (3) + BDC FE; (4) + year FE. Coefficients are followed by t statistics in parentheses. |
| Table F.1: Unlevered Beta | Table 4: BDC Asset Climate Beta and Portfolio Climate Beta | Closest economic analogue. BDC asset beta de-levers the traded equity beta. |
| Table F.2: Utilized Exposure | No exact BDC counterpart | Public BDC filings do not provide a committed-versus-utilized credit split comparable to Y-14. We do not fabricate a proxy. |
| Table F.3: Firm-level Beta | No exact BDC counterpart | Most BDC portfolio companies are private, so a borrower-firm market beta is unavailable. FF49 is a sector mapping, not a firm-level substitute. |
| Table F.4: Period interactions | No exact BDC counterpart in the main table sequence | The BDC sample begins in 2021, so a 2012/COVID-period interaction test is not economically parallel. We do not relabel a different robustness as the same test. |
| Section 7 / Appendix N factor and estimation robustness | Table 5: KOL continuation breadth and inference | Extension-specific robustness: published top-five continuation versus a pre-specified cumulative-75% basket, daily versus weekly estimation, and small-cluster inference. |
| No direct original table | Table 6: BDC Exposure Measurement Resolution | Extension-specific diagnostic showing why coarse brown-share/FF12 exposure is weaker than continuous FF49 and estimator-aligned DCC-FF49 exposure. |
| Table 2: CRISK decomposition | Bank replication outputs only | Retained for the bank replication. We do not force the bank prudential-capital decomposition onto BDCs. |
| CRISK / mCRISK application | Table 7: BDC asset-coverage stress | New extension. BDCs are interpreted under statutory asset coverage rather than the bank k=8% capital rule. |

The published top-five KOL continuation remains the strict replication. The cumulative-75% continuation uses the official September 30, 2020 N-PORT schedule and selects holdings until coverage first exceeds 75%; this produces 15 securities covering 77.1% of reported common-stock market value. It is a pre-specified robustness check, not an outcome-selected replacement.
