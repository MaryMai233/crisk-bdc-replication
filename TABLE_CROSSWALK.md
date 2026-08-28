# Table crosswalk to Jung, Engle, and Berner (2025)

Version 4.4 keeps the paper compact while retaining the displays that are central to the replication and BDC extension.

**Climate-factor scope:** the original paper constructs four climate transition factors (Stranded Asset, Emission, Brown Minus Green, and CEP). This archive currently implements the **Stranded Asset factor** only. The top-five and cumulative-75% KOL baskets are continuation rules for that same factor.

| Original paper | v4.4 paper | Relationship |
|---|---|---|
| Main CRISK application | Table 1: Bank Replication Benchmarks | Compact replication benchmark. |
| Figure 3 / climate-beta time series | Figure 1: Annual Climate Beta for Banks and BDCs | Replication plus BDC comparison. |
| Table 1: Bank Climate Beta and Loan Portfolio Climate Beta | Table 2: BDC Climate Beta and Portfolio Climate Beta | Direct four-column specification analogue. |
| Section 5 / exposure validation | Figure 2: Measurement Resolution and the BDC Portfolio Mechanism | Extension-specific diagnostic on exposure granularity. |
| Section 7 / alternative factor implementation | Table 3 and Figure 3: KOL continuation breadth | Extension-specific factor-maintenance robustness. |
| CRISK capital interpretation | Table 4: BDC Asset-Coverage Stress | BDC-specific statutory-capacity extension. |

The paper does not include separate appendix tables. More detailed robustness results remain available in the replication modules.
