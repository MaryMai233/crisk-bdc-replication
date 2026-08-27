# v4.2 Inline Table/Figure Layout

Version 4.2 no longer collects tables and figures in an end-of-paper display section. The paper source in the packaged release places each display next to the narrative that motivates or interprets it.

Formatting rule:
- Table/Figure number and title appear above the display.
- Notes appear below the display.
- Main displays are embedded in the relevant section near first substantive discussion.
- In Word, a long table may begin on a fresh page to avoid splitting, but it remains inside the corresponding section.

Placement map:

| Display | Narrative location |
|---|---|
| Table 2: BDC Summary Statistics | Data and Method → Samples and portfolio exposure |
| Table 1: Bank Replication Benchmarks | Finding 1, immediately after the benchmark paragraph |
| Figure 1: Annual Climate Beta for Banks and BDCs | Finding 1, immediately after Table 1 |
| Figure 2: Top-Four Bank CRISK Around the 2020 Shock | Finding 1, after the event/timing discussion |
| Table 3: BDC Climate Beta and Portfolio Climate Beta | Finding 2, after the direct original-Table-1 analogue is introduced |
| Table 4: BDC Asset Climate Beta and Portfolio Climate Beta | Finding 2, after the de-levered asset-beta bridge |
| Table 5: BDC Exposure Measurement Resolution | Finding 2, after the bank-vs-BDC measurement discussion |
| Figure 3: Measurement Resolution and the BDC Portfolio Mechanism | Finding 2, immediately after Table 5 |
| Table 6: KOL Continuation Breadth and Inference | Finding 2, after factor-maintenance discussion |
| Figure 4: KOL Basket Breadth and the BDC Portfolio Mechanism | Finding 2, immediately after Table 6 |
| Table 7: Climate Stress and BDC Asset-Coverage Capacity | Finding 3, after the statutory-capacity paragraph |
| Figure 5: BDC Asset Coverage Before and After Climate Stress | Finding 3, immediately after Table 7 |
| Tables A1–A2 and Figures A1–A2 | Appendix, interleaved with the appendix explanations rather than dumped together |

The packaged v4.2 `Paper/Climate_Risk_and_BDCs.tex` and `Paper/build_word_report.py` implement this layout directly.
