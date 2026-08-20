from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "Climate_Risk_and_BDCs_Word.docx"


def set_font(run, size: float, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_border(cell, *, top: str | None = None, bottom: str | None = None) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")
    for edge, width in (("top", top), ("bottom", bottom)):
        if width:
            element = borders.find(qn(f"w:{edge}"))
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), width)
            element.set(qn("w:color"), "000000")


def add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_font(run, 9)


def replace_paragraph(paragraph, text: str) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    set_font(run, 10.5)


def remove_after(document: Document, heading: str) -> None:
    body = document._element.body
    marker = None
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == heading:
            marker = paragraph._p
            break
    if marker is None:
        raise RuntimeError(f"Heading not found: {heading}")
    siblings = list(body)
    start = siblings.index(marker)
    for element in siblings[start:]:
        if element.tag != qn("w:sectPr"):
            body.remove(element)


def clean_body(document: Document) -> None:
    remove_after(document, "Tables and Figures")
    for paragraph in list(document.paragraphs):
        if paragraph.text.strip() == "99":
            paragraph._element.getparent().remove(paragraph._element)

    replacements = {
        "Table [tab:h1]": "Table 1",
        "Table [tab:desc]": "Table 2",
        "Table [tab:h2main]": "Table 3",
        "Table [tab:factor]": "Table 4",
        "Table [tab:stress]": "Table 5",
        "Figure [fig:bankcross]": "Figure A1",
        "Figure [fig:benchmarkplot]": "Figure A1",
        "Figure [fig:ff49mechanism]": "Figure 3",
        "Figure [fig:factorsensitivity]": "Figure 4",
        "[tab:h1]": "1",
        "[tab:desc]": "2",
        "[tab:h2main]": "3",
        "[tab:factor]": "4",
        "[tab:stress]": "5",
        "[eq:crisk]": "(5)",
        "[eq:coverage]": "(6)",
    }
    for paragraph in document.paragraphs:
        text = paragraph.text
        for old, new in replacements.items():
            text = text.replace(old, new)
        if text != paragraph.text:
            replace_paragraph(paragraph, text)

    for paragraph in document.paragraphs:
        if paragraph.text.startswith("CRISK maps a traded stranded-asset factor"):
            replace_paragraph(
                paragraph,
                paragraph.text.replace("shortfall .", "shortfall (Jung, Engle, and Berner, 2025)."),
            )
        if paragraph.text.startswith("Following , the stranded-asset factor"):
            replace_paragraph(
                paragraph,
                "Following Jung, Engle, and Berner (2025), the stranded-asset factor is",
            )
        if paragraph.text.startswith("KOL supplies the coal return"):
            text = paragraph.text.replace("December 2024 .", "December 2024 (IEEFA, 2025).")
            replace_paragraph(paragraph, text)
        if paragraph.text.startswith("Table\u00a0 and Figure\u00a0 show that mean bank climate beta"):
            replace_paragraph(
                paragraph,
                "Table 1 and Figure 1 show that mean bank climate beta rises from 0.193 in 2019 to 0.424 in 2020. "
                "The paired increase is 0.230 (standard error 0.029), and all ten changes are positive. The daily "
                "diagnostic gives the same point estimate with HAC(203) standard error 0.104. Both estimates are "
                "significant at conventional levels. These statistics establish a common directional movement across "
                "institutions; because all banks face the same factor realization, they are not ten independent event "
                "replications. The twenty-BDC validation is similarly positive: the mean change is 0.177 (standard "
                "error 0.023), with 19 of 20 changes positive. Figure A1 displays the institution-level changes and "
                "the replication ratios.",
            )
        if paragraph.text.startswith("The magnitude comparison is also close"):
            replace_paragraph(
                paragraph,
                "The magnitude comparison is also close. End-2020 top-four marginal CRISK is $221.7 billion versus "
                "the published $260.0 billion; the signed CRISK increase is $372.8 billion versus $430.89 billion. "
                "The 85.3-percent marginal-CRISK ratio equals a 0.972 equity-scale ratio times a 0.877 climate-loss-rate "
                "ratio. Panel C preserves the nonlinear identity by computing 0.92E×LRMES daily and then averaging, "
                "rather than evaluating LRMES at mean beta.",
            )
        if paragraph.text.startswith("The annual result does not isolate a single transition event"):
            replace_paragraph(
                paragraph,
                "The annual result does not isolate a single transition event. The raw cross-bank mean is 0.1974 on "
                "9 March, 0.6329157 on 17 March, and reaches 1.0318 on 10 November after the Pfizer announcement. The "
                "17 March value is a raw ten-bank cross-sectional mean; WFC’s numerically similar 0.6328319 is a "
                "different object—its 31 December 127-day beta. The code audit verifies the two source slices and their "
                "8.37×10⁻⁵ difference. Figure 2 therefore separates the oil, COVID, and vaccine/value-rotation dates "
                "rather than assigning the November maximum to the March shock.",
            )
        if paragraph.text.startswith("Table\u00a0 describes the BDC panel"):
            replace_paragraph(
                paragraph,
                "Table 2 describes the BDC panel, while Table 3 and Figure 3 show how measurement resolution changes "
                "the portfolio result. In the firm- and quarter-fixed-effects equity specification, the coefficient is "
                "−0.062 for the broad brown share, 0.055 for FF12 portfolio beta, 0.103 for rolling-OLS FF49 beta, and "
                "0.152 for estimator-aligned DCC-FF49 beta. The final estimate has a clustered standard error of 0.085 "
                "and is significant at 10 percent under conventional two-sided inference. It remains 0.163 when "
                "low-confidence mappings are excluded, 0.196 after dropping 2021, and 0.150 after dropping the four "
                "geography-table imputations. The corresponding asset-beta estimate is 0.153 (standard error 0.100), "
                "nearly identical in magnitude but less precise. With 19 clusters, the equity wild-cluster bootstrap "
                "p-value is 0.153. The evidence is therefore suggestive and measurement-sensitive, not a robust "
                "rejection under every inference method.",
            )
        if paragraph.text.startswith("Table\u00a0 and Figure\u00a0 report a separate factor-maintenance diagnostic"):
            replace_paragraph(
                paragraph,
                "Table 4 and Figure 4 report a separate factor-maintenance diagnostic. Holding the international "
                "top-five basket fixed, weekly aggregation raises KOL tracking from 0.454 to 0.832 and the coarse-share "
                "equity coefficient from 0.025 to 0.154. The original sample ends in 2021, so the published basket "
                "bridges roughly one year after KOL liquidation; the article does not state a periodic refresh rule. "
                "Extending the same constituents through 2025 is a different maintenance problem, especially after "
                "Adaro’s 2024 thermal-coal separation. A live continuation needs transparent refresh triggers for "
                "mergers, spin-offs, and material business-mix changes. Public information does not establish V-Lab’s "
                "current production basket.",
            )
        if paragraph.text.startswith("The baseline buffer is"):
            replace_paragraph(
                paragraph,
                "The baseline buffer is AC − T. Equation (5) is retained only for the bank replication; "
                "its k = 8% parameter is not treated as a BDC capital requirement. Robustness uses "
                "month-end LRMES, book equity, a market-factor placebo, and each factor’s empirical "
                "first-percentile six-month return (−34.1% for climate and −16.8% for the market).",
            )
        if paragraph.text.startswith("The statutory mapping in Equation"):
            replace_paragraph(
                paragraph,
                "The statutory mapping in Equation (6) reduces mean coverage from 203.28% to 197.43% "
                "and the mean buffer from 53.15 to 47.30 percentage points. Table 5 and Figure 5 report "
                "5.85 points of mean compression, no primary-scenario breach, and an increase from six "
                "to eleven observations within 10 points of the threshold. The bank k = 8% calculation "
                "is shown only as a non-portability check and is not used to characterize BDC capital adequacy.",
            )


def style_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.90)
    section.right_margin = Inches(0.90)
    add_page_number(section)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    style_map = {style.name: style for style in document.styles}
    for style_name, size in (("Title", 15), ("Heading 1", 12.5), ("Heading 2", 11.5)):
        style = style_map[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(4)

    for i, paragraph in enumerate(document.paragraphs):
        if i < 4:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            if paragraph.style.name not in {"Title", "Heading 1", "Heading 2"}:
                set_font(run, 10.5, bold=run.bold, italic=run.italic)


def add_caption(document: Document, number: str, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(f"{number}.  {title}")
    set_font(run, 11, bold=True)


def add_note(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    lead = paragraph.add_run("Notes: ")
    set_font(lead, 8, italic=True)
    rest = paragraph.add_run(text)
    set_font(rest, 8)


def add_academic_table(
    document: Document,
    caption_no: str,
    title: str,
    rows: list[list[str]],
    panel_rows: set[int],
    header_rows: set[int],
    note: str,
    font_size: float = 7.5,
) -> None:
    document.add_page_break()
    add_caption(document, caption_no, title)
    table = document.add_table(rows=len(rows), cols=max(len(row) for row in rows))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_idx, row in enumerate(rows):
        for c_idx in range(len(table.columns)):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            text = row[c_idx] if c_idx < len(row) else ""
            cell.text = text
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_font(run, font_size, bold=(r_idx in panel_rows or r_idx in header_rows))
            top = "12" if r_idx == 0 else None
            bottom = "8" if r_idx in header_rows or r_idx == len(rows) - 1 else None
            set_cell_border(cell, top=top, bottom=bottom)
        if r_idx in panel_rows and len(row) == 1:
            table.cell(r_idx, 0).merge(table.cell(r_idx, len(table.columns) - 1))
    add_note(document, note)


def add_figure(document: Document, number: str, title: str, images: list[tuple[Path, float]], note: str) -> None:
    document.add_page_break()
    add_caption(document, number, title)
    for image, width in images:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run().add_picture(str(image), width=Inches(width))
    add_note(document, note)


def append_tables_and_figures(document: Document) -> None:
    add_academic_table(
        document,
        "Table 1",
        "Bank Replication of the 2020 Climate-Risk Shock",
        [
            ["", "Annual beta", "Daily beta", "Signed CRISK", "Positive CRISK"],
            ["Panel A: 2020 changes"],
            ["Estimate", "0.230***", "0.230**", "343.019***", "181.394***"],
            ["", "(0.029)", "(0.104)", "(81.640)", "(42.114)"],
            ["Observations", "10", "505", "505", "505"],
            ["Panel B: Published-magnitude comparison"],
            ["", "Replicated", "Published", "Ratio", "Qualification"],
            ["Annual-average peak year", "2021", "2020", "—", "2020–2021 plateau"],
            ["Raw daily peak", "1.0318", "—", "—", "Vaccine/value rotation"],
            ["Maximum 127-day beta (WFC)", "0.6328", "<0.500", "1.27", "Smoothing not stated"],
            ["Top-four mCRISK (USD bn)", "221.7", "260.0", "0.85", "85.3%"],
            ["Top-four CRISK increase (USD bn)", "372.8", "430.9", "0.87", "86.5%"],
            ["Panel C: End-2020 top-four 127-day mean construction"],
            ["Institution", "Mean beta (reference)", "Mean E×LRMES", "Mean mCRISK", "Identity error"],
            ["BAC", "0.5815", "73.32", "67.5", "0.00e+00"],
            ["C", "0.6326", "37.18", "34.2", "0.00e+00"],
            ["JPM", "0.5015", "93.69", "86.2", "0.00e+00"],
            ["WFC", "0.6328", "36.76", "33.8", "7.28e−15"],
            ["Total", "—", "240.94", "221.7", "7.28e−15"],
            ["Panel D: BDC time-series validation"],
            ["Sample", "Mean beta 2019", "Mean beta 2020", "Mean change", "Positive changes"],
            ["20 BDCs", "0.068", "0.245", "0.177***\n(0.023)", "19/20"],
            ["19-BDC exposure sample", "0.065", "0.244", "0.179***\n(0.024)", "18/19"],
        ],
        {1, 5, 12, 19},
        {0, 6, 13, 20},
        "Panel A’s CRISK estimates are differences between daily calendar-year means. Daily diagnostics use HAC(203). Because institutions share one factor realization, paired statistics measure cross-institution consistency rather than independent event replications. Panel C applies mCRISK = 0.92E×LRMES daily before averaging; mean beta is reference only. ***, **, and * denote two-sided significance at 1%, 5%, and 10%.",
        7.1,
    )

    add_academic_table(
        document,
        "Table 2",
        "BDC Panel Descriptive Statistics",
        [
            ["Variable", "N", "Mean", "SD", "Median", "Minimum", "Maximum"],
            ["Equity climate beta", "380", "0.098", "0.104", "0.094", "−0.276", "0.409"],
            ["Asset climate beta", "380", "0.047", "0.049", "0.045", "−0.103", "0.194"],
            ["Broad carbon-intensive share (%)", "380", "10.190", "7.681", "10.200", "0.000", "38.700"],
            ["Narrow carbon-intensive share (%)", "380", "4.258", "2.843", "4.300", "0.000", "11.834"],
            ["Total assets (USD mn)", "380", "5,257.993", "6,055.510", "3,081.480", "454.118", "31,235.000"],
            ["Debt to assets", "380", "0.507", "0.071", "0.517", "0.283", "0.689"],
            ["Quarterly ROA", "380", "0.011", "0.012", "0.011", "−0.072", "0.056"],
            ["Book to market", "380", "1.0278", "0.2500", "1.0312", "0.5026", "2.6256"],
            ["Market beta", "380", "0.642", "0.215", "0.629", "0.205", "1.354"],
            ["Reported asset coverage (%)", "376", "203.283", "39.438", "187.650", "155.700", "347.100"],
            ["Market to NAV", "380", "1.0277", "0.2496", "0.9697", "0.3809", "1.9897"],
        ],
        set(),
        {0},
        "The balanced panel contains 19 BDCs from 2021Q1 through 2025Q4. Book-to-market and market-to-NAV are observation-level reciprocals. The DCC-FF49 crosswalk maps 95.2% of reported weight overall and 97.5% in the median BDC-quarter. Four observations lack a resolved coverage ratio.",
        7.8,
    )

    add_academic_table(
        document,
        "Table 3",
        "BDC Portfolio Climate Beta and Market Climate Beta",
        [
            ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
            ["Panel A: Estimator-aligned DCC-FF49 portfolio beta"],
            ["Portfolio climate beta", "0.079", "0.078", "0.152*", "0.089", "0.044", "0.153"],
            ["", "(0.099)", "(0.093)", "(0.085)", "(0.131)", "(0.117)", "(0.100)"],
            ["Outcome", "Equity", "Equity", "Equity", "Asset", "Asset", "Asset"],
            ["Financial controls", "No", "Yes", "No", "No", "Yes", "No"],
            ["BDC fixed effects", "No", "No", "Yes", "No", "No", "Yes"],
            ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Observations", "380", "380", "380", "380", "380", "380"],
            ["R²", "0.346", "0.452", "0.450", "0.354", "0.441", "0.466"],
            ["Panel B: Measurement and sample robustness, equity beta"],
            ["", "Brown share", "FF12 beta", "FF49 OLS", "DCC-FF49", "High/medium", "Post-2021"],
            ["Coefficient", "−0.062", "0.055", "0.103", "0.152*", "0.163*", "0.196*"],
            ["", "(0.104)", "(0.091)", "(0.080)", "(0.085)", "(0.088)", "(0.107)"],
            ["BDC fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Quarter fixed effects", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Observations", "380", "380", "380", "380", "380", "304"],
            ["R²", "0.445", "0.445", "0.447", "0.450", "0.450", "0.454"],
        ],
        {1, 10},
        {0, 11},
        "Outcomes and portfolio measures are standardized. Industry and BDC betas use the same median scalar-DCC parameter vector. Standard errors are clustered by BDC. The full-sample equity wild-cluster bootstrap p-value is 0.153 with 9,999 draws. ***, **, and * denote two-sided significance at 1%, 5%, and 10%.",
        7.6,
    )

    add_academic_table(
        document,
        "Table 4",
        "Coarse-Share and Factor-Continuation Diagnostics",
        [
            ["Panel A: Continuation and return frequency"],
            ["", "Top-five daily\nEquity", "Top-five daily\nAsset", "U.S. coal daily\nEquity", "U.S. coal daily\nAsset", "Top-five weekly\nEquity", "Top-five weekly\nAsset"],
            ["Broad investment share", "0.025\n(0.093)", "0.014\n(0.089)", "0.094\n(0.102)", "0.114\n(0.078)", "0.154\n(0.238)", "0.149\n(0.185)"],
            ["KOL tracking correlation", "0.454", "0.454", "0.601", "0.601", "0.832", "0.832"],
            ["Observations", "380", "380", "380", "380", "380", "380"],
            ["Panel B: Classification, timing, and power"],
            ["", "Narrow", "Lower", "Upper", "Lagged", "Post-2021", "MDE₈₀"],
            ["Equity beta", "−0.007\n(0.084)", "0.021\n(0.092)", "−0.002\n(0.067)", "0.029\n(0.097)", "−0.004\n(0.103)", "0.273"],
            ["Asset beta", "−0.021\n(0.101)", "0.015\n(0.087)", "−0.005\n(0.073)", "0.012\n(0.089)", "−0.016\n(0.097)", "0.260"],
            ["Observations", "380", "380", "380", "361", "304", "380"],
            ["Panel C: High-yield credit-return factor"],
            ["", "Baseline eq.", "HY equity", "Baseline asset", "HY asset", "HY+controls", "HY+BDC FE"],
            ["Broad investment share", "0.025\n(0.093)", "0.078\n(0.134)", "0.014\n(0.089)", "0.048\n(0.117)", "−0.007\n(0.076)", "0.102\n(0.140)"],
            ["Observations", "380", "380", "380", "380", "380", "380"],
            ["Correlation-one extrapolation", "0.215", "—", "—", "—", "—", "—"],
        ],
        {0, 5, 10},
        {1, 6, 11},
        "These are coarse carbon-intensive-share diagnostics; the granular portfolio-beta models appear in Table 3. All regressions include quarter fixed effects and BDC-clustered standard errors. The correlation-one extrapolation is descriptive. No coefficient reaches conventional two-sided significance.",
        7.1,
    )

    add_academic_table(
        document,
        "Table 5",
        "Climate Stress and BDC Asset-Coverage Capacity",
        [
            ["Panel A: Scenario comparison"],
            ["", "Climate 50%", "Climate p01", "Market p01", "Market 50%", "Climate/NAV"],
            ["Mean buffer compression (pp)", "5.849", "3.633", "10.583", "33.671", "5.908"],
            ["Median buffer compression (pp)", "5.734", "3.511", "9.367", "29.860", "5.601"],
            ["Mean compression / mean buffer (%)", "11.01", "6.84", "19.91", "63.35", "11.11"],
            ["Legal breaches", "0", "0", "0", "82", "0"],
            ["Observations", "376", "376", "376", "376", "376"],
            ["Panel B: Climate 50-percent timing and loss mapping"],
            ["", "Monthly mean", "Month-end", "Beta-implied", "Positive-loss", ""],
            ["Mean buffer compression (pp)", "5.849", "6.449", "5.899", "6.515", ""],
            ["Median buffer compression (pp)", "5.734", "5.665", "5.751", "5.734", ""],
            ["Within 10 pp after stress", "11", "15", "11", "13", ""],
            ["Legal breaches", "0", "1", "0", "0", ""],
        ],
        {0, 7},
        {1, 8},
        "The p01 columns apply each factor’s separate marginal first percentile: −34.1% for climate and −16.8% for the market. The 50% market shock is a scale placebo. NAV replaces market equity. No stars are reported because positive beta mechanically maps into positive compression and first-stage DCC uncertainty is omitted.",
        7.5,
    )

    b1 = ROOT / "01_Bank_CRISK_Replication" / "Results"
    b2 = ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results"
    b3 = ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results"
    add_figure(document, "Figure 1", "Annual Climate Beta for Banks and BDCs", [(b1 / "Figure_1_Aggregate_Climate_Beta.png", 6.35)], "The upper panel plots annual cross-institution mean dynamic climate beta; the lower panel reports active institutions. KOL is used through 14 December 2020 and the fixed top-five basket thereafter.")
    add_figure(document, "Figure 2", "Top-Four Bank CRISK Around the 2020 Shock", [(b1 / "Figure_2_Top_Four_Bank_CRISK.png", 6.35)], "The upper panel reports raw daily series and the lower panel reports trailing 127-day means. Vertical markers identify the oil-price break, COVID dislocation, and Pfizer–BioNTech announcement.")
    add_figure(document, "Figure 3", "Measurement Resolution and the BDC Portfolio Mechanism", [(b2 / "Figure_3_FF49_DCC_Portfolio_Mechanism.png", 6.35)], "Bars are 90% confidence intervals under BDC-clustered standard errors; orange markers denote conventional two-sided significance at 10%. The corresponding wild-cluster bootstrap does not reject at 10%.")
    add_figure(document, "Figure 4", "Exposure Sensitivity to Factor Construction", [(b2 / "Figure_5_Factor_Continuation_Sensitivity.png", 6.35)], "The connected top-five points hold basket content fixed and change frequency; the U.S. coal point also changes economic content. The correlation-one extrapolation is descriptive.")
    add_figure(document, "Figure 5", "BDC Asset Coverage Before and After Climate Stress", [(b3 / "Figure_6_Asset_Coverage_Before_After.png", 5.8), (b3 / "Figure_7_Threshold_Proximity.png", 5.8)], "Panel A plots mean reported and climate-stressed asset coverage. Panel B counts BDC-quarter observations within cumulative distances from the statutory threshold. No observation crosses the threshold in the primary scenario.")
    add_figure(document, "Figure A1", "Cross-Institution Replication Diagnostics", [(b1 / "Figure03_Bank_Level_Beta_Changes.png", 5.8), (b1 / "Figure04_Published_vs_Replicated.png", 5.8)], "Panel A shows that all ten banks have positive 2019–2020 beta changes. Panel B compares published and independently replicated top-four magnitudes.")
    add_figure(document, "Figure A2", "BDC Exposure Diagnostics", [(b2 / "Figure_3_Investment_Exposure_and_Climate_Beta.png", 5.8), (b2 / "Figure_4_Investment_Exposure_Trends.png", 5.8)], "Panel A provides the raw coarse-share relationship retained in Table 4. Panel B plots cross-sectional exposure summaries through time; it does not establish within-firm persistence. The main DCC-FF49 result appears in Table 3 and Figure 3.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="crisk_word_") as temporary:
        source = Path(temporary) / "pandoc_source.docx"
        subprocess.run(
            [
                "pandoc",
                str(HERE / "Climate_Risk_and_BDCs.tex"),
                "--from=latex",
                "--to=docx",
                f"--resource-path={HERE}:{ROOT}",
                f"--output={source}",
            ],
            check=True,
        )
        document = Document(source)
        clean_body(document)
        style_document(document)
        append_tables_and_figures(document)
        document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
