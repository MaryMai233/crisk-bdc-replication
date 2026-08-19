from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from lxml import html


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = ROOT / "Data" / "Processed" / "Audit"
DEFAULT_RAW_ZIP = ROOT / "Data" / "Raw" / "SEC_BDC_Filings_2021_2025.zip"


def load_base_module():
    path = Path(__file__).with_name("02_extract_bdc_portfolio_exposure.py")
    spec = importlib.util.spec_from_file_location("dynamic_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def grid_cells(tr) -> list[tuple[int, int, str]]:
    position = 0
    cells: list[tuple[int, int, str]] = []
    for cell in tr.xpath("./th|./td"):
        span = int(cell.get("colspan", "1"))
        cells.append((position, span, BASE.clean_text(cell.text_content())))
        position += span
    return cells


def infer_first_percent_position(table) -> int | None:
    for tr in table.xpath(".//tr"):
        cells = grid_cells(tr)
        label = BASE.clean_label(next((text for _, _, text in cells if text), ""))
        if not BASE.valid_industry_label(label):
            continue
        for index, (position, _, text) in enumerate(cells):
            if text.strip() != "%" and "%" not in text:
                continue
            for previous in reversed(cells[:index]):
                number = BASE.parse_number(previous[2])
                if number is not None and 0 <= number <= 100:
                    return previous[0]
    return None


def rows_at_percent_position(table, method: str, table_index: int) -> list[dict]:
    percent_position = infer_first_percent_position(table)
    if percent_position is None:
        return []
    rows: list[dict] = []
    for tr in table.xpath(".//tr"):
        cells = grid_cells(tr)
        label = BASE.clean_label(next((text for _, _, text in cells if text), ""))
        if not BASE.valid_industry_label(label):
            continue
        value = next((text for position, _, text in cells if position == percent_position), "")
        pct = BASE.parse_number(value)
        if pct is None or not 0 <= pct <= 100:
            continue
        rows.append({
            "industry_reported": label,
            "reported_pct": float(pct),
            "source_method": method,
            "source_table_index": str(table_index),
        })
    return rows


def extract_cgbd(document) -> list[dict]:
    tables = document.xpath("//table")
    starts = [
        index for index, table in enumerate(tables)
        if re.search(r"Industry.*Amortized Cost.*Fair Value.*% of Fair Value", BASE.clean_text(table.text_content()), re.I)
    ]
    for start in starts:
        block: list[dict] = []
        used: list[int] = []
        for index in range(start, min(start + 4, len(tables))):
            text = BASE.clean_text(tables[index].text_content())
            if index > start and not re.search(r"Industry.*Amortized Cost.*Fair Value.*% of Fair Value", text, re.I):
                break
            part = rows_at_percent_position(
                tables[index], "split_direct_industry_fair_value_percentage", index
            )
            if not part:
                break
            block.extend(part)
            used.append(index)
            total = sum(item["reported_pct"] for item in block)
            if 97 <= total <= 103:
                for item in block:
                    item["source_table_index"] = ";".join(map(str, used))
                return block
    raise ValueError("CGBD complete current-period industry percentage block not found")


def extract_gbdc(document, report_date: str) -> list[dict]:
    phrase = BASE.report_date_phrase(report_date)
    candidates: list[tuple[int, list[dict]]] = []
    for index, table in enumerate(document.xpath("//table")):
        text = BASE.clean_text(table.text_content()).lower()
        if phrase not in text or "fair value:" not in text:
            continue
        rows = rows_at_percent_position(
            table, "comparative_fair_value_industry_percentage", index
        )
        total = sum(item["reported_pct"] for item in rows)
        if len(rows) >= 20 and 90 <= total <= 110:
            candidates.append((index, rows))
    if not candidates:
        raise ValueError("GBDC current-period fair-value industry table not found")
    return max(candidates, key=lambda pair: len(pair[1]))[1]


def extract_pflt(document, report_date: str) -> list[dict]:
    phrase = BASE.report_date_phrase(report_date)
    candidates: list[tuple[int, list[dict]]] = []
    for index, table in enumerate(document.xpath("//table")):
        text = BASE.clean_text(table.text_content()).lower()
        if "industry classification" not in text or phrase not in text:
            continue
        rows: list[dict] = []
        for tr in table.xpath(".//tr"):
            values = BASE.row_cells(tr)
            if not values:
                continue
            label = BASE.clean_label(next((value for value in values if value), ""))
            if not BASE.valid_industry_label(label):
                continue
            pct = next(
                (
                    number for value in values[1:]
                    if (number := BASE.parse_number(value)) is not None and 0 <= number <= 100
                ),
                None,
            )
            if pct is None:
                continue
            rows.append({
                "industry_reported": label,
                "reported_pct": float(pct),
                "source_method": "direct_industry_classification_percentage",
                "source_table_index": str(index),
            })
        total = sum(item["reported_pct"] for item in rows)
        if len(rows) >= 15 and 90 <= total <= 110:
            candidates.append((index, rows))
    if not candidates:
        raise ValueError("PFLT current-period industry classification table not found")
    return max(candidates, key=lambda pair: len(pair[1]))[1]


def extract_cswc(document, report_date: str) -> list[dict]:
    phrase = BASE.report_date_phrase(report_date)
    aggregate: dict[str, float] = defaultdict(float)
    source_tables: set[int] = set()
    pattern = re.compile(r"^Subtotal:\s*(.*?)(?:\s*\(([-+]?\d+(?:\.\d+)?)%\)\*?)?$", re.I)
    for index, table in enumerate(document.xpath("//table")):
        text = BASE.clean_text(table.text_content()).lower()
        if phrase not in text[:700]:
            continue
        for tr in table.xpath(".//tr"):
            values = BASE.row_cells(tr)
            if not values:
                continue
            match = pattern.search(values[0])
            if not match:
                continue
            pct = float(match.group(2)) if match.group(2) else BASE.first_percent(values)
            if pct is None:
                continue
            aggregate[BASE.clean_label(match.group(1))] += pct
            source_tables.add(index)
    total = sum(aggregate.values())
    if len(aggregate) >= 10 and total > 0:
        return [
            {
                "industry_reported": industry,
                "reported_pct": value / total * 100,
                "source_method": "schedule_industry_subtotal_percent_of_net_assets_normalized",
                "source_table_index": ";".join(map(str, sorted(source_tables))),
            }
            for industry, value in sorted(aggregate.items())
        ]

    # Earlier filings place an industry field on each investment row rather
    # than reporting industry subtotals.  Aggregate current-period line-item
    # fair values using the HTML grid positions declared in each repeated
    # schedule header.
    fair_values: dict[str, float] = defaultdict(float)
    line_tables: set[int] = set()
    current_industry: str | None = None
    for index, table in enumerate(document.xpath("//table")):
        text = BASE.clean_text(table.text_content()).lower()
        if phrase not in text[:700] or "schedule of investments" not in text[:700]:
            continue
        header_positions: dict[str, int] = {}
        for tr in table.xpath(".//tr"):
            for position, _, value in grid_cells(tr):
                low = value.lower()
                if low.startswith("portfolio company"):
                    header_positions.setdefault("company", position)
                elif low.startswith("type of investment") or low == "type of":
                    header_positions.setdefault("type", position)
                elif low == "industry":
                    header_positions.setdefault("industry", position)
                elif low.startswith("fair value") or low == "fair":
                    header_positions.setdefault("fair_value", position)
        if not {"company", "type", "industry", "fair_value"}.issubset(header_positions):
            continue
        for tr in table.xpath(".//tr"):
            cells = grid_cells(tr)
            cell_map = {position: value for position, _, value in cells}
            proposed = BASE.clean_label(cell_map.get(header_positions["industry"], ""))
            if BASE.valid_industry_label(proposed):
                current_industry = proposed
            company = cell_map.get(header_positions["company"], "").strip()
            investment_type = cell_map.get(header_positions["type"], "").strip()
            if not current_industry or not (company or investment_type):
                continue
            fair_value = None
            for position, _, value in cells:
                if position < header_positions["fair_value"]:
                    continue
                parsed = BASE.parse_number(value)
                if parsed is not None:
                    fair_value = parsed
                    break
            if fair_value is None or fair_value == 0:
                continue
            fair_values[current_industry] += fair_value
            line_tables.add(index)
    total_fair_value = sum(fair_values.values())
    if len(fair_values) < 10 or total_fair_value <= 0:
        raise ValueError("CSWC current-period schedule subtotals or line items not found")
    return [
        {
            "industry_reported": industry,
            "reported_pct": value / total_fair_value * 100,
            "source_method": "schedule_line_item_fair_value_aggregated_by_industry",
            "source_table_index": ";".join(map(str, sorted(line_tables))),
        }
        for industry, value in sorted(fair_values.items())
    ]


def classify_rows(rows: list[dict], manifest_row: dict[str, str]) -> list[dict]:
    total = sum(float(item["reported_pct"]) for item in rows)
    normalize = 97 <= total <= 110
    output: list[dict] = []
    for item in rows:
        pct = float(item["reported_pct"]) / total * 100 if normalize else float(item["reported_pct"])
        narrow, broad, narrow_match, broad_match = BASE.classify_label(str(item["industry_reported"]))
        output.append({
            "ticker": manifest_row["ticker"],
            "report_date": manifest_row["report_date"],
            "calendar_quarter": manifest_row["calendar_quarter"],
            "form": manifest_row["form"],
            "industry_reported": item["industry_reported"],
            "fair_value": pd.NA,
            "portfolio_fair_value_pct": pct,
            "source_weight_sum_raw": total,
            "weight_normalized_to_100": int(normalize),
            "brown_narrow": narrow,
            "brown_broad": broad,
            "narrow_rule_match": narrow_match,
            "broad_extra_rule_match": broad_match,
            "source_method": item["source_method"],
            "source_table_index": item["source_table_index"],
            "source_file": manifest_row["local_file"],
        })
    return output


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="object"):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", type=Path, default=DEFAULT_RAW_ZIP)
    args = parser.parse_args()
    base = pd.read_csv(PROCESSED / "dynamic_industry_exposure_summary_only.csv")
    status = pd.read_csv(AUDIT / "dynamic_extraction_filing_status.csv", dtype={"calendar_quarter": str})
    fallback_keys = set(
        zip(
            status.loc[status["extraction_status"].eq("FALLBACK_REQUIRED"), "ticker"],
            status.loc[status["extraction_status"].eq("FALLBACK_REQUIRED"), "report_date"],
        )
    )
    details: list[dict] = []
    fallback_audit: list[dict] = []
    with zipfile.ZipFile(args.raw_zip) as archive:
        manifest = [
            row for row in BASE.load_manifest(archive)
            if row["form"] in {"10-Q", "10-K"}
            and row["status"] in {"DOWNLOADED", "ALREADY_PRESENT"}
            and (row["ticker"], row["report_date"]) in fallback_keys
        ]
        for sequence, row in enumerate(sorted(manifest, key=lambda value: (value["ticker"], value["report_date"])), start=1):
            document = html.fromstring(archive.read(BASE.source_member(row)))
            extractor = {
                "CGBD": lambda: extract_cgbd(document),
                "CSWC": lambda: extract_cswc(document, row["report_date"]),
                "GBDC": lambda: extract_gbdc(document, row["report_date"]),
                "PFLT": lambda: extract_pflt(document, row["report_date"]),
            }[row["ticker"]]
            extracted = extractor()
            classified = classify_rows(extracted, row)
            details.extend(classified)
            fallback_audit.append({
                "ticker": row["ticker"],
                "report_date": row["report_date"],
                "industry_rows": len(classified),
                "weight_sum": sum(item["portfolio_fair_value_pct"] for item in classified),
                "source_weight_sum_raw": classified[0]["source_weight_sum_raw"],
                "source_method": classified[0]["source_method"],
                "source_table_index": classified[0]["source_table_index"],
                "status": "PASS",
            })
            print(f"[{sequence:02d}/{len(manifest):02d}] {row['ticker']} {row['report_date']}: PASS")

    fallback = pd.DataFrame(details)
    combined = pd.concat([base, fallback], ignore_index=True, sort=False)
    group_sum = combined.groupby(["ticker", "report_date"])["portfolio_fair_value_pct"].transform("sum")
    normalize_mask = group_sum.between(97.0, 110.0)
    combined.loc[normalize_mask, "portfolio_fair_value_pct"] = (
        combined.loc[normalize_mask, "portfolio_fair_value_pct"]
        / group_sum.loc[normalize_mask]
        * 100
    )
    combined.loc[normalize_mask, "weight_normalized_to_100"] = 1
    combined = combined.sort_values(["ticker", "report_date", "industry_reported"]).reset_index(drop=True)
    combined.to_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv", index=False)
    export_dta(combined, PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.dta")
    fallback_frame = pd.DataFrame(fallback_audit)
    fallback_frame.to_csv(AUDIT / "dynamic_fallback_extraction_audit.csv", index=False)

    company_quarter = (
        combined.groupby(["ticker", "report_date", "calendar_quarter"], as_index=False)
        .agg(
            industry_rows=("industry_reported", "nunique"),
            mapped_weight_pct=("portfolio_fair_value_pct", "sum"),
            brown_narrow_pct=("portfolio_fair_value_pct", lambda s: float(s[combined.loc[s.index, "brown_narrow"].eq(1)].sum())),
            brown_broad_pct=("portfolio_fair_value_pct", lambda s: float(s[combined.loc[s.index, "brown_broad"].eq(1)].sum())),
            source_method=("source_method", "first"),
        )
        .sort_values(["ticker", "report_date"])
    )
    company_quarter.to_csv(PROCESSED / "dynamic_bdc_company_quarter_exposure_2021_2025.csv", index=False)
    export_dta(company_quarter, PROCESSED / "dynamic_bdc_company_quarter_exposure_2021_2025.dta")
    counts = company_quarter.groupby("ticker").size()
    audit = {
        "status": "PASS" if len(company_quarter) == 380 and counts.eq(20).all() else "FAIL",
        "company_quarters": int(len(company_quarter)),
        "firms": int(company_quarter["ticker"].nunique()),
        "quarters_per_firm_min": int(counts.min()),
        "quarters_per_firm_max": int(counts.max()),
        "direct_or_summary_quarters": int(len(company_quarter) - len(fallback_frame)),
        "fallback_quarters": int(len(fallback_frame)),
        "mapped_weight_min": float(company_quarter["mapped_weight_pct"].min()),
        "mapped_weight_max": float(company_quarter["mapped_weight_pct"].max()),
        "duplicate_company_quarters": int(company_quarter.duplicated(["ticker", "report_date"]).sum()),
    }
    (AUDIT / "dynamic_complete_exposure_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("Dynamic complete exposure audit failed")


if __name__ == "__main__":
    main()
