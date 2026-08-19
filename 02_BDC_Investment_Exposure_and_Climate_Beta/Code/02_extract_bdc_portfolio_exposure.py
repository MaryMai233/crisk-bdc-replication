from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ZIP = ROOT / "Data" / "Raw" / "SEC_BDC_Filings_2021_2025.zip"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = ROOT / "Data" / "Processed" / "Audit"

NARROW_PATTERNS = [
    r"\benergy\b", r"oil", r"consumable fuels", r"\bcoal\b",
    r"\bmaterials\b", r"chemicals?", r"construction materials",
    r"containers?\s*(?:&|and)\s*packaging", r"\bpackaging\b",
    r"metals?\s*(?:&|and)\s*mining", r"paper\s*(?:&|and)\s*forest",
    r"air freight", r"airlines?", r"marine transportation",
    r"ground transportation", r"transportation infrastructure",
    r"\btransportation\b", r"\btrucking\b", r"road\s*(?:&|and)\s*rail",
    r"electric utilities", r"gas utilities", r"multi-utilities",
]
BROAD_EXTRA_PATTERNS = [
    r"automobiles?", r"auto components?", r"automobile components?",
    r"automotive", r"aerospace", r"capital goods",
    r"construction\s*(?:&|and)\s*engineering", r"electrical equipment",
    r"\bmachinery\b", r"manufacturing technology",
    r"trading companies", r"independent power",
]

SUMMARY_HEADER_PATTERNS = [
    r"percentage\s+of\s+portfolio",
    r"percentage\s+of\s+total\s+investments",
    r"percent\s+of\s+total\s*investments",
    r"percent\s+of\s+total\s*investment",
    r"percent\s+of\s+totalinvestments",
    r"percent\s+of\s+total\s*investments\s+at\s+fair\s*value",
    r"percent\s+of\s+totalinvestments\s+at\s+fairvalue",
    r"percent\s+of\s+total\s*investments\s*\(at\s+fair\s*value\)",
    r"industry.{0,80}%\s+of\s+fair\s+value",
    r"industry.{0,80}percentage",
    r"industry\s+type.{0,80}percent",
    r"industry\s+classification.{0,120}percentage",
    r"investments\s+at\s+fair\s+value\s+by\s+industry",
]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def row_cells(tr) -> list[str]:
    return [clean_text(c.text_content()) for c in tr.xpath("./th|./td")]


def parse_number(value: str) -> float | None:
    text = value.replace(",", "").replace("$", "").strip()
    if text in {"", "$", "%", "-", "—", "–", "N/A", "n/a"}:
        return None
    negative = re.fullmatch(r"\(([-+]?\d+(?:\.\d+)?)\)", text)
    if negative:
        return -abs(float(negative.group(1)))
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)
    return None


def first_percent(values: list[str]) -> float | None:
    """Return the first percentage represented in a SEC table row."""
    for index, value in enumerate(values):
        match = re.fullmatch(r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*%\s*\)?", value.replace(",", ""))
        if match:
            number = float(match.group(1))
            return number if 0 <= number <= 100 else None
        if value.strip() == "%":
            for previous in reversed(values[:index]):
                number = parse_number(previous)
                if number is not None:
                    return number if 0 <= number <= 100 else None
    return None


def infer_percent_column(rows: list[list[str]]) -> int | None:
    """Infer the current-period percentage column from the first visible % marker."""
    for values in rows:
        label = clean_label(next((value for value in values if value), ""))
        if not valid_industry_label(label):
            continue
        for index, value in enumerate(values):
            if value.strip() != "%" and "%" not in value:
                continue
            for previous_index in range(index - 1, -1, -1):
                if parse_number(values[previous_index]) is not None:
                    return previous_index
    return None


def classify_label(label: str) -> tuple[int, int, str, str]:
    normalized = re.sub(r"\(\d+\)", "", label).strip().lower()
    narrow_matches = [p for p in NARROW_PATTERNS if re.search(p, normalized)]
    broad_extra_matches = [p for p in BROAD_EXTRA_PATTERNS if re.search(p, normalized)]
    narrow = int(bool(narrow_matches))
    broad = int(bool(narrow_matches) or bool(broad_extra_matches))
    return narrow, broad, "; ".join(narrow_matches), "; ".join(broad_extra_matches)


def is_summary_header(table_text: str) -> bool:
    normalized = clean_text(table_text).lower()
    pattern_match = any(re.search(pattern, normalized) for pattern in SUMMARY_HEADER_PATTERNS)
    industry_context = "industry" in normalized and any(
        phrase in normalized
        for phrase in (
            "fair value", "% of portfolio", "percent of total",
            "percentage", "portfolio investments",
        )
    )
    return pattern_match or industry_context


def clean_label(value: str) -> str:
    return clean_text(re.sub(r"\(\d+\)", "", value)).strip(" :")


def valid_industry_label(label: str) -> bool:
    low = label.lower()
    if not label or len(label) > 180 or not re.search(r"[A-Za-z]", label):
        return False
    exclusions = (
        "industry", "total", "subtotal", "as of", "december", "march", "june",
        "september", "january", "february", "april", "may", "july", "august",
        "october", "november", "portfolio company", "investments", "fair value",
        "percentage", "percent of", "dollars in", "debt investments at",
    )
    return not low.startswith(exclusions)


def extract_summary_candidates(document: html.HtmlElement) -> list[dict]:
    candidates: list[dict] = []
    for table_index, table in enumerate(document.xpath("//table")):
        text = clean_text(table.text_content())
        psec_cost_fair_layout = bool(re.search(
            r"cost\s*%\s*of\s*portfolio\s*fair\s*value\s*%\s*of\s*portfolio",
            text.lower(),
        ))
        row_elements = table.xpath(".//tr")
        rows = [row_cells(tr) for tr in row_elements]
        psec_fair_pct_position = None
        if psec_cost_fair_layout:
            for tr in row_elements:
                position = 0
                occurrences: list[int] = []
                for cell in tr.xpath("./th|./td"):
                    value = clean_text(cell.text_content()).lower()
                    if value == "% of portfolio":
                        occurrences.append(position)
                    position += int(cell.get("colspan", "1"))
                if len(occurrences) >= 2:
                    psec_fair_pct_position = occurrences[1]
                    break
        percent_column = infer_percent_column(rows)
        if percent_column is None:
            continue
        extracted: list[tuple[str, float]] = []
        for tr, values in zip(row_elements, rows):
            if not values:
                continue
            label = clean_label(next((value for value in values if value), ""))
            if not valid_industry_label(label):
                continue
            if psec_cost_fair_layout:
                pct = None
                position = 0
                for cell in tr.xpath("./th|./td"):
                    value = clean_text(cell.text_content())
                    if position == psec_fair_pct_position:
                        pct = parse_number(value)
                        break
                    position += int(cell.get("colspan", "1"))
            else:
                pct = first_percent(values)
                if pct is None and percent_column < len(values):
                    pct = parse_number(values[percent_column])
            if pct is None or not 0 <= pct <= 100:
                continue
            extracted.append((label, pct))
        if len(extracted) < 4:
            continue
        frame = pd.DataFrame(extracted, columns=["industry_reported", "reported_pct"])
        # Inline-XBRL filings commonly repeat an identical visual table in the
        # same HTML node.  Remove exact duplicates before adding genuine rows.
        # A two-period summary often appears as two logical rows with the same
        # industry label.  The filing presents the current period first, so keep
        # its first occurrence instead of averaging current and prior weights.
        frame = frame.drop_duplicates(subset=["industry_reported"], keep="first")
        frame = frame.groupby("industry_reported", as_index=False)["reported_pct"].sum()
        total = float(frame["reported_pct"].sum())
        candidates.append({
            "table_index": table_index,
            "rows": frame.to_dict(orient="records"),
            "industry_rows": int(len(frame)),
            "weight_sum": total,
            "percent_column": percent_column,
            "strong_summary_header": int(is_summary_header(text)),
            "header_excerpt": text[:500],
        })
    return candidates


def report_date_phrase(report_date: str) -> str:
    parsed = pd.Timestamp(report_date)
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}".lower()


def candidate_score(candidate: dict, report_date: str = "") -> float:
    total = float(candidate["weight_sum"])
    n = int(candidate["industry_rows"])
    score = 0.0
    if 97.0 <= total <= 103.0:
        score += 100.0
    elif 90.0 <= total <= 110.0:
        score += 60.0
    elif 70.0 <= total <= 130.0:
        score += 20.0
    # Complete industry tables normally contain substantially more categories
    # than geography, ratings, or asset-type summaries.
    score += min(n, 50)
    header = str(candidate["header_excerpt"]).lower()
    labels = [str(row["industry_reported"]) for row in candidate.get("rows", [])]
    company_like = sum(
        bool(re.search(r"\b(llc|inc\.?|corp\.?|corporation|l\.p\.?|plc|ltd\.?|holdings?|buyer|borrower)\b", label.lower()))
        for label in labels
    )
    asset_like = sum(
        bool(re.search(r"first lien|second lien|secured debt|unsecured|preferred stock|common stock|member units|intellectual property", label.lower()))
        for label in labels
    )
    numeric_like = sum(not bool(re.search(r"[A-Za-z]", label)) for label in labels)
    geography_like = sum(
        bool(re.search(
            r"\b(midwest|northeast|southeast|southwest|west|international|"
            r"united states|canada|europe|asia|north america)\b",
            label.lower(),
        ))
        for label in labels
    )
    if labels and company_like / len(labels) >= 0.25:
        score -= 100.0
    if labels and asset_like / len(labels) >= 0.25:
        score -= 100.0
    if labels and numeric_like / len(labels) >= 0.25:
        score -= 100.0
    if labels and geography_like / len(labels) >= 0.40:
        score -= 100.0
    if "portfolio company" in header:
        score -= 30.0
    if re.search(
        r"per share|net investment income|compensation expense|operating expense|"
        r"financial highlights|supplemental non-gaap|investment rating|asset type|"
        r"interest income|interest and financing expenses|change % change|"
        r"results of operations|balance sheets|net increase|net decrease",
        header,
    ):
        score -= 100.0
    if "all other industries" in header and n <= 6:
        score -= 60.0
    if n < 7 and not int(candidate.get("strong_summary_header", 0)):
        score -= 50.0
    if "percentage of portfolio" in header or "percent of total" in header:
        score += 10.0
    if int(candidate.get("strong_summary_header", 0)):
        score += 40.0
    if report_date and report_date_phrase(report_date) in header:
        score += 20.0
    if "fair value:" in header:
        score += 15.0
    if "cost:" in header:
        score -= 15.0
    return score


def choose_candidate(candidates: list[dict], report_date: str) -> dict | None:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda candidate: (candidate_score(candidate, report_date), candidate["industry_rows"]),
        reverse=True,
    )
    best = ranked[0]
    return best if candidate_score(best, report_date) >= 50.0 else None


def load_manifest(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    path = "SEC_BDC_Filings_2021_2025/audit/filing_manifest.csv"
    with archive.open(path) as handle:
        reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
        return list(reader)


def source_member(row: dict[str, str]) -> str:
    return "SEC_BDC_Filings_2021_2025/" + row["local_file"].replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", type=Path, default=DEFAULT_RAW_ZIP)
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic limit; 0 means all filings.")
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    details: list[dict] = []
    filing_audit: list[dict] = []
    candidate_audit: list[dict] = []
    with zipfile.ZipFile(args.raw_zip) as archive:
        manifest = [
            row for row in load_manifest(archive)
            if row["form"] in {"10-Q", "10-K"} and row["status"] in {"DOWNLOADED", "ALREADY_PRESENT"}
        ]
        manifest = sorted(manifest, key=lambda row: (row["ticker"], row["report_date"], row["filing_date"]))
        if args.limit:
            manifest = manifest[: args.limit]
        for sequence, row in enumerate(manifest, start=1):
            member = source_member(row)
            raw = archive.read(member)
            document = html.fromstring(raw)
            candidates = extract_summary_candidates(document)
            selected = choose_candidate(candidates, row["report_date"])
            for candidate in candidates:
                candidate_audit.append({
                    "ticker": row["ticker"],
                    "report_date": row["report_date"],
                    "form": row["form"],
                    "table_index": candidate["table_index"],
                    "industry_rows": candidate["industry_rows"],
                    "weight_sum": candidate["weight_sum"],
                    "percent_column": candidate["percent_column"],
                    "strong_summary_header": candidate["strong_summary_header"],
                    "candidate_score": candidate_score(candidate, row["report_date"]),
                    "selected": int(candidate is selected),
                    "header_excerpt": candidate["header_excerpt"],
                })
            status = "SUMMARY_SELECTED" if selected else "FALLBACK_REQUIRED"
            filing_audit.append({
                "ticker": row["ticker"],
                "report_date": row["report_date"],
                "calendar_quarter": row["calendar_quarter"],
                "form": row["form"],
                "source_file": row["local_file"],
                "html_tables": len(document.xpath("//table")),
                "summary_candidates": len(candidates),
                "selected_table_index": selected["table_index"] if selected else np.nan,
                "selected_industry_rows": selected["industry_rows"] if selected else 0,
                "selected_weight_sum_raw": selected["weight_sum"] if selected else np.nan,
                "extraction_status": status,
            })
            if selected:
                total = float(selected["weight_sum"])
                for item in selected["rows"]:
                    industry = str(item["industry_reported"])
                    # Nearly complete tables (rounding around 100%) are
                    # normalized.  Subportfolio tables explicitly measured as
                    # a percent of the total portfolio (for example HRZN debt)
                    # retain their reported weights; the residual stays
                    # unclassified rather than being redistributed.
                    normalize = 97.0 <= total <= 110.0
                    pct = float(item["reported_pct"]) / total * 100 if normalize else float(item["reported_pct"])
                    narrow, broad, narrow_match, broad_match = classify_label(industry)
                    details.append({
                        "ticker": row["ticker"],
                        "report_date": row["report_date"],
                        "calendar_quarter": row["calendar_quarter"],
                        "form": row["form"],
                        "industry_reported": industry,
                        "fair_value": np.nan,
                        "portfolio_fair_value_pct": pct,
                        "source_weight_sum_raw": total,
                        "weight_normalized_to_100": int(normalize),
                        "brown_narrow": narrow,
                        "brown_broad": broad,
                        "narrow_rule_match": narrow_match,
                        "broad_extra_rule_match": broad_match,
                        "source_method": "complete_filing_industry_percentage_table",
                        "source_table_index": selected["table_index"],
                        "source_file": row["local_file"],
                    })
            print(f"[{sequence:03d}/{len(manifest):03d}] {row['ticker']} {row['report_date']}: {status}")
            del document, raw
            gc.collect()

    filing_frame = pd.DataFrame(filing_audit)
    detail_frame = pd.DataFrame(details)
    candidate_frame = pd.DataFrame(candidate_audit)
    filing_frame.to_csv(AUDIT / "dynamic_extraction_filing_status.csv", index=False)
    candidate_frame.to_csv(AUDIT / "dynamic_summary_candidate_audit.csv", index=False)
    detail_frame.to_csv(PROCESSED / "dynamic_industry_exposure_summary_only.csv", index=False)
    summary = {
        "filings_processed": int(len(filing_frame)),
        "summary_selected": int(filing_frame["extraction_status"].eq("SUMMARY_SELECTED").sum()),
        "fallback_required": int(filing_frame["extraction_status"].eq("FALLBACK_REQUIRED").sum()),
        "summary_selected_by_ticker": filing_frame.loc[
            filing_frame["extraction_status"].eq("SUMMARY_SELECTED")
        ].groupby("ticker").size().to_dict(),
        "fallback_required_by_ticker": filing_frame.loc[
            filing_frame["extraction_status"].eq("FALLBACK_REQUIRED")
        ].groupby("ticker").size().to_dict(),
    }
    (AUDIT / "dynamic_summary_extraction_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
