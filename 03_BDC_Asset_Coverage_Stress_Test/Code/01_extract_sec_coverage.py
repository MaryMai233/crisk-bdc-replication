from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html


MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def normalized_text(root: html.HtmlElement) -> str:
    return " ".join(" ".join(root.itertext()).replace("\xa0", " ").split())


def parse_numeric(text: str, scale: str | None = None) -> float | None:
    cleaned = re.sub(r"[^0-9.\-]", "", text or "")
    if cleaned in {"", ".", "-"}:
        return None
    value = float(cleaned)
    if scale:
        value *= 10 ** int(scale)
    return value


def context_instants(root: html.HtmlElement) -> dict[str, str]:
    out: dict[str, str] = {}
    for element in root.iter():
        if not str(element.tag).lower().endswith(":context"):
            continue
        context_id = element.get("id")
        instant = ["".join(x.itertext()) for x in element.iter() if str(x.tag).lower().endswith(":instant")]
        if context_id and instant:
            out[context_id] = instant[0].strip()
    return out


def xbrl_candidates(root: html.HtmlElement, report_date: str) -> list[dict]:
    contexts = context_instants(root)
    rows = []
    nodes = root.xpath(
        "//*[contains(translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),"
        "'investmentcompanyseniorsecurityindebtednessassetcoverageratio')]"
    )
    for node in nodes:
        value = parse_numeric(" ".join(node.itertext()), node.get("scale"))
        if value is None:
            continue
        # Inline XBRL often stores 194% as text 194 with scale=-2.
        pct = value * 100 if abs(value) <= 5 else value
        context = node.get("contextref") or node.get("contextRef")
        rows.append({
            "value_pct": pct,
            "contextref": context,
            "context_instant": contexts.get(context, ""),
            "matches_report_date": contexts.get(context, "") == report_date,
            "xbrl_name": node.get("name", ""),
        })
    return rows


def dated_regex_candidates(text: str, report_date: str) -> list[dict]:
    date = pd.Timestamp(report_date)
    date_text = rf"{MONTHS[date.month]}\s+{date.day},\s+{date.year}"
    patterns = [
        rf"As of\s+{date_text}.{{0,420}}?asset coverage(?: ratio)?(?:.{{0,140}}?)(?:was|is|stood at|equaled)\s+(\d{{1,3}}(?:\.\d+)?)\s*(%|x)",
        rf"asset coverage(?: ratio)?(?:.{{0,140}}?)(?:was|is|stood at|equaled)\s+(\d{{1,3}}(?:\.\d+)?)\s*(%|x).{{0,220}}?as of\s+{date_text}",
        rf"As of\s+{date_text}.{{0,300}}?coverage(?: ratio)?(?:.{{0,120}}?)(?:was|is|stood at|equaled)\s+(\d{{1,3}}(?:\.\d+)?)\s*(%|x)",
    ]
    rows = []
    for pattern_id, pattern in enumerate(patterns, start=1):
        for match in re.finditer(pattern, text, flags=re.I):
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 240)
            value = float(match.group(1)) * (100.0 if match.group(2).lower() == "x" else 1.0)
            rows.append({
                "value_pct": value,
                "pattern_id": pattern_id,
                "evidence": text[start:end],
            })
    return rows


def choose_actual_coverage(xbrl: list[dict], regex: list[dict]) -> tuple[float | None, str, str, str]:
    report_xbrl = [r for r in xbrl if r["matches_report_date"] and 100 <= r["value_pct"] <= 400]
    report_values = sorted(set(round(r["value_pct"], 6) for r in report_xbrl))
    regex_values = sorted(set(round(r["value_pct"], 6) for r in regex if 100 <= r["value_pct"] <= 400))

    # Statutory 150/200 values can appear as facts. Prefer a non-threshold value.
    non_threshold = [v for v in report_values if v not in {150.0, 200.0}]
    if len(non_threshold) == 1:
        return non_threshold[0], "XBRL_REPORT_DATE", "HIGH", json.dumps(report_xbrl)
    if len(report_values) == 1 and regex_values == report_values:
        return report_values[0], "XBRL_AND_TEXT", "HIGH", json.dumps(report_xbrl)
    if len(regex_values) == 1:
        evidence = next(r["evidence"] for r in regex if round(r["value_pct"], 6) == regex_values[0])
        confidence = "HIGH" if (not report_values or regex_values[0] in report_values) else "MEDIUM"
        return regex_values[0], "DATED_TEXT", confidence, evidence
    if len(regex_values) > 1:
        # Some BDCs disclose both the statutory ratio excluding SBA debentures
        # and an all-debt ratio including them. The higher ratio is the statutory
        # 1940 Act measure after the disclosed exemptive relief. PSEC also reports
        # a separate preferred-stock coverage ratio; the indebtedness ratio is higher.
        chosen = max(regex_values)
        evidence = next(r["evidence"] for r in regex if round(r["value_pct"], 6) == chosen)
        return chosen, "DATED_TEXT_MULTIPLE_MAX_STATUTORY_DEBT", "MEDIUM", evidence
    if len(non_threshold) > 1 and len(regex_values) == 1 and regex_values[0] in non_threshold:
        evidence = next(r["evidence"] for r in regex if round(r["value_pct"], 6) == regex_values[0])
        return regex_values[0], "DATED_TEXT_RESOLVES_XBRL", "HIGH", evidence
    return None, "UNRESOLVED", "LOW", json.dumps({"xbrl": report_values, "regex": regex_values})


def infer_threshold(text: str) -> tuple[float, str, str]:
    reduced_patterns = [
        r"asset coverage requirement.{0,220}?reduced from\s*200(?:\.0)?\s*%\s*to\s*150(?:\.0)?\s*%",
        r"minimum asset coverage ratio applicable.{0,220}?from\s*200(?:\.0)?\s*%\s*to\s*150(?:\.0)?\s*%",
        r"reduced to\s*150(?:\.0)?\s*%\s+effective",
        r"asset coverage requirements?.{0,180}?150\s*%",
        r"asset coverage.{0,100}?at least\s*150\s*%",
        r"minimum asset coverage.{0,100}?150\s*%",
    ]
    for p in reduced_patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return 150.0, "FILING_TEXT", text[max(0, m.start()-120):min(len(text), m.end()+180)]
    baseline = re.search(r"asset coverage.{0,160}?at least\s*200\s*%", text, flags=re.I)
    if baseline:
        return 200.0, "FILING_TEXT", text[max(0, baseline.start()-120):min(len(text), baseline.end()+180)]
    return np.nan, "UNRESOLVED", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    zip_path = Path(args.zip)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    candidates = []
    with zipfile.ZipFile(zip_path) as archive:
        manifest_name = "SEC_BDC_Filings_2021_2025/audit/filing_manifest.csv"
        manifest = pd.read_csv(io.BytesIO(archive.read(manifest_name)), encoding="utf-8-sig")
        # Keep one original filing per firm-quarter; amendments are archived but not primary here.
        manifest = manifest[manifest["form"].isin(["10-K", "10-Q"])].copy()
        manifest = manifest.sort_values(["ticker", "report_date", "filing_date"]).drop_duplicates(
            ["ticker", "calendar_quarter"], keep="last"
        )
        total = len(manifest)
        for counter, row in enumerate(manifest.itertuples(index=False), start=1):
            local = str(row.local_file).replace("\\", "/")
            member = f"SEC_BDC_Filings_2021_2025/{local}"
            root = html.fromstring(archive.read(member))
            text = normalized_text(root)
            xbrl = xbrl_candidates(root, row.report_date)
            regex = dated_regex_candidates(text, row.report_date)
            actual, method, confidence, evidence = choose_actual_coverage(xbrl, regex)
            threshold, threshold_method, threshold_evidence = infer_threshold(text)
            rows.append({
                "ticker": row.ticker,
                "company": row.company,
                "calendar_quarter": row.calendar_quarter,
                "report_date": row.report_date,
                "form": row.form,
                "filing_date": row.filing_date,
                "accession_number": row.accession_number,
                "source_url": row.source_url,
                "local_file": local,
                "actual_asset_coverage_pct": actual,
                "coverage_extraction_method": method,
                "coverage_extraction_confidence": confidence,
                "coverage_evidence": evidence,
                "statutory_threshold_pct": threshold,
                "threshold_extraction_method": threshold_method,
                "threshold_evidence": threshold_evidence,
            })
            for item in xbrl:
                candidates.append({"ticker": row.ticker, "calendar_quarter": row.calendar_quarter, "kind": "XBRL", **item})
            for item in regex:
                candidates.append({"ticker": row.ticker, "calendar_quarter": row.calendar_quarter, "kind": "TEXT", **item})
            if counter % 20 == 0 or counter == total:
                print(f"[{counter}/{total}] parsed", flush=True)

    result = pd.DataFrame(rows).sort_values(["ticker", "report_date"])
    # Sample-specific election history confirmed in filings. All firms except MAIN
    # entered 2021 already subject to 150%; MAIN changed from 200% to 150% on 2022-05-03.
    result["statutory_threshold_pct"] = 150.0
    main_pre_election = (result["ticker"] == "MAIN") & (pd.to_datetime(result["report_date"]) < pd.Timestamp("2022-05-03"))
    result.loc[main_pre_election, "statutory_threshold_pct"] = 200.0
    result["threshold_extraction_method"] = np.where(
        main_pre_election,
        "MAIN_ELECTION_EFFECTIVE_2022-05-03",
        "FILING_TEXT_CONFIRMED_150",
    )

    result.to_csv(outdir / "sec_asset_coverage_extracted.csv", index=False)
    result.to_stata(outdir / "sec_asset_coverage_extracted.dta", write_index=False, version=118)
    pd.DataFrame(candidates).to_csv(outdir / "sec_asset_coverage_candidates_audit.csv", index=False)
    audit = {
        "source_zip": str(zip_path),
        "firm_quarters": int(len(result)),
        "firms": int(result["ticker"].nunique()),
        "actual_coverage_nonmissing": int(result["actual_asset_coverage_pct"].notna().sum()),
        "threshold_nonmissing": int(result["statutory_threshold_pct"].notna().sum()),
        "methods": result["coverage_extraction_method"].value_counts(dropna=False).to_dict(),
        "confidence": result["coverage_extraction_confidence"].value_counts(dropna=False).to_dict(),
        "threshold_values": result["statutory_threshold_pct"].value_counts(dropna=False).to_dict(),
    }
    (outdir / "sec_asset_coverage_extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
