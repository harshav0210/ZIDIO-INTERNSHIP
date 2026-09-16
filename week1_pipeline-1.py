"""
PROJECT FORESIGHT - WEEK 1 DATA FOUNDATION
NorthBay Living | Zidio Internship

Run from project root:
    python src/pipeline.py

Place the four provided CSV extracts in:
    data/raw/
        sales_daily.csv
        sku_master.csv
        calendar.csv
        inventory_snapshots.csv

This script does not fabricate replacement data.
It validates, cleans, audits, and merges the provided extracts.
"""

from pathlib import Path
import json
import logging
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_FILES = {
    "sales_daily": RAW_DIR / "sales_daily.csv",
    "sku_master": RAW_DIR / "sku_master.csv",
    "calendar": RAW_DIR / "calendar.csv",
    "inventory_snapshots": RAW_DIR / "inventory_snapshots.csv",
}

REQUIRED = {
    "sales_daily": ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"],
    "sku_master": ["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
    "calendar": ["date", "week", "month", "season", "is_holiday", "promo_event"],
    "inventory_snapshots": ["date", "sku_id", "on_hand_units", "on_order_units",
                            "lead_time_days", "reorder_point"],
}

DATE_COLS = {
    "sales_daily": ["date"],
    "sku_master": ["launch_date"],
    "calendar": ["date"],
    "inventory_snapshots": ["date"],
}

NUMERIC_COLS = {
    "sales_daily": ["units_sold", "revenue", "unit_price"],
    "sku_master": ["unit_cost", "list_price"],
    "calendar": [],
    "inventory_snapshots": ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("FORESIGHT-WEEK1")


def snake_case(name):
    name = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def load_and_clean(name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset: {path}\n"
            "Place the four provided project extracts in data/raw/."
        )

    df = pd.read_csv(path)
    rows_before = len(df)
    df.columns = [snake_case(c) for c in df.columns]

    missing_schema = sorted(set(REQUIRED[name]) - set(df.columns))
    if missing_schema:
        raise ValueError(
            f"{name}: missing required columns {missing_schema}. "
            f"Found: {list(df.columns)}"
        )

    duplicate_before = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    invalid_dates = {}
    for col in DATE_COLS[name]:
        before = int(df[col].notna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce")
        invalid_dates[col] = before - int(df[col].notna().sum())

    for col in NUMERIC_COLS[name]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise descriptive labels without changing their meaning.
    for col in ["category", "subcategory", "season", "promo_event"]:
        if col in df.columns:
            df[col] = (
                df[col].astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
            )

    # Standardise recognised boolean values. Unknown values become missing
    # and are reported instead of being guessed.
    for col in ["promo_flag", "is_holiday"]:
        if col in df.columns:
            mapping = {
                "true": True, "false": False, "yes": True, "no": False,
                "y": True, "n": False, "1": True, "0": False
            }
            df[col] = (
                df[col].astype("string").str.strip().str.lower().map(mapping)
                .astype("boolean")
            )

    suspicious = {}
    for col in NUMERIC_COLS[name]:
        if col in df.columns:
            suspicious[col] = int((df[col] < 0).sum())

    notes = []
    if duplicate_before:
        notes.append(f"Removed {duplicate_before} exact duplicate row(s).")
    if any(invalid_dates.values()):
        notes.append(f"Invalid date values converted to missing: {invalid_dates}.")
    if any(suspicious.values()):
        notes.append(f"Suspicious negative numeric values detected: {suspicious}.")
    notes.append(
        "Missing demand/inventory/price values were not blindly imputed; "
        "unknown observations must not be treated as zero demand."
    )

    audit = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns": len(df.columns),
        "duplicate_rows_before": duplicate_before,
        "duplicate_rows_after": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "invalid_dates": invalid_dates,
        "suspicious_negative_values": suspicious,
        "notes": notes,
    }
    return df, audit


def validate_relationships(sales, master, calendar, inventory):
    master_skus = set(master["sku_id"].dropna().astype(str))
    sales_skus = set(sales["sku_id"].dropna().astype(str))
    inventory_skus = set(inventory["sku_id"].dropna().astype(str))

    calendar_dates = set(calendar["date"].dropna())
    sales_dates = set(sales["date"].dropna())
    inventory_dates = set(inventory["date"].dropna())

    return {
        "sales_skus_not_in_master": len(sales_skus - master_skus),
        "inventory_skus_not_in_master": len(inventory_skus - master_skus),
        "sales_dates_not_in_calendar": len(sales_dates - calendar_dates),
        "inventory_dates_not_in_calendar": len(inventory_dates - calendar_dates),
    }


def build_analysis_ready(sales, master, calendar, inventory):
    # Left joins retain sales observations so unmatched master/calendar data
    # remains visible for quality review.
    df = sales.merge(master, on="sku_id", how="left", validate="many_to_one")
    df = df.merge(calendar, on="date", how="left", validate="many_to_one")
    df = df.merge(
        inventory, on=["date", "sku_id"], how="left", validate="many_to_one",
        suffixes=("", "_inventory")
    )

    df["quality_missing_sku_master"] = df["category"].isna()
    df["quality_missing_calendar"] = df["season"].isna()

    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


def pct(x, total):
    return round(100 * x / total, 2) if total else 0.0


def make_report(cleaned, audits, relationships, analysis_ready):
    lines = [
        "# Project FORESIGHT — Week 1 Data Quality Report",
        "",
        "**Client:** NorthBay Living  ",
        "**Program:** Zidio Internship  ",
        "**Stage:** Week 1 — Data Foundation",
        "",
        "## 1. Dataset Summary",
        "",
        "| Dataset | Rows Before | Rows After | Columns | Duplicates | Missing Cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, audit in audits.items():
        lines.append(
            f"| {name} | {audit['rows_before']:,} | {audit['rows_after']:,} | "
            f"{audit['columns']} | {audit['duplicate_rows_before']:,} | "
            f"{audit['missing_cells']:,} |"
        )

    lines += [
        "",
        "## 2. Analysis-Ready Dataset",
        "",
        f"- Rows: **{len(analysis_ready):,}**",
        f"- Columns: **{len(analysis_ready.columns):,}**",
        f"- Unique SKUs: **{analysis_ready['sku_id'].nunique():,}**",
        f"- Date range: **{analysis_ready['date'].min()} to {analysis_ready['date'].max()}**",
        f"- Categories: **{analysis_ready['category'].nunique():,}**",
        "",
        "## 3. Relationship Validation",
        "",
    ]

    for key, value in relationships.items():
        lines.append(f"- `{key}`: **{value:,}**")

    lines += ["", "## 4. Cleaning Decisions", ""]
    for name, audit in audits.items():
        lines.append(f"### {name}")
        for note in audit["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    lines += [
        "## 5. Required Week 1 Outputs",
        "",
        "- Cleaned versions of all four provided extracts",
        "- `analysis_ready.csv`",
        "- `data_quality_summary.json`",
        "- This data-quality report",
        "",
        "## 6. Next Step",
        "",
        "Use the processed data in Week 2 for business-focused EDA: demand trends, "
        "top movers, dead stock, seasonality, promotions, category effects, "
        "inventory position, lead time, and reorder points.",
    ]
    return "\n".join(lines)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = {}
    audits = {}

    for name, path in RAW_FILES.items():
        log.info("Loading %s", name)
        cleaned[name], audits[name] = load_and_clean(name, path)

    relationships = validate_relationships(
        cleaned["sales_daily"],
        cleaned["sku_master"],
        cleaned["calendar"],
        cleaned["inventory_snapshots"],
    )

    log.info("Building analysis-ready dataset")
    analysis_ready = build_analysis_ready(
        cleaned["sales_daily"],
        cleaned["sku_master"],
        cleaned["calendar"],
        cleaned["inventory_snapshots"],
    )

    for name, df in cleaned.items():
        df.to_csv(PROCESSED_DIR / f"{name}_clean.csv", index=False)

    analysis_ready.to_csv(PROCESSED_DIR / "analysis_ready.csv", index=False)

    summary = {
        "datasets": audits,
        "relationships": relationships,
        "analysis_ready_rows": len(analysis_ready),
        "analysis_ready_columns": len(analysis_ready.columns),
    }
    (PROCESSED_DIR / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    (REPORT_DIR / "data_quality_report.md").write_text(
        make_report(cleaned, audits, relationships, analysis_ready),
        encoding="utf-8"
    )

    log.info("Week 1 pipeline completed successfully.")
    log.info("Analysis-ready data: %s rows x %s columns",
             len(analysis_ready), len(analysis_ready.columns))


if __name__ == "__main__":
    main()
