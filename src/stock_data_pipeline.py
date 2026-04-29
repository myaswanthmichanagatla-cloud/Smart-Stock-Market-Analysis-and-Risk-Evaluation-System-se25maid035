"""
stock_data_pipeline.py
======================
Smart Stock Market Analysis and Risk Evaluation System — Milestone 1 Data Pipeline

Loads stock market CSV dataset, preprocesses structured financial data,
creates technical indicators + risk labels + semantic text descriptions,
and saves a ready-to-use JSON subset for retrieval + GenAI analysis.

Usage:
    python stock_data_pipeline.py --input data/stock_data.csv --output data/stock_market_dataset.json

Requirements:
    pip install pandas numpy
"""

import os
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


# ─── Column Validator ────────────────────────────────────────────────────────

# All columns that the pipeline needs to function
REQUIRED_COLUMNS = ["Open", "Close", "High", "Low", "Volume", "PE_Ratio", "Company", "Sector"]
OPTIONAL_COLUMNS = ["News"]  # Will be filled with placeholder if missing

def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #3 & #7: Validate required columns exist.
    - Raises clear error if critical columns are missing.
    - Adds placeholder for optional columns if missing.
    """
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_required:
        log.error(f"Missing required columns: {missing_required}")
        log.error(f"Available columns in CSV: {list(df.columns)}")
        raise ValueError(
            f"CSV is missing required columns: {missing_required}. "
            f"Please check your dataset or column mapping."
        )

    # Add optional columns with defaults if missing
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            log.warning(f"Optional column '{col}' not found. Filling with 'No news available'.")
            df[col] = "No news available"

    log.info(f"Column validation passed. Columns found: {list(df.columns)}")
    return df


# ─── Helper Functions ────────────────────────────────────────────────────────

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    FIX #8: Clean numeric columns that may contain:
    - Comma-separated numbers: "1,234.56" → 1234.56
    - Percentage signs: "12.5%" → 12.5
    - Whitespace padding
    Then coerce to float, replacing non-parseable values with NaN.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #1: Clean and fill missing values.
    - Numeric columns: clean string formatting first, then fill NaN with median.
    - Text columns: fill NaN with 'Unknown'.

    Original used dtype check that missed newer pandas nullable types (Int64, Float64).
    Now explicitly cleans numeric columns before type-checking.
    """
    # First clean all required numeric columns of string artifacts
    numeric_cols = ["Open", "Close", "High", "Low", "Volume", "PE_Ratio"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    # Now fill missing values
    for col in df.columns:
        # FIX #1: Use pd.api.types.is_numeric_dtype for robust type detection
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info(f"  Filled numeric column '{col}' NaNs with median: {median_val:.4f}")
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def calculate_daily_return(open_price, close_price) -> float:
    """
    Daily Return % = ((Close - Open) / Open) * 100

    FIX #8: Added type safety — converts inputs to float before calculation.
    Returns 0.0 if inputs are invalid or open_price is zero.
    """
    try:
        open_price  = float(open_price)
        close_price = float(close_price)
        if open_price == 0:
            return 0.0
        return round(((close_price - open_price) / open_price) * 100, 4)
    except (TypeError, ValueError):
        return 0.0


def calculate_volatility(high, low, open_price) -> float:
    """
    Simple intraday volatility = (High - Low) / Open

    FIX #8: Added type safety — converts inputs to float before calculation.
    Returns 0.0 if inputs are invalid or open_price is zero.
    """
    try:
        high        = float(high)
        low         = float(low)
        open_price  = float(open_price)
        if open_price == 0:
            return 0.0
        return round((high - low) / open_price, 6)
    except (TypeError, ValueError):
        return 0.0


def classify_risk(volatility, pe_ratio) -> str:
    """
    Risk classification heuristic based on volatility and PE ratio.

    FIX #8: Added type safety for non-numeric inputs.
    """
    try:
        vol = float(volatility)
        pe  = float(pe_ratio)
    except (TypeError, ValueError):
        return "Unknown Risk"

    if vol > 0.4 or pe > 40:
        return "High Risk"
    elif vol > 0.2 or pe > 25:
        return "Moderate Risk"
    return "Low Risk"


def safe_float(value, default=0.0) -> float:
    """
    FIX #5: Safely convert any value to float.
    Returns default if conversion fails.
    Used in record building to prevent KeyError / ValueError crashes.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value, default="Unknown") -> str:
    """
    Safely convert any value to string.
    Returns default if value is NaN or None.
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip() or default


def build_financial_text_description(row) -> str:
    """
    Generate semantic text for retrieval model.

    FIX #2: Original crashed if any field was missing/NaN.
    Now uses safe_str / safe_float with fallback defaults for all fields.

    Example output:
    'Technology stock TCS with Low Risk, PE ratio 24.0,
     positive return, daily return 1.25%, volume 1000000.
     News sentiment: Strong quarterly results.'
    """
    company       = safe_str(row.get("Company"),      default="Unknown Company")
    sector        = safe_str(row.get("Sector"),       default="Unknown Sector")
    risk_level    = safe_str(row.get("Risk_Level"),   default="Unknown Risk")
    news          = safe_str(row.get("News"),         default="No news available")
    daily_return  = safe_float(row.get("Daily_Return"), default=0.0)
    pe_ratio      = safe_float(row.get("PE_Ratio"),     default=0.0)
    volume        = safe_float(row.get("Volume"),       default=0.0)

    trend = "positive return" if daily_return > 0 else "negative return"

    return (
        f"{sector} stock {company} with {risk_level}, "
        f"PE ratio {pe_ratio:.2f}, {trend}, "
        f"daily return {daily_return:.2f}%, "
        f"volume {volume:.0f}. "
        f"News sentiment: {news}"
    )


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str  = "data/stock_data.csv",
    output_path: str = "data/stock_market_dataset.json"
):
    """
    Full stock market data pipeline:
      1. Load stock dataset
      2. Validate columns
      3. Clean missing values
      4. Generate technical indicators
      5. Generate risk labels
      6. Build semantic descriptions
      7. Save JSON metadata
    """

    log.info(f"Loading stock dataset from: {input_path}")

    # ── Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
    log.info(f"Columns detected: {list(df.columns)}")

    # ── Step 1: Validate Columns ──────────────────────────────────────────
    log.info("Step 1: Validating columns...")
    df = validate_columns(df)

    # ── Step 2: Clean Data ────────────────────────────────────────────────
    log.info("Step 2: Cleaning missing values...")
    df = clean_missing_values(df)

    # ── Step 3: Technical Indicators ──────────────────────────────────────
    log.info("Step 3: Generating technical indicators...")

    df["Daily_Return"] = df.apply(
        lambda row: calculate_daily_return(row["Open"], row["Close"]),
        axis=1
    )

    df["Volatility"] = df.apply(
        lambda row: calculate_volatility(row["High"], row["Low"], row["Open"]),
        axis=1
    )

    # Simple Moving Average on Close price
    # FIX #6: Guard SMA_5 creation and usage together — if Close missing,
    # SMA_5 defaults to 0.0 so the records loop never crashes
    if "Close" in df.columns:
        df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        log.info("  SMA_5 (5-period simple moving average) computed.")
    else:
        df["SMA_5"] = 0.0
        log.warning("  'Close' column missing — SMA_5 set to 0.0.")

    # ── Step 4: Risk Classification ───────────────────────────────────────
    log.info("Step 4: Classifying investment risk...")

    df["Risk_Level"] = df.apply(
        lambda row: classify_risk(row["Volatility"], row["PE_Ratio"]),
        axis=1
    )

    # ── Step 5: Semantic Text Descriptions ────────────────────────────────
    log.info("Step 5: Building financial text descriptions...")

    df["text_description"] = df.apply(
        build_financial_text_description,
        axis=1
    )

    # ── Step 6: Build JSON Records ────────────────────────────────────────
    log.info("Step 6: Building JSON records...")

    records = []
    total   = len(df)

    # FIX #4: Use enumerate for sequential counter instead of df index (idx)
    # Original: `if (idx + 1) % 100 == 0` — idx is DataFrame index, not count
    for count, (idx, row) in enumerate(df.iterrows(), start=1):
        try:
            # FIX #5: Use safe_float / safe_str for all fields
            record = {
                "id":               int(idx),
                "Company":          safe_str(row.get("Company")),
                "Sector":           safe_str(row.get("Sector")),
                "Open":             safe_float(row.get("Open")),
                "Close":            safe_float(row.get("Close")),
                "High":             safe_float(row.get("High")),
                "Low":              safe_float(row.get("Low")),
                "Volume":           safe_float(row.get("Volume")),
                "PE_Ratio":         safe_float(row.get("PE_Ratio")),
                "Daily_Return":     safe_float(row.get("Daily_Return")),
                "Volatility":       safe_float(row.get("Volatility")),
                "SMA_5":            safe_float(row.get("SMA_5")),        # FIX #6
                "Risk_Level":       safe_str(row.get("Risk_Level")),
                "News":             safe_str(row.get("News"),   default="No news available"),
                "text_description": safe_str(row.get("text_description")),
            }

            records.append(record)

        except Exception as e:
            # Never crash the whole pipeline on one bad row
            log.warning(f"Skipped row index {idx}: {e}")

        # FIX #4: Use `count` (sequential 1-based) not `idx` (df index)
        if count % 100 == 0 or count == total:
            log.info(f"  Processed {count}/{total} records...")

    # ── Step 7: Save JSON ─────────────────────────────────────────────────
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    log.info(f"Saved processed dataset → {output_file}")

    # ── Summary Statistics ────────────────────────────────────────────────
    risk_counts = df["Risk_Level"].value_counts().to_dict()
    sector_counts = df["Sector"].value_counts().to_dict()

    log.info("\n── Dataset Summary ──────────────────────────────────────────")
    log.info(f"  Total records      : {len(records)}")
    log.info(f"  Unique sectors     : {len(sector_counts)}")
    log.info(f"  Risk distribution  : {risk_counts}")
    log.info(f"  Avg Daily Return   : {df['Daily_Return'].mean():.4f}%")
    log.info(f"  Avg Volatility     : {df['Volatility'].mean():.4f}")
    log.info(f"  Top 5 sectors      : {list(sector_counts.items())[:5]}")
    log.info("─" * 60)


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Market Data Pipeline for Smart Analysis System"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/stock_data.csv",
        help="Input CSV dataset path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/stock_market_dataset.json",
        help="Output JSON dataset path"
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
    )
    