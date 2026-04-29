"""
smart_stock_market_rag.py
=========================
Smart Stock Market Analysis and Risk Evaluation System — Milestone 1 Preliminary Model
Financial RAG-style retrieval + risk analysis + baseline forecasting on stock market dataset.

Inspired by the architecture of clip_retrieval.py:
    1. Build market index from stock dataset
    2. Query with natural language
    3. Ablation: baseline prediction vs semantic retrieval

Usage:
    # Step 1: Build market index
    python smart_stock_market_rag.py --mode index --data "/Users/satyam/Desktop/yaswanth/MW-NIFTY-50-29-Apr-2026.csv"

    # Step 2: Query market insights
    python smart_stock_market_rag.py --mode query --query "Safe long term IT stocks with strong growth"

    # Step 3: Ablation study
    python smart_stock_market_rag.py --mode ablation --data "/Users/satyam/Desktop/yaswanth/MW-NIFTY-50-29-Apr-2026.csv"

Requirements:
    pip install pandas numpy sentence-transformers scikit-learn
"""

import os
import argparse
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/market_index.pkl"

# ─── NIFTY-50 Column Mapping ──────────────────────────────────────────────────
# Maps internal standard names → possible real CSV column names (case-insensitive)
# Adjust right-hand values if your CSV has different headers
COLUMN_MAP = {
    "Company":    ["symbol", "company", "name", "ticker", "scrip name", "security name"],
    "Sector":     ["sector", "industry", "series"],
    "Open":       ["open", "open price", "prev. close"],
    "Close":      ["close", "ltp", "last price", "closing price", "last traded price"],
    "Volume":     ["volume", "volume (shares)", "total traded volume", "ttq", "shares traded"],
    "PE_Ratio":   ["pe ratio", "p/e", "pe", "price/earnings"],
    "Volatility": ["volatility", "52w h/l", "day range", "52w high/low"],
    "News":       ["news", "remarks", "notes", "comment"],
}


def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect CSV columns by matching against COLUMN_MAP.
    Returns a dict: standard_name -> actual_col_name (or None if not found).

    FIX: Original code assumed fixed column names — this handles real-world
         NIFTY-50 CSVs that have different/varying headers.
    """
    actual_cols_lower = {c.lower().strip(): c for c in df.columns}
    resolved = {}
    for std_name, candidates in COLUMN_MAP.items():
        found = None
        for candidate in candidates:
            if candidate.lower() in actual_cols_lower:
                found = actual_cols_lower[candidate.lower()]
                break
        resolved[std_name] = found
        if found is None:
            log.warning(f"Column '{std_name}' not found in CSV. Will use default/fallback.")
    return resolved


def get_val(row, col_name, default="N/A"):
    """
    Safely get a value from row using resolved column name.

    FIX: Original code used row['ColumnName'] directly which raises KeyError
         if column is missing or renamed.
    """
    if col_name and col_name in row.index:
        val = row[col_name]
        # Return default for NaN values
        if pd.isna(val):
            return default
        return val
    return default


# ─── Model Loader ─────────────────────────────────────────────────────────────
def load_financial_model():
    """Load sentence transformer model for financial semantic retrieval."""
    log.info(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded successfully.")
    return model


# ─── Embedding Helpers ────────────────────────────────────────────────────────
def stock_to_text(row, col_map: dict) -> str:
    """
    Convert structured stock row into semantic text.

    FIX: Original used hardcoded column names (row['Company'], row['Sector'], etc.)
         which crashes on NIFTY-50 CSV. Now uses resolved column mapping with
         safe fallbacks for missing fields.
    """
    company    = get_val(row, col_map.get("Company"),    default="Unknown Company")
    sector     = get_val(row, col_map.get("Sector"),     default="Unknown Sector")
    open_p     = get_val(row, col_map.get("Open"),       default="N/A")
    close_p    = get_val(row, col_map.get("Close"),      default="N/A")
    volume     = get_val(row, col_map.get("Volume"),     default="N/A")
    pe_ratio   = get_val(row, col_map.get("PE_Ratio"),   default="N/A")
    volatility = get_val(row, col_map.get("Volatility"), default="N/A")
    news       = get_val(row, col_map.get("News"),       default="No news available")

    return (
        f"Company {company} in sector {sector}. "
        f"Open price {open_p}, close price {close_p}, "
        f"volume {volume}, PE ratio {pe_ratio}, "
        f"volatility {volatility}. News sentiment: {news}"
    )


def embed_stock(model, row, col_map: dict):
    """Embed a single stock row into a normalized vector."""
    text = stock_to_text(row, col_map)
    return model.encode(text, normalize_embeddings=True)


def embed_query(model, query: str):
    """Embed a user query string into a normalized vector."""
    return model.encode(query, normalize_embeddings=True)


# ─── Semantic Search ──────────────────────────────────────────────────────────
def cosine_search(query_vec: np.ndarray, index_vecs: np.ndarray, top_k: int = 5):
    """
    Compute cosine similarity (dot product on normalized vectors) and
    return top-k (index, score) pairs.

    FIX: Added guard for top_k > available records to avoid index errors.
    """
    top_k = min(top_k, len(index_vecs))  # FIX: Guard against requesting more than available
    scores = index_vecs @ query_vec
    top_ids = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_ids]


# ─── Risk Assessment ──────────────────────────────────────────────────────────
def risk_assessment(volatility, pe_ratio) -> str:
    """
    Simple heuristic risk classification.

    FIX: Original crashed on "N/A" or non-numeric values from missing columns.
         Now safely converts to float with fallback.
    """
    try:
        vol = float(str(volatility).replace(",", "").split("-")[0].strip())
    except (ValueError, TypeError):
        vol = 0.0  # Default: treat unknown volatility as low

    try:
        pe = float(str(pe_ratio).replace(",", "").strip())
    except (ValueError, TypeError):
        pe = 0.0  # Default: treat unknown PE as low

    if vol > 0.4 or pe > 40:
        return "High Risk"
    elif vol > 0.2 or pe > 25:
        return "Moderate Risk"
    return "Low Risk"


# ─── Index Builder ────────────────────────────────────────────────────────────
def build_index(data_path: str, index_path: str = INDEX_PATH):
    """
    Build stock market embedding index:
      - embeddings matrix (float32)
      - metadata records (list of dicts)
      - column mapping (for consistent decoding later)

    FIX 1: Original used `i` from iterrows() for logging count — iterrows()
            index `i` is the DataFrame index (not sequential row count), so
            logging was incorrect. Fixed with separate counter `count`.

    FIX 2: col_map is now saved inside the index so query mode uses the
            same mapping — original code didn't persist this, causing
            KeyErrors at query time.
    """
    log.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Show detected columns for user awareness
    log.info(f"Detected columns: {list(df.columns)}")

    # Resolve column mapping once for the whole dataset
    col_map = resolve_columns(df)
    log.info(f"Column mapping resolved: {col_map}")

    model = load_financial_model()

    embeddings = []
    metadata = []

    total = len(df)
    log.info(f"Embedding {total} stock records...")
    t0 = time.time()

    count = 0  # FIX: Separate counter instead of relying on iterrows() index
    for i, row in df.iterrows():
        try:
            vec = embed_stock(model, row, col_map)
            embeddings.append(vec)
            metadata.append(row.to_dict())
            count += 1

            # FIX: Use `count` (sequential) not `i` (df index) for logging
            if count % 10 == 0 or count == total:
                log.info(f"  Embedded {count}/{total}")

        except Exception as e:
            log.warning(f"Skipped row index {i}: {e}")

    if not embeddings:
        log.error("No records were embedded. Check your CSV format.")
        return

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Save index with col_map included — FIX: original didn't save col_map
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": emb_matrix,
                "metadata":   metadata,
                "col_map":    col_map,     # FIX: persist mapping for query mode
            },
            f,
        )

    log.info(f"Index saved → {index_path}")
    log.info(f"Embedding matrix shape: {emb_matrix.shape}")
    log.info(f"Time elapsed: {time.time() - t0:.1f}s")


# ─── Query Runner ─────────────────────────────────────────────────────────────
def query_market(query_text: str, top_k: int = 5, index_path: str = INDEX_PATH):
    """
    Retrieve top-k relevant stocks for a user financial query.

    FIX: Now loads col_map from index (saved during build_index) so that
         metadata display uses correct column names consistently.
    """
    if not Path(index_path).exists():
        log.error(f"Index not found at '{index_path}'. Run --mode index first.")
        return

    model = load_financial_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]
    metadata   = index_data["metadata"]
    # FIX: Load col_map from index — original had no col_map, causing KeyErrors
    col_map    = index_data.get("col_map", {})

    query_vec = embed_query(model, query_text)
    results   = cosine_search(query_vec, emb_matrix, top_k=top_k)

    # Resolve display column names using col_map
    company_col    = col_map.get("Company")
    sector_col     = col_map.get("Sector")
    close_col      = col_map.get("Close")
    volatility_col = col_map.get("Volatility")
    pe_col         = col_map.get("PE_Ratio")
    news_col       = col_map.get("News")

    print("\n" + "=" * 120)
    print(f"Query: {query_text}")
    print("=" * 120)
    print(f"{'Rank':<6} {'Score':<8} {'Company':<25} {'Sector':<20} {'Close':<14} {'Risk':<15}")
    print("-" * 120)

    for rank, (idx, score) in enumerate(results, 1):
        stock = metadata[idx]

        # FIX: Use safe get_val with col_map instead of direct dict access
        company    = stock.get(company_col,    "N/A") if company_col    else "N/A"
        sector     = stock.get(sector_col,     "N/A") if sector_col     else "N/A"
        close      = stock.get(close_col,      "N/A") if close_col      else "N/A"
        volatility = stock.get(volatility_col, "N/A") if volatility_col else "N/A"
        pe_ratio   = stock.get(pe_col,         "N/A") if pe_col         else "N/A"
        news       = stock.get(news_col,       "N/A") if news_col       else "N/A"

        risk = risk_assessment(volatility, pe_ratio)

        print(
            f"{rank:<6} {score:<8.4f} {str(company):<25} {str(sector):<20} "
            f"{str(close):<14} {risk:<15}"
        )
        print(f"       News: {news}")

    # Investment insight for top result
    if results:
        best_idx   = results[0][0]
        best       = metadata[best_idx]
        best_co    = best.get(company_col,    "N/A") if company_col    else "N/A"
        best_close = best.get(close_col,      "N/A") if close_col      else "N/A"
        best_vol   = best.get(volatility_col, "N/A") if volatility_col else "N/A"
        best_pe    = best.get(pe_col,         "N/A") if pe_col         else "N/A"

        print("\nInvestment Insight Summary:")
        print(
            f"{best_co} appears most relevant for this query based on semantic similarity, "
            f"with a closing price of {best_close} and "
            f"{risk_assessment(best_vol, best_pe)} profile."
        )


# ─── Baseline Forecasting ─────────────────────────────────────────────────────
def baseline_forecast(data_path: str) -> float:
    """
    Random Forest baseline for stock closing price prediction.

    FIX 1: Original assumed fixed column names ['Open', 'Volume', 'PE_Ratio',
            'Volatility', 'Close'] which don't exist in NIFTY-50 CSV.
            Now uses resolve_columns() and falls back gracefully.

    FIX 2: Added numeric coercion — NIFTY-50 columns often have commas in
            numbers (e.g., "1,234.56") which must be cleaned before modeling.

    FIX 3: Added minimum sample guard to avoid train/test split failures
            on tiny datasets.
    """
    log.info(f"Loading data for baseline forecast: {data_path}")
    df = pd.read_csv(data_path)

    col_map = resolve_columns(df)

    # Build feature column list from resolved names
    feature_keys = ["Open", "Volume", "PE_Ratio", "Volatility"]
    target_key   = "Close"

    feature_cols = [col_map[k] for k in feature_keys if col_map.get(k)]
    target_col   = col_map.get(target_key)

    # FIX: Validate that we have enough columns to proceed
    if not target_col:
        log.error(
            f"Target column 'Close' could not be resolved. "
            f"Available columns: {list(df.columns)}"
        )
        return float("nan")

    if not feature_cols:
        log.error("No feature columns could be resolved. Cannot run baseline forecast.")
        return float("nan")

    log.info(f"Using features: {feature_cols}")
    log.info(f"Using target:   {target_col}")

    # FIX: Clean numeric strings (remove commas, strip whitespace)
    for col in feature_cols + [target_col]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in relevant columns
    df_clean = df[feature_cols + [target_col]].dropna()

    # FIX: Guard against too few samples for train/test split
    if len(df_clean) < 10:
        log.error(
            f"Not enough valid rows for forecasting ({len(df_clean)} rows after cleaning). "
            f"Need at least 10."
        )
        return float("nan")

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    preds = rf_model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))

    log.info(f"Baseline RMSE: {rmse:.4f}")
    return rmse


# ─── Ablation Study ───────────────────────────────────────────────────────────
def run_ablation(data_path: str, index_path: str = INDEX_PATH):
    """
    Compare:
      A) Baseline RandomForest numeric price prediction
      B) Semantic Retrieval + Risk-based relevance ranking

    FIX: Original didn't handle missing index gracefully in ablation.
         Added clearer messaging and guarded semantic section.
    """
    print("\n" + "=" * 80)
    print("ABLATION: Baseline Forecasting vs Financial Semantic Retrieval")
    print("=" * 80)

    # ── Part A: Baseline ──────────────────────────────────────────────────────
    print("\n[A] Baseline Random Forest Forecasting")
    rmse = baseline_forecast(data_path)
    if not np.isnan(rmse):
        print(f"    Baseline RandomForest RMSE: {rmse:.4f}")
    else:
        print("    Baseline could not be computed (see warnings above).")

    # ── Part B: Semantic Retrieval ────────────────────────────────────────────
    print("\n[B] Semantic Retrieval Evaluation")
    if not Path(index_path).exists():
        print(f"    Semantic index not found at '{index_path}'. Build index first with --mode index.")
        print("\nConclusion:")
        print("    Baseline predicts price numerically. Build index to enable semantic comparison.")
        return

    model = load_financial_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]

    sample_queries = [
        "low risk technology stocks",
        "high growth banking stocks",
        "stable energy sector investment",
        "NIFTY 50 blue chip safe stocks",
        "high dividend yield defensive stocks",
    ]

    print(f"\n    {'Query':<40} {'Top Similarity Score'}")
    print(f"    {'-'*40} {'-'*20}")
    for q in sample_queries:
        query_vec = embed_query(model, q)
        results   = cosine_search(query_vec, emb_matrix, top_k=1)
        score     = results[0][1] if results else 0.0
        print(f"    {q:<40} {score:.4f}")

    print("\nConclusion:")
    print(
        "  Baseline RandomForest predicts closing price numerically (RMSE-based evaluation).\n"
        "  Semantic RAG retrieval provides contextual investment intelligence by matching\n"
        "  natural language queries to relevant stocks — complementary approaches."
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Stock Market Analysis and Risk Evaluation System"
    )

    parser.add_argument(
        "--mode",
        choices=["index", "query", "ablation"],
        default="query",
        help="Operation mode: build index, run query, or ablation study",
    )

    # FIX: Updated default data path to match actual file location
    parser.add_argument(
        "--data",
        default="/Users/satyam/Desktop/yaswanth/MW-NIFTY-50-29-Apr-2026.csv",
        help="Path to stock market CSV dataset",
    )

    parser.add_argument(
        "--index",
        default=INDEX_PATH,
        help="Path to save/load the embedding index",
    )

    parser.add_argument(
        "--query",
        default="Safe long term growth stocks",
        help="Natural language query for stock retrieval",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top results to retrieve",
    )

    args = parser.parse_args()

    if args.mode == "index":
        build_index(args.data, args.index)

    elif args.mode == "query":
        query_market(args.query, args.topk, args.index)

    elif args.mode == "ablation":
        run_ablation(args.data, args.index)