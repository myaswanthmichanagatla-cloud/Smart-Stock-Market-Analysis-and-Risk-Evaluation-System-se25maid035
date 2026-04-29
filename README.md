"""
Stock-Market-RAG--SE25MAID035
=============================
SE25MAID035 · Smart Stock Market Analysis and Investment Risk Assistant
Milestone 1 · Preliminary Model

A user enters a natural-language financial query
(e.g., "safe long-term technology stocks with stable returns"),
and the system retrieves semantically relevant stocks with
market context, sector, risk profile, and financial indicators.

This project is adapted from Art-Style-RAG architecture:
Retrieve — embed stock/market metadata + financial query
Generate (Final Milestone) — grounded investment explanation using LLM

----------------------------------------------------------------------
PROJECT STRUCTURE
.
├── stock_data_pipeline.py         # Load, preprocess & save stock metadata
├── stock_retrieval.py             # Build financial index, query, ablation
├── requirements.txt               # Dependencies
├── data/
│   ├── stock_market_dataset.json  # Output of stock_data_pipeline.py
│   └── stock_index.pkl            # Output of stock_retrieval.py --mode index
└── outputs.txt                    # Sample run logs
----------------------------------------------------------------------

Quickstart:
    # Step 1: Build stock dataset
    python stock_data_pipeline.py --input data/stock_data.csv --output data/stock_market_dataset.json

    # Step 2: Build stock index
    python stock_retrieval.py --mode index --data data/stock_market_dataset.json

    # Step 3: Query stocks
    python stock_retrieval.py --mode query --query "low risk banking stocks"

    # Step 4: Run ablation
    python stock_retrieval.py --mode ablation

Requirements:
    pip install pandas numpy sentence-transformers scikit-learn
"""

import os
import json
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

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
ROLL_NO = "SE25MAID035"
PROJECT_TITLE = "Smart Stock Market Analysis and Investment Risk Assistant"
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/stock_index.pkl"


# ─── Model Loader ───────────────────────────────────────────────────────────
def load_financial_model():
    """
    Load semantic embedding model for stock retrieval.
    """
    log.info(f"[{ROLL_NO}] Loading financial embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded successfully.")
    return model


# ─── Financial Text Builder ────────────────────────────────────────────────
def build_stock_description(row):
    """
    Convert structured stock record into semantic text.
    """
    return (
        f"{row['Sector']} company {row['Company']} "
        f"with PE ratio {row['PE_Ratio']}, "
        f"daily return {row['Daily_Return']} percent, "
        f"volatility {row['Volatility']}, "
        f"risk level {row['Risk_Level']}. "
        f"News sentiment: {row['News']}"
    )


# ─── Embedding Helpers ─────────────────────────────────────────────────────
def embed_stock(model, row):
    """
    Return normalized stock embedding vector.
    """
    return model.encode(
        build_stock_description(row),
        normalize_embeddings=True
    )


def embed_query(model, query):
    """
    Return normalized user query embedding.
    """
    return model.encode(query, normalize_embeddings=True)


# ─── Semantic Search ───────────────────────────────────────────────────────
def cosine_search(query_vec, index_vecs, top_k=5):
    """
    Brute-force cosine similarity retrieval.
    """
    scores = index_vecs @ query_vec
    top_ids = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_ids]


# ─── Risk Classification ───────────────────────────────────────────────────
def classify_risk(volatility, pe_ratio):
    """
    Investment risk heuristic.
    """
    if volatility > 0.4 or pe_ratio > 40:
        return "High Risk"
    elif volatility > 0.2 or pe_ratio > 25:
        return "Moderate Risk"
    return "Low Risk"


# ─── Index Builder ─────────────────────────────────────────────────────────
def build_index(data_path, index_path=INDEX_PATH):
    """
    Build stock semantic retrieval index:
      - embeddings matrix
      - metadata
    """
    if not Path(data_path).exists():
        log.error(f"Dataset not found → {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    model = load_financial_model()

    embeddings = []
    metadata = []

    log.info(f"[{ROLL_NO}] Embedding {len(records)} stock records...")
    t0 = time.time()

    for i, row in enumerate(records):
        try:
            vec = embed_stock(model, row)
            embeddings.append(vec)
            metadata.append(row)

            if (i + 1) % 100 == 0:
                log.info(f"  Embedded {i+1}/{len(records)}")

        except Exception as e:
            log.warning(f"Skipped record {i}: {e}")

    emb_matrix = np.array(embeddings, dtype=np.float32)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": emb_matrix,
                "metadata": metadata,
                "roll_no": ROLL_NO,
                "project_title": PROJECT_TITLE,
            },
            f,
        )

    log.info(f"Index saved → {index_path}")
    log.info(f"Shape: {emb_matrix.shape}")
    log.info(f"Time: {time.time() - t0:.1f}s")


# ─── Query Runner ──────────────────────────────────────────────────────────
def query_stock(query_text, top_k=5, index_path=INDEX_PATH):
    """
    Query semantically similar stocks using natural language.
    """
    if not Path(index_path).exists():
        log.error("Index not found. Run --mode index first.")
        return

    model = load_financial_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]
    metadata = index_data["metadata"]

    query_vec = embed_query(model, query_text)
    results = cosine_search(query_vec, emb_matrix, top_k=top_k)

    print("\n" + "=" * 130)
    print(f"{PROJECT_TITLE} | {ROLL_NO}")
    print(f"Query: {query_text}")
    print("=" * 130)

    print(
        f"{'Rank':<6} {'Score':<8} {'Company':<20} {'Sector':<15} "
        f"{'Close':<12} {'Risk':<15}"
    )

    print("-" * 130)

    for rank, (idx, score) in enumerate(results, 1):
        stock = metadata[idx]

        print(
            f"{rank:<6} {score:<8.4f} "
            f"{stock['Company']:<20} "
            f"{stock['Sector']:<15} "
            f"{stock['Close']:<12} "
            f"{stock['Risk_Level']:<15}"
        )

        print(f"       └─ {stock['text_description']}")

    print("\nTop Investment Insight:")
    best = metadata[results[0][0]]

    print(
        f"{best['Company']} is the most relevant match for this query "
        f"with {best['Risk_Level']} profile, PE ratio {best['PE_Ratio']}, "
        f"and daily return {best['Daily_Return']:.2f}%."
    )


# ─── Baseline Forecast ─────────────────────────────────────────────────────
def baseline_forecast(data_path):
    """
    RandomForest baseline for closing price prediction.
    """
    df = pd.read_json(data_path)

    features = ["Open", "Volume", "PE_Ratio", "Volatility"]
    X = df[features]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return rmse


# ─── Ablation Study ────────────────────────────────────────────────────────
def run_ablation(data_path, index_path=INDEX_PATH):
    """
    Compare:
      A) Baseline price prediction
      B) Semantic stock retrieval
    """
    print("\n" + "=" * 90)
    print(f"ABLATION STUDY — {PROJECT_TITLE} ({ROLL_NO})")
    print("=" * 90)

    rmse = baseline_forecast(data_path)

    print(f"Baseline RandomForest RMSE: {rmse:.4f}")

    if not Path(index_path).exists():
        print("Stock index not found. Build index first.")
        return

    sample_queries = [
        "safe long term IT stocks",
        "high growth energy companies",
        "low volatility banking sector",
    ]

    model = load_financial_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]

    print("\nSemantic Retrieval Performance:")

    for q in sample_queries:
        query_vec = embed_query(model, q)
        results = cosine_search(query_vec, emb_matrix, top_k=1)

        print(
            f"Query: {q:<35} Top Similarity Score: {results[0][1]:.4f}"
        )

    print("\nConclusion:")
    print(
        "Baseline predicts stock price numerically, "
        "while Financial RAG retrieves investment-relevant stocks "
        "using semantic market intelligence."
    )


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_TITLE} — {ROLL_NO}"
    )

    parser.add_argument(
        "--mode",
        choices=["index", "query", "ablation"],
        default="query"
    )

    parser.add_argument(
        "--data",
        default="data/stock_market_dataset.json",
        help="Processed stock dataset path"
    )

    parser.add_argument(
        "--index",
        default=INDEX_PATH,
        help="Index save/load path"
    )

    parser.add_argument(
        "--query",
        default="safe long term growth stocks"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.mode == "index":
        build_index(args.data, args.index)

    elif args.mode == "query":
        query_stock(args.query, args.topk, args.index)

    elif args.mode == "ablation":
        run_ablation(args.data, args.index)
