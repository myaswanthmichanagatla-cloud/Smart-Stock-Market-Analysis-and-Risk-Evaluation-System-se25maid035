# Stock-Market-RAG--SE25MAID035,se25maid014,se25maid016,se25maid021
A user enters or uploads a stock market query (e.g., "safe long-term banking stocks with stable returns") and the system retrieves financially and semantically relevant stocks with context about sector, volatility, company fundamentals, and investment risk.


> **Milestone 1 · Preliminary Model**  
> Retrieve stocks from structured financial datasets by natural-language investment descriptions using semantic embeddings + Financial RAG.

---

## Overview

Finding good investment opportunities is difficult with keyword search alone. A query like *"low-volatility dividend-paying technology companies with strong long-term upside"* cannot be meaningfully answered by a standard spreadsheet or basic stock screener.

This project builds a **Stock Market RAG** system:

1. **Retrieve** — embed a free-text financial query with sentence-transformer semantic embeddings and find financially relevant stocks from a structured market dataset.
2. **Generate** *(final milestone)* — produce a grounded investment recommendation for each retrieved stock using an LLM, with every recommendation traceable to real stock fundamentals and market metadata.

---

## Project Structure

```bash
.
├── stock_data_pipeline.py        # Load, preprocess & save stock metadata
├── stock_retrieval.py            # Build stock index, query, and run ablation
├── requirements.txt              # Python dependencies
├── data/
│   ├── stock_market_dataset.json # Output of stock_data_pipeline.py
│   └── stock_index.pkl           # Output of stock_retrieval.py --mode index
└── outputs.txt                   # Sample run logs and ablation results
1. Install dependencies
pip install -r requirements.txt
requirements.txt
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.2
transformers>=4.36.0
torch>=2.0.0
scikit-learn>=1.3.0
yfinance>=0.2.28
tqdm>=4.66.0
matplotlib>=3.7.0
nltk>=3.8.1
python-dotenv>=1.0.0
python stock_data_pipeline.py --input data/stock_data.csv --output data/stock_market_dataset.json
| Argument   | Default                          | Description             |
| ---------- | -------------------------------- | ----------------------- |
| `--input`  | `data/stock_data.csv`            | Input stock CSV dataset |
| `--output` | `data/stock_market_dataset.json` | Output metadata JSON    |
| `--seed`   | `42`                             | Random seed             |
Sample Output:
Total records : 5000
Unique sectors: 12
Risk Levels   : {'Low Risk': 2100, 'Moderate Risk': 1900, 'High Risk': 1000}
Top sectors   : [('Technology', 1100), ('Banking', 920), ('Energy', 700)]
3. Build the Financial Index

Embed all stock records into normalized semantic vectors and save to disk:
python stock_retrieval.py --mode index --data data/stock_market_dataset.json
Output:
data/stock_index.pkl
4. Query by text

Retrieve top-k semantically relevant stocks:
python stock_retrieval.py --mode query --query "safe long-term technology stocks"
| Argument  | Default                          | Description               |
| --------- | -------------------------------- | ------------------------- |
| `--query` | `"safe long-term growth stocks"` | Free-text financial query |
| `--topk`  | `5`                              | Number of results         |
| `--index` | `data/stock_index.pkl`           | Saved stock index         |
Sample Output:
Rank   Score    Company              Sector          Close     Risk
──────────────────────────────────────────────────────────────────────────────
1      0.9123   Infosys              Technology      1540.20   Low Risk
       └─ Technology stock Infosys with PE ratio 22, positive return...
       5. Run the Ablation Study

Compare baseline forecasting vs semantic retrieval:
python stock_retrieval.py --mode ablation
Ablation Results
| Query Type                    | Baseline ML | Financial RAG |
| ----------------------------- | ----------- | ------------- |
| Price Prediction Accuracy     | High        | Moderate      |
| Semantic Investment Relevance | Low         | High          |
| Risk Awareness                | Medium      | High          |
| User-Friendly Insights        | Low         | High          |
Architecture
User query (natural language)
        │
        ▼
Financial Text Encoder (SentenceTransformer / FinBERT)
        │
        ▼
Semantic Embedding Vector
        │
        ▼
Cosine Similarity Search
        │
        ▼
Top-K Relevant Stocks
(Company, Sector, PE Ratio, Volatility, Risk)
        │
        ▼
[Final Milestone] LLM generates
grounded investment recommendation
Dataset

Structured Stock Dataset
Possible fields:

Company
Sector
Open
Close
High
Low
Volume
PE Ratio
News Sentiment
Expected Performance
| Metric                       | Milestone 1  | Final Milestone |
| ---------------------------- | ------------ | --------------- |
| Retrieval Relevance@5        | 0.70+ target | 0.90+           |
| Risk Classification Accuracy | 0.75+        | 0.90+           |
| Forecast RMSE                | Baseline     | Improved        |
| LLM Investment Explanation   | —            | ROUGE-L > 0.35  |
References
Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
Araci, D. (2019). FinBERT: Financial Sentiment Analysis.
Lewis, P. et al. (2020). Retrieval-Augmented Generation.
Yahoo Finance API Documentation
NSE/BSE Historical Market Data
