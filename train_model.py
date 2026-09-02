"""
Trains the cash-out risk model + hotspot density model for the
Cybercrime Cash-out Prediction prototype API.

This is the missing link between generate.py (data) and main.py (API):
main.py expects output/training_table.csv, models/risk_model.joblib,
models/risk_model_features.joblib and models/hotspot_kde.joblib to
already exist - this script produces all four.

Reads:
  output/accounts.csv, output/transactions.csv, output/cashouts.csv
  output/graph_features.csv   (optional - written by neo4j_load.py)

Writes:
  output/training_table.csv
  models/risk_model.joblib
  models/risk_model_features.joblib
  models/hotspot_kde.joblib

Run (from the project root, after generate.py):
  pip install scikit-learn xgboost joblib networkx pandas numpy
  python3 train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.neighbors import KernelDensity
import xgboost as xgb

DATA_DIR = os.environ.get("DATA_DIR", "output")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "pagerank",
    "in_degree",
    "out_degree",
    "kyc_completeness",
    "account_age_days",
    "prior_fraud_flag",
]

# ---------------------------------------------------------------------------
# 1. Load base data
# ---------------------------------------------------------------------------

accounts_df = pd.read_csv(f"{DATA_DIR}/accounts.csv")
txns_df = pd.read_csv(f"{DATA_DIR}/transactions.csv")
cashouts_df = pd.read_csv(f"{DATA_DIR}/cashouts.csv")

# ---------------------------------------------------------------------------
# 2. Graph features: use Neo4j GDS output if present (see neo4j_load.py),
#    otherwise compute an equivalent PageRank locally with NetworkX so the
#    pipeline still works without a running Neo4j instance.
# ---------------------------------------------------------------------------

graph_features_path = f"{DATA_DIR}/graph_features.csv"
if os.path.exists(graph_features_path):
    print(f"Found {graph_features_path} - using Neo4j GDS PageRank")
    gf = pd.read_csv(graph_features_path)[["account_id", "pagerank"]]
else:
    print("No graph_features.csv found - computing PageRank locally with NetworkX")
    G = nx.DiGraph()
    G.add_nodes_from(accounts_df["account_id"])
    for _, t in txns_df.iterrows():
        G.add_edge(t["from_account"], t["to_account"], weight=float(t["amount"]))
    pr = nx.pagerank(G, weight="weight")
    gf = pd.DataFrame({"account_id": list(pr.keys()), "pagerank": list(pr.values())})

# in/out-degree are always computed locally - not part of the GDS export
out_degree = txns_df.groupby("from_account").size().rename("out_degree")
in_degree = txns_df.groupby("to_account").size().rename("in_degree")

# ---------------------------------------------------------------------------
# 3. Build the training table
# ---------------------------------------------------------------------------

df = accounts_df.merge(gf, on="account_id", how="left")
df = df.merge(in_degree, left_on="account_id", right_index=True, how="left")
df = df.merge(out_degree, left_on="account_id", right_index=True, how="left")

df["pagerank"] = df["pagerank"].fillna(0.0)
df["in_degree"] = df["in_degree"].fillna(0).astype(int)
df["out_degree"] = df["out_degree"].fillna(0).astype(int)
df["prior_fraud_flag"] = df["prior_fraud_flag"].astype(bool).astype(int)

cashout_ids = set(cashouts_df["cashout_account"])
df["is_cashout"] = df["account_id"].isin(cashout_ids).astype(int)

training_table = df[["account_id"] + FEATURE_COLS + ["is_cashout"]].copy()
training_table.to_csv(f"{DATA_DIR}/training_table.csv", index=False)
print(f"Wrote {DATA_DIR}/training_table.csv ({len(training_table)} rows, "
      f"{training_table['is_cashout'].sum()} positive)")

# ---------------------------------------------------------------------------
# 4. Train the risk model
# ---------------------------------------------------------------------------

X = training_table[FEATURE_COLS]
y = training_table["is_cashout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
pr_auc = average_precision_score(y_test, proba)

print(f"\nEvaluation on held-out test set:")
print(f"  precision = {precision:.3f}")
print(f"  recall    = {recall:.3f}")
print(f"  PR-AUC    = {pr_auc:.3f}")
print(
    "\nHonest note: with this synthetic data generator, cash-out accounts "
    "sit at a very distinctive in/out-degree combination, so near-perfect "
    "scores here are a synthetic-data artifact, not evidence of real-world "
    "performance. Say so plainly in a demo rather than presenting the "
    "number as if it would hold on real NCRP data."
)

joblib.dump(model, f"{MODEL_DIR}/risk_model.joblib")
joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/risk_model_features.joblib")

# ---------------------------------------------------------------------------
# 5. Hotspot density model (feeds /api/hotspots)
#    Fit on confirmed cash-out locations - that's the "where money is
#    actually leaving the system" signal the map is meant to surface.
#    main.py queries this with score_samples() on a lat/lon grid in
#    radians, so it must be fit with a haversine-compatible metric on
#    radians too.
# ---------------------------------------------------------------------------

coords_deg = cashouts_df[["cashout_latitude", "cashout_longitude"]].dropna().values
coords_rad = np.radians(coords_deg)

hotspot_kde = KernelDensity(bandwidth=0.05, metric="haversine", kernel="gaussian")
hotspot_kde.fit(coords_rad)

joblib.dump(hotspot_kde, f"{MODEL_DIR}/hotspot_kde.joblib")

print(
    f"\nWrote {MODEL_DIR}/risk_model.joblib, "
    f"{MODEL_DIR}/risk_model_features.joblib, "
    f"{MODEL_DIR}/hotspot_kde.joblib"
)
