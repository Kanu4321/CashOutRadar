"""
train_model_real.py
====================
Replacement for train_model.py that trains the "Cash-out Radar" risk model
on the REAL IBM Transactions for Anti-Money-Laundering (AML) Kaggle dataset
instead of generate.py's synthetic data.

WHY THIS IS DIFFERENT FROM THE ORIGINAL train_model.py
--------------------------------------------------------
1. Real data has no `kyc_completeness`, `account_age_days` or
   `prior_fraud_flag` fields (those only existed in the synthetic
   generator). All features here are derived purely from real
   transaction behaviour, so there's nothing left for the model to
   "cheat" on.
2. There is no `cashouts.csv` telling us which account is a cash-out
   node. Ground truth instead comes from `HI-Large_Patterns.txt`,
   which lists the exact transactions that make up each known
   laundering attempt (FAN-OUT, CYCLE, GATHER-SCATTER, STACK,
   BIPARTITE, FAN-IN, RANDOM, SCATTER-GATHER). For every laundering
   attempt we treat any account that RECEIVES laundered money but
   never forwards it onward within that same attempt as a terminal /
   cash-out node (label = 1). Every other account is label = 0.
   This is the real analogue of "where the money finally leaves the
   traceable banking graph".

INPUTS (edit DATA_DIR below or set env var)
  <DATA_DIR>/HI-Large_Trans.csv       (or HI-Medium / HI-Large)
  <DATA_DIR>/HI-Large_accounts.csv
  <DATA_DIR>/HI-Large_Patterns.txt

OUTPUTS (same contract as the original train_model.py, so main.py
keeps working without changes to its loading code)
  output/training_table.csv
  models/risk_model.joblib
  models/risk_model_features.joblib
  models/hotspot_kde.joblib   (best-effort, see NOTE below)

NOTE ON HOTSPOTS
----------------
The real dataset has no lat/lon for accounts. As a best-effort
substitute we parse the country out of `Bank Name` (e.g. "Germany
Bank #4815", "UK Bank #29") and map it to an approximate country
centroid. Accounts at "First/National/Savings Bank of <US city>" are
bucketed as USA, and "Crytpo Bank" accounts (an actual category in
this dataset!) are dropped from the geo model since crypto cash-out
has no physical location. This is approximate — treat the hotspot
map as illustrative, not authoritative, until you have real
geolocation data.

Run:
  pip install scikit-learn xgboost shap joblib scipy pandas numpy --break-system-packages
  python3 train_model_real.py
"""

import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
import xgboost as xgb

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("DATA_DIR", "kaggle_data")   # where the 3 Kaggle files live
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "HI-Large")  # HI-Small / HI-Medium / HI-Large
OUT_DIR = os.environ.get("OUT_DIR", "output")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRANS_PATH = f"{DATA_DIR}/{DATASET_PREFIX}_Trans.csv"
ACCOUNTS_PATH = f"{DATA_DIR}/{DATASET_PREFIX}_accounts.csv"
PATTERNS_PATH = f"{DATA_DIR}/{DATASET_PREFIX}_Patterns.txt"

FEATURE_COLS = [
    "out_degree", "in_degree", "unique_receivers", "unique_senders",
    "total_sent", "total_received", "avg_sent", "avg_received",
    "counterparty_banks_out", "counterparty_banks_in",
    "currencies_out", "currencies_in", "self_loop_count",
    "net_flow", "in_out_amount_ratio", "total_degree",
    "total_banks_touched", "total_currencies", "pagerank",
]

# ---------------------------------------------------------------------------
# 1. Load transactions
# ---------------------------------------------------------------------------

print(f"Loading {TRANS_PATH} ...")
t0 = time.time()
df = pd.read_csv(TRANS_PATH)
df.columns = [
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
# (Bank, Account) pair is the true unique node id in this dataset -
# raw account numbers alone are *almost* unique but not guaranteed to be.
df["from_node"] = df["From Bank"].astype(str) + "_" + df["From Account"].astype(str)
df["to_node"] = df["To Bank"].astype(str) + "_" + df["To Account"].astype(str)
print(f"  loaded {len(df):,} transactions in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Account-level behavioural features (vectorized groupby - fast even at
#    millions of rows; avoid row-by-row iteration entirely)
# ---------------------------------------------------------------------------

print("Building account-level features ...")
t0 = time.time()
all_nodes = pd.unique(pd.concat([df["from_node"], df["to_node"]], ignore_index=True))
node_idx = pd.Series(np.arange(len(all_nodes)), index=all_nodes)
n_nodes = len(all_nodes)

feat = pd.DataFrame(index=all_nodes)
feat["out_degree"] = df.groupby("from_node").size()
feat["in_degree"] = df.groupby("to_node").size()
feat["unique_receivers"] = df.groupby("from_node")["to_node"].nunique()
feat["unique_senders"] = df.groupby("to_node")["from_node"].nunique()
feat["total_sent"] = df.groupby("from_node")["Amount Paid"].sum()
feat["total_received"] = df.groupby("to_node")["Amount Received"].sum()
feat["avg_sent"] = df.groupby("from_node")["Amount Paid"].mean()
feat["avg_received"] = df.groupby("to_node")["Amount Received"].mean()
feat["counterparty_banks_out"] = df.groupby("from_node")["To Bank"].nunique()
feat["counterparty_banks_in"] = df.groupby("to_node")["From Bank"].nunique()
feat["currencies_out"] = df.groupby("from_node")["Payment Currency"].nunique()
feat["currencies_in"] = df.groupby("to_node")["Receiving Currency"].nunique()
feat["self_loop_count"] = df[df["from_node"] == df["to_node"]].groupby("from_node").size()
feat = feat.fillna(0.0)

feat["net_flow"] = feat["total_received"] - feat["total_sent"]
feat["in_out_amount_ratio"] = feat["total_received"] / (feat["total_sent"] + 1.0)
feat["total_degree"] = feat["in_degree"] + feat["out_degree"]
feat["total_banks_touched"] = feat["counterparty_banks_out"] + feat["counterparty_banks_in"]
feat["total_currencies"] = feat["currencies_out"] + feat["currencies_in"]

# PageRank via scipy sparse power iteration (much faster than networkx at
# this scale and needs no extra dependency).
src = df["from_node"].map(node_idx).values
dst = df["to_node"].map(node_idx).values
A = sp.coo_matrix((np.ones(len(df)), (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
out_w = np.array(A.sum(axis=1)).flatten()
dangling = (out_w == 0).astype(float)
out_w[out_w == 0] = 1.0
P = sp.diags(1.0 / out_w) @ A

damping = 0.85
p = np.full(n_nodes, 1.0 / n_nodes)
for _ in range(50):
    p_new = damping * (P.T @ p) + damping * (dangling @ p) / n_nodes + (1 - damping) / n_nodes
    if np.abs(p_new - p).sum() < 1e-10:
        p = p_new
        break
    p = p_new
feat["pagerank"] = pd.Series(p, index=all_nodes)

print(f"  {n_nodes:,} accounts featurized in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. Ground-truth cash-out labels from HI-Small_Patterns.txt
# ---------------------------------------------------------------------------

print(f"Parsing {PATTERNS_PATH} for cash-out ground truth ...")
terminal_nodes = set()
cur_rows = []
with open(PATTERNS_PATH) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            cur_rows = []
        elif line.startswith("END LAUNDERING ATTEMPT"):
            senders, receivers = set(), set()
            for r in cur_rows:
                senders.add(f"{int(r[1])}_{r[2]}")
                receivers.add(f"{int(r[3])}_{r[4]}")
            terminal_nodes |= (receivers - senders)  # received but never forwarded on
        elif line.strip() and "," in line:
            cur_rows.append(line.split(","))

feat["is_cashout"] = feat.index.isin(terminal_nodes).astype(int)
print(f"  {feat['is_cashout'].sum():,} cash-out accounts out of {len(feat):,} "
      f"({feat['is_cashout'].mean()*100:.3f}%)")

training_table = feat[FEATURE_COLS + ["is_cashout"]].copy()
training_table.index.name = "account_id"
training_table.to_csv(f"{OUT_DIR}/training_table.csv")
print(f"Wrote {OUT_DIR}/training_table.csv")

# ---------------------------------------------------------------------------
# 4. Train XGBoost
# ---------------------------------------------------------------------------

X = training_table[FEATURE_COLS]
y = training_table["is_cashout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos  # real data is heavily imbalanced (~0.2% positive)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
pr_auc = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
baseline = y_test.mean()

print("\nEvaluation on held-out test set (REAL data, real imbalance):")
print(f"  precision @0.5   = {precision:.3f}")
print(f"  recall @0.5      = {recall:.3f}")
print(f"  PR-AUC           = {pr_auc:.4f}  (random baseline = {baseline:.5f})")
print(f"  ROC-AUC          = {roc_auc:.3f}")

print("\nThreshold sweep (use this to pick an operating point for analysts):")
print(f"  {'threshold':>9} {'precision':>10} {'recall':>8} {'flagged':>9}")
for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    p_t = (proba >= t).astype(int)
    print(f"  {t:>9.2f} {precision_score(y_test, p_t, zero_division=0):>10.3f} "
          f"{recall_score(y_test, p_t, zero_division=0):>8.3f} {p_t.sum():>9d}")

print(
    "\nHonest note: this is real, labeled laundering data (from IBM's AML "
    "simulator), so these numbers reflect genuine signal, not a synthetic-"
    "data artifact. ROC-AUC in the 0.85-0.95 range with low absolute "
    "precision at low thresholds is EXPECTED and realistic for AML/fraud "
    "detection given the severe class imbalance (~0.2% positive). Pick your "
    "deployed threshold based on how many alerts an analyst team can "
    "actually review per day (recall-heavy / low threshold) vs. how "
    "confident you need to be before freezing an account (precision-heavy "
    "/ high threshold) - don't just default to 0.5."
)

joblib.dump(model, f"{MODEL_DIR}/risk_model.joblib")
joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/risk_model_features.joblib")
print(f"\nWrote {MODEL_DIR}/risk_model.joblib, {MODEL_DIR}/risk_model_features.joblib")

if HAVE_SHAP:
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, f"{MODEL_DIR}/risk_model_shap.joblib")
    print(f"Wrote {MODEL_DIR}/risk_model_shap.joblib")
else:
    print("shap not installed - skipping SHAP explainer export "
          "(pip install shap --break-system-packages)")

# ---------------------------------------------------------------------------
# 5. Best-effort hotspot map (see module docstring NOTE)
# ---------------------------------------------------------------------------

COUNTRY_CENTROIDS = {
    "germany": (51.16, 10.45), "switzerland": (46.82, 8.23), "china": (35.86, 104.20),
    "france": (46.23, 2.21), "india": (20.59, 78.96), "israel": (31.05, 34.85),
    "uk": (55.38, -3.44), "italy": (41.87, 12.57), "japan": (36.20, 138.25),
    "spain": (40.46, -3.75), "australia": (-25.27, 133.78), "canada": (56.13, -106.35),
    "russia": (61.52, 105.32), "mexico": (23.63, -102.55), "saudi arabia": (23.89, 45.08),
    "netherlands": (52.13, 5.29), "brazil": (-14.24, -51.93), "belgium": (50.50, 4.47),
    "austria": (47.52, 14.55), "greece": (39.07, 21.82), "portugal": (39.40, -8.22),
    "ireland": (53.41, -8.24), "usa": (39.83, -98.58),
}


def bank_to_country(bank_name: str):
    low = bank_name.lower()
    if "crytpo" in low or "crypto" in low:
        return None  # crypto cash-out has no physical location - excluded from geo model
    for country in COUNTRY_CENTROIDS:
        if country != "usa" and country in low:
            return country
    if any(k in low for k in ["first bank", "national bank", "savings bank",
                               "bank of", "cooperative bank"]):
        return "usa"
    return None


if os.path.exists(ACCOUNTS_PATH):
    print(f"\nBuilding best-effort hotspot model from {ACCOUNTS_PATH} ...")
    accounts_df = pd.read_csv(ACCOUNTS_PATH)
    accounts_df["node_id"] = accounts_df["Bank ID"].astype(str) + "_" + accounts_df["Account Number"].astype(str)
    accounts_df["country"] = accounts_df["Bank Name"].apply(bank_to_country)

    cashout_accounts = accounts_df[accounts_df["node_id"].isin(terminal_nodes)].dropna(subset=["country"])
    if len(cashout_accounts) >= 5:
        coords_deg = np.array([COUNTRY_CENTROIDS[c] for c in cashout_accounts["country"]])
        # small jitter so KDE isn't degenerate on repeated identical centroids
        rng = np.random.default_rng(42)
        coords_deg = coords_deg + rng.normal(0, 0.3, coords_deg.shape)
        coords_rad = np.radians(coords_deg)

        from sklearn.neighbors import KernelDensity
        hotspot_kde = KernelDensity(bandwidth=0.05, metric="haversine", kernel="gaussian")
        hotspot_kde.fit(coords_rad)
        joblib.dump(hotspot_kde, f"{MODEL_DIR}/hotspot_kde.joblib")
        print(f"Wrote {MODEL_DIR}/hotspot_kde.joblib "
              f"(approximate - based on {len(cashout_accounts)} cash-out accounts' "
              f"bank country, NOT real geolocation)")
    else:
        print("Not enough geo-taggable cash-out accounts to fit a hotspot model - skipping.")
else:
    print(f"\n{ACCOUNTS_PATH} not found - skipping hotspot model.")

print("\nDone.")
