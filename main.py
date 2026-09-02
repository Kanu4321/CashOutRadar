"""
Backend API for the Cybercrime Cash-out Prediction prototype.

Serves data from the synthetic dataset + trained models (see
generate.py and train_model.py, which must be run first) to the React
frontend: complaints, per-complaint fraud-ring graphs, ranked risk
alerts (with SHAP explanations), a hotspot grid for the map, a
simulated tamper-evident audit log, and a WebSocket that replays
complaints as a "live" feed for demo purposes.

Run (after generate.py and train_model.py have produced ./output and
./models):
  pip install fastapi "uvicorn[standard]" shap
  uvicorn main:app --reload --port 8000

Docs then available at http://localhost:8000/docs
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = os.environ.get("DATA_DIR", "output")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

app = FastAPI(title="Cybercrime Cash-out Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data + models once at startup
# ---------------------------------------------------------------------------

accounts_df = pd.read_csv(f"{DATA_DIR}/accounts.csv")
txns_df = pd.read_csv(f"{DATA_DIR}/transactions.csv")
complaints_df = pd.read_csv(f"{DATA_DIR}/complaints.csv")
cashouts_df = pd.read_csv(f"{DATA_DIR}/cashouts.csv")
training_df = pd.read_csv(f"{DATA_DIR}/training_table.csv")

risk_model = joblib.load(f"{MODEL_DIR}/risk_model.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/risk_model_features.joblib")
hotspot_kde = joblib.load(f"{MODEL_DIR}/hotspot_kde.joblib")

explainer = shap.TreeExplainer(risk_model)

# Precompute a risk score + top SHAP driver for every account once,
# rather than recomputing per-request.
X_all = training_df[feature_cols].fillna(0)
risk_scores = risk_model.predict_proba(X_all)[:, 1]
shap_values = explainer.shap_values(X_all)

training_df = training_df.copy()
training_df["risk_score"] = risk_scores

FEATURE_LABELS = {
    "pagerank": "network centrality (PageRank)",
    "in_degree": "number of incoming transactions",
    "out_degree": "number of outgoing transactions",
    "kyc_completeness": "KYC completeness",
    "account_age_days": "account age",
    "prior_fraud_flag": "prior fraud flag on this account",
}


def top_driver_for_row(i):
    row_shap = shap_values[i]
    idx = int(np.argmax(np.abs(row_shap)))
    feat = feature_cols[idx]
    direction = "increases" if row_shap[idx] > 0 else "decreases"
    return f"{FEATURE_LABELS.get(feat, feat)} {direction} risk"


training_df["top_driver"] = [top_driver_for_row(i) for i in range(len(training_df))]

# ---------------------------------------------------------------------------
# Simple hash-chained audit log (blockchain-concept demo, not real Fabric)
# ---------------------------------------------------------------------------

audit_log = []


def append_audit(event_type: str, payload: dict):
    prev_hash = audit_log[-1]["hash"] if audit_log else "0" * 64
    entry = {
        "index": len(audit_log),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    entry_str = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
    audit_log.append(entry)
    return entry


append_audit("system_start", {"note": "API started, models loaded"})

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/stats")
def get_stats():
    return {
        "total_complaints": int(len(complaints_df)),
        "total_accounts": int(len(accounts_df)),
        "total_transactions": int(len(txns_df)),
        "confirmed_cashouts": int(len(cashouts_df)),
        "avg_hops_to_cashout": round(float(cashouts_df["hops_from_victim"].mean()), 2),
    }


@app.get("/api/complaints")
def list_complaints(limit: int = 50):
    df = complaints_df.sort_values("filed_at", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/api/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    row = complaints_df[complaints_df["complaint_id"] == complaint_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="complaint not found")
    return row.iloc[0].to_dict()


@app.get("/api/complaints/{complaint_id}/graph")
def get_complaint_graph(complaint_id: str):
    row = complaints_df[complaints_df["complaint_id"] == complaint_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="complaint not found")
    ring_id = int(row.iloc[0]["ring_id"])

    ring_txns = txns_df[txns_df.get("ring_id") == ring_id]
    if ring_txns.empty:
        raise HTTPException(status_code=404, detail="no graph data for this complaint")

    node_ids = pd.unique(ring_txns[["from_account", "to_account"]].values.ravel())
    node_rows = accounts_df[accounts_df["account_id"].isin(node_ids)]

    risk_lookup = dict(zip(training_df["account_id"], training_df["risk_score"]))

    nodes = []
    for _, n in node_rows.iterrows():
        nodes.append({
            "id": n["account_id"],
            "type": n["account_type"],
            "bank": n["bank"],
            "city": n["branch_city"],
            "latitude": n["latitude"],
            "longitude": n["longitude"],
            "risk_score": round(float(risk_lookup.get(n["account_id"], 0.0)), 4),
        })

    links = []
    for _, t in ring_txns.iterrows():
        links.append({
            "source": t["from_account"],
            "target": t["to_account"],
            "amount": t["amount"],
            "channel": t["channel"],
            "timestamp": t["timestamp"],
            "is_cashout": bool(t["is_cashout"]),
        })

    return {"complaint_id": complaint_id, "ring_id": ring_id, "nodes": nodes, "links": links}


@app.get("/api/alerts")
def get_alerts(top_k: int = 20, min_score: float = 0.0):
    df = training_df[training_df["risk_score"] >= min_score].copy()
    df = df.sort_values("risk_score", ascending=False).head(top_k)

    merged = df.merge(
        accounts_df[["account_id", "branch_city", "latitude", "longitude", "bank"]],
        on="account_id", how="left", suffixes=("", "_acct"),
    )

    alerts = []
    for _, r in merged.iterrows():
        alerts.append({
            "account_id": r["account_id"],
            "risk_score": round(float(r["risk_score"]), 4),
            "explanation": r["top_driver"],
            "city": r.get("branch_city_acct", r.get("branch_city")),
            "latitude": r.get("latitude_acct", r.get("latitude")),
            "longitude": r.get("longitude_acct", r.get("longitude")),
            "bank": r.get("bank_acct", r.get("bank")),
        })

    append_audit("alerts_dispatched", {"count": len(alerts), "min_score": min_score})
    return {"count": len(alerts), "alerts": alerts}


@app.get("/api/hotspots")
def get_hotspots(grid_size: int = 40):
    lat_min, lat_max = 6.0, 37.0
    lon_min, lon_max = 68.0, 97.0

    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    grid_rad = np.radians(grid_points)

    log_density = hotspot_kde.score_samples(grid_rad)
    density = np.exp(log_density)
    density = (density - density.min()) / (density.max() - density.min() + 1e-9)

    points = [
        {"latitude": float(lat), "longitude": float(lon), "intensity": float(d)}
        for (lat, lon), d in zip(grid_points, density)
        if d > 0.05  # drop near-zero points to keep the payload small
    ]
    return {"count": len(points), "points": points}


@app.get("/api/audit-log")
def get_audit_log(limit: int = 100):
    return {"count": len(audit_log), "entries": audit_log[-limit:]}


@app.get("/api/audit-log/verify")
def verify_audit_log():
    """Recomputes each entry's hash to prove the log hasn't been tampered with."""
    for i, entry in enumerate(audit_log):
        check = {k: v for k, v in entry.items() if k != "hash"}
        recomputed = hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest()
        if recomputed != entry["hash"]:
            return {"valid": False, "broken_at_index": i}
        if i > 0 and entry["prev_hash"] != audit_log[i - 1]["hash"]:
            return {"valid": False, "broken_at_index": i}
    return {"valid": True, "entries_checked": len(audit_log)}


# ---------------------------------------------------------------------------
# WebSocket: replays complaints as a simulated live feed
# ---------------------------------------------------------------------------


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        records = complaints_df.sort_values("filed_at").to_dict(orient="records")
        i = 0
        while True:
            record = records[i % len(records)]
            append_audit("complaint_received", {"complaint_id": record["complaint_id"]})
            await websocket.send_json({"event": "new_complaint", "data": record})
            i += 1
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": [
            "/api/stats", "/api/complaints", "/api/complaints/{id}",
            "/api/complaints/{id}/graph", "/api/alerts", "/api/hotspots",
            "/api/audit-log", "/api/audit-log/verify", "/ws/live (websocket)",
        ],
    }
