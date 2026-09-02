Markdown
# Cash-out Radar

**Real-Time Cybercrime Ring Forensics & Mule-Account Interception Engine**

Cash-out Radar is an intelligence console engineered to dismantle organized financial cybercrime networks. Designed around modern money-mule laundering typologies, it reconstructs multi-hop transaction trails from initial victim complaints, predicts downstream terminal cash-out points via graph-augmented machine learning, and maintains a cryptographically verifiable evidentiary log.

---

### Core Architecture

[ Synthetic NCRP Ledger Engine ]
│
▼
[ NetworkX / Neo4j GDS ] ────────► Topological Features (PageRank, In/Out-Degree)
│
▼
[ XGBoost Classifier ] ───────► Risk Probability Scoring
│
▼
[ TreeSHAP Explainability ] ───► Forensic Drivers per Flagged Node
│
▼
[ FastAPI Backend (Render) ] ───► REST Endpoints + WebSocket Ticker
│
▼
[ Vanilla SPA (Vercel CDN) ] ───► Force-Directed Ring Topology & Spatial KDE


* **Network Topology & Flow Analysis:** Models smurfing and rapid fan-out structures across intermediate mule tiers using directed graph metrics (NetworkX baseline with optional Neo4j Graph Data Science offload).
* **Predictive Cash-Out Pinpointing:** Uses gradient-boosted decision trees (`XGBoost`) trained on graph centrality and structural transaction metadata to distinguish routing intermediaries from terminal cash-out nodes.
* **Explainable AI Justification:** Implements `SHAP` (SHapley Additive exPlanations) to decompose high-risk predictions into compliant legal justifications (e.g., degree anomalies, rapid balance dispersal) before dispatching freezing alerts.
* **Forensic Tamper Resistance:** Maintains an append-only audit trail chained via SHA-256 block hashing, enabling mathematical verification of evidentiary integrity.
* **Decoupled Deployment:** Zero-build static frontend deployed globally on edge CDNs (Vercel), interfacing with an asynchronous backend service (FastAPI on Render).

---

### Project Structure

cash-out-radar/
├── generate.py           # Synthetic ledger & NCRP complaint generation engine
├── train_model.py        # Feature engineering, XGBoost training, & spatial KDE fitting
├── main.py               # Asynchronous FastAPI engine, WebSocket feeds, and REST routes
├── neo4j_load.py         # Optional enterprise loader for Neo4j GDS algorithms
├── index.html            # High-performance dashboard (D3-force, SVG canvas, vanilla JS)
├── requirements.txt      # Production runtime dependencies
├── models/               # Serialized model artifacts (.joblib)
└── output/               # Structured transaction ledgers (.csv)


---

### Key Operational Metrics

| Metric | Target Value | Description |
|---|---|---|
| **Avg Hop Distance** | `4.4 hops` | Mean path length from victim origination to terminal extraction node. |
| **Audit Verification** | `SHA-256 Chain` | Mathematical proof against log tampering or backdated alteration. |
| **Explanation Latency** | `< 20ms` | Real-time TreeSHAP contribution computation per high-risk node. |
| **Topology Engine** | `D3 Force Simulation` | Dynamic physics-driven multi-layer node graph rendering. |

---

### Local Installation & Reproduction

**1. Clone and Configure Environment**
```bash
git clone [https://github.com/your-username/cash-out-radar.git](https://github.com/your-username/cash-out-radar.git)
cd cash-out-radar
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate
pip install -r requirements.txt
```
**2. Generate Graph Ledgers & Fit Models**

Bash
# Generate synthetic accounts, layering hops, and NCRP incident reports
python generate.py

# Optional: Load into local Neo4j instance for enterprise GDS metrics
# python neo4j_load.py

# Extract graph centrality features, fit XGBoost, and cache spatial KDE grid
python train_model.py

**3. Launch Backend Engine**

Bash
uvicorn main:app --reload --port 8000

**4. Serve Dashboard**

Bash
# Serve frontend via local HTTP server
python -m http.server 5500
Navigate to http://localhost:5500 or configure the live production endpoint directly in the console header.

System Capabilities
Human-in-the-Loop (HITL) Triaging: Eliminates autonomous freezing liability risks by providing fraud analysts with instantaneous evidentiary context rather than rigid black-box decisions.

Spatial Hotspot Heatmaps: Visualizes jurisdictional cybercrime clusters using two-dimensional Kernel Density Estimation over geographic coordinates.

Interactive Money Trails: Traces individual complaints from root victims through arbitrary mule layers to terminal ATM/point-of-sale cashouts via interactive SVG vector graphs.

Methodological Transparency
Demonstration Caveat:

The high classification metrics obtained on the current benchmark dataset reflect structural topological separation inherent to synthetic smurfing patterns (e.g., tightly bounded in-degree to out-degree ratios for sink accounts). In real-world core banking feeds, operational noise, dormancy periods, and blended organic transactions yield continuous feature drift, requiring regular retraining pipelines and dynamic boundary tuning.
