# Cash-out Radar — full stack, wired up

You had three separate pieces that didn't connect yet:

1. **`generate.py`** → synthetic data (`output/*.csv`) — already generated for you.
2. **`main.py`** → FastAPI backend — but it wouldn't start, because it
   loads `output/training_table.csv` and `models/*.joblib`, and nothing
   in either of your zips produced them.
3. **`index.html`** → the dashboard frontend — already correctly wired
   to call `http://localhost:8000/api/...` and `ws://localhost:8000/ws/live`,
   it just had no backend to talk to.

The missing link was a training script. I added **`train_model.py`**,
which reads the generated CSVs, builds the same features `main.py`
expects (`pagerank`, `in_degree`, `out_degree`, `kyc_completeness`,
`account_age_days`, `prior_fraud_flag`), trains the XGBoost cash-out
risk model, and fits the hotspot density model — using NetworkX
PageRank as a fallback if you haven't loaded the data into Neo4j
(`neo4j_load.py` is still here and optional; if you run it first,
`train_model.py` will detect `output/graph_features.csv` and use the
real GDS PageRank instead).

I ran the feature-engineering and model-fitting logic here to confirm
it produces a clean `training_table.csv` (1124 rows, 60 positives, no
missing values) and a working hotspot grid. I couldn't pip-install
`xgboost`/`fastapi`/`shap` inside this sandbox (no network egress), so
I verified the same logic with a NetworkX + scikit-learn stand-in
model rather than running the literal final script — the code you're
getting uses the real `xgboost`/`shap` stack from `requirements.txt`,
which is what `main.py` expects. Run the steps below locally to do the
final live check.

## Run it

```bash
pip install -r requirements.txt

# 1. (data is already in output/, but to regenerate:)
python3 generate.py

# 2. (optional) load into Neo4j for real graph algorithms - see README_NEO4J.md
#    python3 neo4j_load.py

# 3. train the models (the piece that was missing)
python3 train_model.py

# 4. run the API
uvicorn main:app --reload --port 8000
```

Then open `index.html` directly in your browser (double-click it, or
`python3 -m http.server 5500` and visit it) — the API base field at
the top already defaults to `http://localhost:8000`, so it should
light up immediately: stats, live complaint feed, risk alerts, hotspot
map, fraud-ring graph view, and the tamper-evident audit log.

## Files

| File | Role |
|---|---|
| `generate.py` | synthetic NCRP-style dataset generator |
| `output/*.csv` | pre-generated data (accounts, transactions, complaints, cashouts) |
| `neo4j_load.py` | optional: loads data into Neo4j, runs PageRank/Louvain/FastRP |
| `train_model.py` | **new** — builds `training_table.csv`, trains the risk model, fits the hotspot KDE |
| `main.py` | FastAPI backend serving the dashboard |
| `index.html` | the dashboard itself (static, no build step) |

## Honest note (carried over from the original README)

On this synthetic data the risk model scores near-perfect
precision/recall — that's because cash-out accounts sit at a
structurally distinctive in/out-degree combination in the generator,
not a realistic result. Say that plainly in a demo.
