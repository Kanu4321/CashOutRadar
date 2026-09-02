# Setting up Neo4j for the prototype

This script needs a running Neo4j instance with the **Graph Data Science (GDS)**
plugin installed. Two easy ways to get one:

## Option A: Docker (fastest)

```bash
docker run -d \
  --name fraud-graph \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4JLABS_PLUGINS='["graph-data-science"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=gds.* \
  neo4j:5
```

Then open http://localhost:7474 in a browser to confirm it's up
(login: `neo4j` / `password`). Bolt connections (used by the Python
script) are on `bolt://localhost:7687`.

## Option B: Neo4j Desktop (no Docker needed)

1. Download Neo4j Desktop: https://neo4j.com/download/
2. Create a new local database (any password you like)
3. On that database, go to "Plugins" and install **Graph Data Science Library**
4. Start the database
5. Note the Bolt port shown (usually `7687`)

## Configure the script

Set these environment variables to match your setup (defaults already
match the Docker command above):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

On Windows (PowerShell):
```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="password"
```

## Run it

Make sure `output/accounts.csv` and `output/transactions.csv` already
exist (run `generate.py` first if not), then:

```bash
pip install neo4j pandas
python3 neo4j_load.py
```

This will:
1. Load every account as a node, every transaction as a relationship
2. Run PageRank (finds structurally important/hub accounts)
3. Run Louvain community detection (finds likely fraud-ring clusters)
4. Run FastRP (creates a numeric embedding per account for later ML use)
5. Write everything back out to `output/graph_features.csv`

That CSV — one row per account with `pagerank`, `community`, and
`embedding` columns — is what the next step (the XGBoost risk model)
will train on.

## Exploring visually

Open http://localhost:7474, log in, and try:

```cypher
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WHERE t.ring_id = 0
RETURN a, t, b
```

This shows one fraud ring's money trail directly in Neo4j's built-in
graph visualizer — useful for screenshots/demo footage beyond the
matplotlib version.
