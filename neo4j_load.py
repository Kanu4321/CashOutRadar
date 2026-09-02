"""
Loads the synthetic accounts/transactions CSVs into Neo4j, then runs
Graph Data Science (GDS) algorithms on top: PageRank (influence/hub
detection), Louvain (community/ring detection), and FastRP (node
embeddings used later as ML features).

Requires:
  - A running Neo4j instance with the GDS plugin installed
    (Neo4j Desktop: add the "Graph Data Science Library" plugin to your
     database, then start it. Or via Docker, see README_NEO4J.md.)
  - pip install neo4j pandas

Env vars (defaults shown):
  NEO4J_URI      bolt://localhost:7687
  NEO4J_USER     neo4j
  NEO4J_PASSWORD password

Run:
  python3 neo4j_load.py
"""

import os
import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

DATA_DIR = "output"
GRAPH_NAME = "fraudGraph"


def batched(df, size=500):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size].to_dict("records")


class Loader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session() as session:
            return list(session.run(query, **params))

    # -- schema -------------------------------------------------------

    def setup_constraints(self):
        self.run("CREATE CONSTRAINT account_id IF NOT EXISTS "
                  "FOR (a:Account) REQUIRE a.account_id IS UNIQUE")

    def wipe(self):
        # Dangerous in a real deployment; fine for a hackathon reset.
        self.run("MATCH (n) DETACH DELETE n")

    # -- load -----------------------------------------------------------

    def load_accounts(self, accounts_df):
        query = """
        UNWIND $rows AS row
        MERGE (a:Account {account_id: row.account_id})
        SET a.holder_name = row.holder_name,
            a.bank = row.bank,
            a.branch_city = row.branch_city,
            a.latitude = toFloat(row.latitude),
            a.longitude = toFloat(row.longitude),
            a.kyc_completeness = toFloat(row.kyc_completeness),
            a.account_age_days = toInteger(row.account_age_days),
            a.opened_on = row.opened_on,
            a.prior_fraud_flag = row.prior_fraud_flag,
            a.account_type = row.account_type
        WITH a, row
        CALL apoc.create.addLabels(a, [row.label]) YIELD node
        RETURN count(*)
        """
        # Fallback if APOC isn't installed: plain label-free load below.
        simple_query = """
        UNWIND $rows AS row
        MERGE (a:Account {account_id: row.account_id})
        SET a.holder_name = row.holder_name,
            a.bank = row.bank,
            a.branch_city = row.branch_city,
            a.latitude = toFloat(row.latitude),
            a.longitude = toFloat(row.longitude),
            a.kyc_completeness = toFloat(row.kyc_completeness),
            a.account_age_days = toInteger(row.account_age_days),
            a.opened_on = row.opened_on,
            a.prior_fraud_flag = row.prior_fraud_flag,
            a.account_type = row.account_type
        """
        rows = accounts_df.to_dict("records")
        for row in rows:
            row["label"] = {
                "victim": "Victim",
                "mule": "Mule",
                "cashout_node": "CashoutNode",
                "legit": "Legit",
            }.get(row["account_type"], "Legit")

        for batch in batched(pd.DataFrame(rows)):
            try:
                self.run(query, rows=batch)
            except Exception:
                # APOC not available - load without dynamic labels.
                self.run(simple_query, rows=batch)

    def load_transactions(self, txns_df):
        query = """
        UNWIND $rows AS row
        MATCH (a:Account {account_id: row.from_account})
        MATCH (b:Account {account_id: row.to_account})
        MERGE (a)-[t:TRANSACTED {txn_id: row.txn_id}]->(b)
        SET t.amount = toFloat(row.amount),
            t.channel = row.channel,
            t.timestamp = row.timestamp,
            t.is_fraud_chain = row.is_fraud_chain,
            t.is_cashout = row.is_cashout,
            t.ring_id = row.ring_id
        """
        df = txns_df.copy()
        df["ring_id"] = df["ring_id"].fillna(-1)
        for batch in batched(df):
            self.run(query, rows=batch)

    # -- graph data science --------------------------------------------

    def project_graph(self):
        self.run(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
        self.run(f"""
        CALL gds.graph.project(
          '{GRAPH_NAME}',
          'Account',
          {{
            TRANSACTED: {{ orientation: 'NATURAL', properties: 'amount' }}
          }}
        )
        """)

    def run_pagerank(self):
        self.run(f"""
        CALL gds.pageRank.write('{GRAPH_NAME}', {{
          writeProperty: 'pagerank',
          relationshipWeightProperty: 'amount'
        }})
        """)

    def run_louvain(self):
        self.run(f"""
        CALL gds.louvain.write('{GRAPH_NAME}', {{
          writeProperty: 'community'
        }})
        """)

    def run_fastrp(self, dim=32):
        self.run(f"""
        CALL gds.fastRP.write('{GRAPH_NAME}', {{
          embeddingDimension: {dim},
          writeProperty: 'embedding',
          relationshipWeightProperty: 'amount'
        }})
        """)

    def export_features(self):
        rows = self.run("""
        MATCH (a:Account)
        RETURN a.account_id AS account_id,
               a.account_type AS account_type,
               a.pagerank AS pagerank,
               a.community AS community,
               a.embedding AS embedding
        """)
        return pd.DataFrame([dict(r) for r in rows])


def main():
    accounts_df = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    txns_df = pd.read_csv(f"{DATA_DIR}/transactions.csv")

    loader = Loader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        print("Setting up constraints...")
        loader.setup_constraints()

        print(f"Loading {len(accounts_df)} accounts...")
        loader.load_accounts(accounts_df)

        print(f"Loading {len(txns_df)} transactions...")
        loader.load_transactions(txns_df)

        print("Projecting in-memory graph for GDS...")
        loader.project_graph()

        print("Running PageRank (hub/influence detection)...")
        loader.run_pagerank()

        print("Running Louvain (community/ring detection)...")
        loader.run_louvain()

        print("Running FastRP (node embeddings for ML features)...")
        loader.run_fastrp()

        print("Exporting graph features...")
        features_df = loader.export_features()
        features_df.to_csv(f"{DATA_DIR}/graph_features.csv", index=False)
        print(f"Wrote {DATA_DIR}/graph_features.csv ({len(features_df)} rows)")

    finally:
        loader.close()


if __name__ == "__main__":
    main()
