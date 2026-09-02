"""
Synthetic dataset generator for the Cybercrime Cash-out Prediction prototype.

Produces a fake-but-structurally-realistic version of what NCRP + CFCFRMS
data would look like: accounts, transactions (legit + fraud mule chains),
complaints, and ground-truth cash-out events. This is NOT real data and
is only meant to let the graph/ML pipeline be built and demoed.

Outputs (CSV) written to ./output:
  accounts.csv       - every account/wallet node in the graph
  transactions.csv   - every transaction edge (legit + fraud)
  complaints.csv      - NCRP-style complaint records
  cashouts.csv        - ground truth: which account/txn was the cash-out point
  example_ring.png    - visualization of one fraud ring's mule chain
"""

import random
import uuid
import json
from datetime import datetime, timedelta

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from faker import Faker

random.seed(42)
fake = Faker("en_IN")
Faker.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

CITIES = [
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Jaipur", 26.9124, 75.7873),
    ("Lucknow", 26.8467, 80.9462),
    ("Patna", 25.5941, 85.1376),
    ("Ranchi", 23.3441, 85.3096),
    ("Guwahati", 26.1445, 91.7362),
    ("Kolkata", 22.5726, 88.3639),
    ("Ahmedabad", 23.0225, 72.5714),
    ("Chennai", 13.0827, 80.2707),
    ("Jalandhar", 31.3260, 75.5762),
    ("Bhopal", 23.2599, 77.4126),
    ("Nagpur", 21.1458, 79.0882),
]

BANKS = ["SBI", "HDFC", "ICICI", "PNB", "Axis", "Kotak", "BoB", "Canara", "IndusInd", "Union Bank"]

FRAUD_TYPES = [
    "Investment fraud",
    "Digital arrest scam",
    "KYC update fraud",
    "Loan app fraud",
    "Job/task fraud",
    "OTP/phishing fraud",
]

CHANNELS = ["UPI", "IMPS", "NEFT", "RTGS"]

N_LEGIT_ACCOUNTS = 800
N_LEGIT_TXNS = 2500
N_FRAUD_RINGS = 60          # each ring = one complaint's money trail
MIN_HOPS, MAX_HOPS = 3, 6
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 8, 31)


def random_time(start=START_DATE, end=END_DATE):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def jitter_location(lat, lon, km=15):
    # ~0.009 deg latitude per km
    d = km / 111.0
    return lat + random.uniform(-d, d), lon + random.uniform(-d, d)


def new_account(acct_type, city=None, opened_before=None):
    city = city or random.choice(CITIES)
    city_name, lat, lon = city
    lat, lon = jitter_location(lat, lon)
    if opened_before is None:
        opened_before = END_DATE
    account_age_days = random.randint(5, 3650) if acct_type != "mule" else random.randint(1, 120)
    opened_on = opened_before - timedelta(days=account_age_days)
    return {
        "account_id": f"ACC-{uuid.uuid4().hex[:10].upper()}",
        "account_type": acct_type,       # victim | mule | legit | cashout_node
        "holder_name": fake.name(),
        "bank": random.choice(BANKS),
        "branch_city": city_name,
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "kyc_completeness": round(random.uniform(0.2, 1.0), 2) if acct_type == "mule" else round(random.uniform(0.7, 1.0), 2),
        "account_age_days": account_age_days,
        "opened_on": opened_on.date().isoformat(),
        "prior_fraud_flag": acct_type == "mule" and random.random() < 0.35,
    }


# ---------------------------------------------------------------------------
# 1. Background "legit" accounts + noise transactions (non-fraud graph mass)
# ---------------------------------------------------------------------------

accounts = {}
transactions = []

legit_accounts = [new_account("legit") for _ in range(N_LEGIT_ACCOUNTS)]
for a in legit_accounts:
    accounts[a["account_id"]] = a

legit_ids = list(accounts.keys())
for _ in range(N_LEGIT_TXNS):
    a, b = random.sample(legit_ids, 2)
    transactions.append({
        "txn_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
        "from_account": a,
        "to_account": b,
        "amount": round(random.uniform(200, 50000), 2),
        "channel": random.choice(CHANNELS),
        "timestamp": random_time().isoformat(),
        "is_fraud_chain": False,
        "is_cashout": False,
    })

# ---------------------------------------------------------------------------
# 2. Fraud rings: victim -> mule hop -> mule hop -> ... -> cash-out node
# ---------------------------------------------------------------------------

complaints = []
cashouts = []

for ring_idx in range(N_FRAUD_RINGS):
    fraud_type = random.choice(FRAUD_TYPES)
    victim_city = random.choice(CITIES)
    victim = new_account("victim", city=victim_city)
    accounts[victim["account_id"]] = victim

    n_hops = random.randint(MIN_HOPS, MAX_HOPS)
    fraud_start_time = random_time(START_DATE, END_DATE - timedelta(days=2))
    chain_amount = round(random.uniform(15000, 500000), 2)

    # modus-operandi shapes how far (geographically) the chain tends to travel
    if fraud_type in ("Investment fraud", "Digital arrest scam"):
        travel_bias = 0.85   # tends to route far from victim's city
    else:
        travel_bias = 0.5

    prev_account = victim
    chain_accounts = [victim]
    t = fraud_start_time

    for hop in range(n_hops):
        is_last_hop = hop == n_hops - 1
        if is_last_hop:
            # cash-out node: location chosen with travel_bias determining
            # how far from the victim's city it tends to be
            if random.random() < travel_bias:
                cashout_city = random.choice([c for c in CITIES if c[0] != victim_city[0]])
            else:
                cashout_city = victim_city
            node = new_account("cashout_node", city=cashout_city, opened_before=t)
        else:
            mule_city = random.choice(CITIES)
            node = new_account("mule", city=mule_city, opened_before=t)

        accounts[node["account_id"]] = node

        t = t + timedelta(minutes=random.randint(10, 600))
        hop_amount = round(chain_amount * random.uniform(0.75, 0.98), 2)
        chain_amount = hop_amount  # layering typically skims a little each hop

        txn = {
            "txn_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
            "from_account": prev_account["account_id"],
            "to_account": node["account_id"],
            "amount": hop_amount,
            "channel": random.choice(CHANNELS),
            "timestamp": t.isoformat(),
            "is_fraud_chain": True,
            "is_cashout": is_last_hop,
            "ring_id": ring_idx,
        }
        transactions.append(txn)

        if is_last_hop:
            cashouts.append({
                "ring_id": ring_idx,
                "cashout_account": node["account_id"],
                "cashout_txn": txn["txn_id"],
                "cashout_city": cashout_city[0],
                "cashout_latitude": node["latitude"],
                "cashout_longitude": node["longitude"],
                "cashout_timestamp": t.isoformat(),
                "hops_from_victim": n_hops,
                "amount_withdrawn": hop_amount,
            })

        prev_account = node
        chain_accounts.append(node)

    complaint_filed_at = fraud_start_time + timedelta(hours=random.randint(1, 48))
    complaints.append({
        "complaint_id": f"NCRP-{100000 + ring_idx}",
        "ring_id": ring_idx,
        "victim_account": victim["account_id"],
        "victim_city": victim_city[0],
        "fraud_type": fraud_type,
        "reported_loss": chain_accounts[0] and round(random.uniform(15000, 500000), 2),
        "filed_at": complaint_filed_at.isoformat(),
        "fraud_start_at": fraud_start_time.isoformat(),
    })

# ---------------------------------------------------------------------------
# 3. Write outputs
# ---------------------------------------------------------------------------

import os
os.makedirs("output", exist_ok=True)

accounts_df = pd.DataFrame(accounts.values())
transactions_df = pd.DataFrame(transactions)
complaints_df = pd.DataFrame(complaints)
cashouts_df = pd.DataFrame(cashouts)

accounts_df.to_csv("output/accounts.csv", index=False)
transactions_df.to_csv("output/transactions.csv", index=False)
complaints_df.to_csv("output/complaints.csv", index=False)
cashouts_df.to_csv("output/cashouts.csv", index=False)

print(f"accounts:      {len(accounts_df):>6}")
print(f"transactions:  {len(transactions_df):>6}  (fraud-chain: {transactions_df['is_fraud_chain'].sum()})")
print(f"complaints:    {len(complaints_df):>6}")
print(f"cashouts:      {len(cashouts_df):>6}")

# ---------------------------------------------------------------------------
# 4. Visualize one example fraud ring
# ---------------------------------------------------------------------------

example_ring_id = 0
ring_txns = transactions_df[transactions_df.get("ring_id") == example_ring_id]

G = nx.DiGraph()
for _, row in ring_txns.iterrows():
    G.add_edge(row["from_account"], row["to_account"], amount=row["amount"])

pos = nx.spring_layout(G, seed=1, k=1.6)
plt.figure(figsize=(9, 5))

node_colors = []
for n in G.nodes():
    acc_type = accounts[n]["account_type"]
    node_colors.append({
        "victim": "#1E2761",
        "mule": "#CADCFC",
        "cashout_node": "#D85A30",
    }.get(acc_type, "#AAAAAA"))

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, edgecolors="#333333")
nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, edge_color="#5A6178", width=1.6)
labels = {n: accounts[n]["account_type"] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="white")

edge_labels = {(u, v): f"\u20b9{d['amount']:,.0f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title(f"Example fraud ring #{example_ring_id} \u2014 money trail from victim to cash-out")
plt.axis("off")
plt.tight_layout()
plt.savefig("output/example_ring.png", dpi=150)
print("saved output/example_ring.png")
