"""
Synthetic dataset generator matching the schema of the real dataset used
for training: IBM "Transactions for Anti-Money-Laundering (AML)" on
Kaggle (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

This does NOT reproduce real data - it generates new, fake transactions,
accounts, and laundering patterns, but in the exact file/column layout
that train_model_real.py expects, so the same training script can run
unmodified against either the real Kaggle files or this synthetic
stand-in (useful for development/demo without needing the real dataset
downloaded, and for testing changes to the pipeline safely).

Outputs written to ./kaggle_data (same directory train_model_real.py
reads from by default):

  HI-Small_Trans.csv      - every transaction (legit + laundering), columns:
                             Timestamp, From Bank, From Account, To Bank,
                             To Account, Amount Received, Receiving Currency,
                             Amount Paid, Payment Currency, Payment Format,
                             Is Laundering
  HI-Small_accounts.csv   - one row per account: Bank ID, Account Number,
                             Bank Name, Country (Country is extra context,
                             harmless - train_model_real.py only reads the
                             first three)
  HI-Small_Patterns.txt   - BEGIN/END LAUNDERING ATTEMPT blocks, each
                             listing the transaction rows (same 11 fields,
                             comma-separated, no header) that make up that
                             laundering chain - this is what
                             train_model_real.py parses for ground truth

Money-trail structure is unchanged from the original generator: victim
account -> mule hop -> mule hop -> ... -> cash-out (terminal) account,
where only the terminal account receives without forwarding the money
onward within that laundering attempt. That's exactly the pattern
train_model_real.py's ground-truth parser looks for (received but never
sent within the same attempt = cash-out / terminal node).

Run:
  pip install pandas faker
  python3 generate.py
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

random.seed(42)
fake = Faker("en_US")
Faker.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = os.environ.get("OUT_DIR", "kaggle_data")
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "HI-Small")

N_BANKS = 250
N_LEGIT_ACCOUNTS = 1800
N_LEGIT_TXNS = 9000
N_LAUNDERING_ATTEMPTS = 90     # each = one fraud ring / one complaint's money trail
MIN_HOPS, MAX_HOPS = 3, 6
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 8, 31)

# Same country list train_model_real.py's bank_to_country() recognizes,
# so the hotspot model can actually geo-locate these accounts later.
COUNTRIES = [
    "germany", "switzerland", "china", "france", "india", "israel", "uk",
    "italy", "japan", "spain", "australia", "canada", "russia", "mexico",
    "saudi arabia", "netherlands", "brazil", "belgium", "austria",
    "greece", "portugal", "ireland",
]
# usa is handled separately (bank_to_country() detects it from phrases
# like "First Bank of X" / "National Bank of X" rather than a country name)
USA_WEIGHT = 0.30       # fraction of non-crypto banks that are US-style
CRYPTO_WEIGHT = 0.03    # fraction of banks that are "Crytpo Bank" (sic -
                         # this typo is a real quirk of the source dataset,
                         # kept here on purpose for schema fidelity)

CURRENCIES = [
    "US Dollar", "Euro", "Yuan", "Yen", "UK Pound", "Rupee", "Ruble",
    "Swiss Franc", "Australian Dollar", "Canadian Dollar", "Mexican Peso",
    "Brazil Real", "Saudi Riyal", "Shekel", "Bitcoin",
]
PAYMENT_FORMATS = ["ACH", "Wire", "Credit Card", "Cheque", "Cash", "Reinvestment", "Bitcoin"]

FRAUD_TYPES = [
    "Investment fraud", "Digital arrest scam", "KYC update fraud",
    "Loan app fraud", "Job/task fraud", "OTP/phishing fraud",
]


def random_time(start=START_DATE, end=END_DATE):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def fmt_ts(dt):
    return dt.strftime("%Y/%m/%d %H:%M")


# ---------------------------------------------------------------------------
# 1. Bank pool - fixed (bank_id, bank_name, country) triples, matching the
#    real dataset's Bank Name conventions closely enough for
#    bank_to_country() in train_model_real.py to classify correctly.
# ---------------------------------------------------------------------------

US_CITIES = [fake.city() for _ in range(40)]
US_PATTERNS = [
    "First Bank of {city}", "National Bank of {city}",
    "{city} Savings Bank", "{city} Cooperative Bank",
]


def make_bank(bank_id):
    roll = random.random()
    if roll < CRYPTO_WEIGHT:
        return {"bank_id": bank_id, "bank_name": "Crytpo Bank", "country": None}
    if roll < CRYPTO_WEIGHT + USA_WEIGHT:
        city = random.choice(US_CITIES)
        pattern = random.choice(US_PATTERNS)
        return {"bank_id": bank_id, "bank_name": pattern.format(city=city), "country": "usa"}
    country = random.choice(COUNTRIES)
    return {"bank_id": bank_id, "bank_name": f"{country.title()} Bank #{bank_id}", "country": country}


banks = [make_bank(1000 + i) for i in range(N_BANKS)]


def random_bank():
    return random.choice(banks)


def new_account_number():
    return uuid.uuid4().hex[:9].upper()


# ---------------------------------------------------------------------------
# 2. Legit background accounts + noise transactions
# ---------------------------------------------------------------------------

account_rows = []      # -> HI-Small_accounts.csv
account_bank = {}      # account_number -> bank dict, for quick lookup

legit_accounts = []
for _ in range(N_LEGIT_ACCOUNTS):
    bank = random_bank()
    acc_num = new_account_number()
    legit_accounts.append((bank["bank_id"], acc_num))
    account_bank[acc_num] = bank
    account_rows.append({
        "Bank ID": bank["bank_id"], "Account Number": acc_num,
        "Bank Name": bank["bank_name"], "Country": bank["country"],
    })

# Hard negatives: a chunk of legit accounts that only ever receive (never
# send), same trait real cash-out nodes have - stops the model from
# learning "out_degree == 0" as a free shortcut.
n_receive_only = max(1, int(N_LEGIT_ACCOUNTS * 0.12))
receive_only = set(random.sample(range(len(legit_accounts)), n_receive_only))
sender_pool = [a for i, a in enumerate(legit_accounts) if i not in receive_only]

trans_rows = []  # -> HI-Small_Trans.csv


def make_txn(t, from_bank, from_acc, to_bank, to_acc, amount, is_laundering):
    currency = random.choice(CURRENCIES)
    payment_format = "Bitcoin" if currency == "Bitcoin" else random.choice(PAYMENT_FORMATS)
    return {
        "Timestamp": fmt_ts(t),
        "From Bank": from_bank,
        "From Account": from_acc,
        "To Bank": to_bank,
        "To Account": to_acc,
        "Amount Received": amount,
        "Receiving Currency": currency,
        "Amount Paid": amount,
        "Payment Currency": currency,
        "Payment Format": payment_format,
        "Is Laundering": int(is_laundering),
    }


for _ in range(N_LEGIT_TXNS):
    from_bank, from_acc = random.choice(sender_pool)
    to_bank, to_acc = random.choice(legit_accounts)
    if from_acc == to_acc:
        continue
    trans_rows.append(make_txn(
        random_time(), from_bank, from_acc, to_bank, to_acc,
        round(random.uniform(50, 20000), 2), is_laundering=False,
    ))

# Accounts that will later be used as mule/cash-out nodes in a laundering
# chain, reserved now so we can also route some ordinary-looking noise
# transactions through them below. Real mule accounts usually have some
# mundane transaction history too - without this, "total transaction
# count" alone trivially separates fraud accounts from everything else,
# which makes the detection problem unrealistically easy.
n_fraud_accounts_estimate = N_LAUNDERING_ATTEMPTS * (MAX_HOPS + 1)
pre_seeded_fraud_accounts = []
for _ in range(n_fraud_accounts_estimate):
    bank = random_bank()
    acc = new_account_number()
    account_bank[acc] = bank
    account_rows.append({
        "Bank ID": bank["bank_id"], "Account Number": acc,
        "Bank Name": bank["bank_name"], "Country": bank["country"],
    })
    pre_seeded_fraud_accounts.append((bank["bank_id"], acc))

noise_pool = legit_accounts + pre_seeded_fraud_accounts
n_mixed_noise_txns = int(N_LEGIT_TXNS * 0.35)
for _ in range(n_mixed_noise_txns):
    from_bank, from_acc = random.choice(noise_pool)
    to_bank, to_acc = random.choice(noise_pool)
    if from_acc == to_acc:
        continue
    trans_rows.append(make_txn(
        random_time(), from_bank, from_acc, to_bank, to_acc,
        round(random.uniform(50, 20000), 2), is_laundering=False,
    ))

fraud_account_cursor = 0

# ---------------------------------------------------------------------------
# 3. Laundering attempts (fraud rings): victim -> mule -> ... -> cash-out
# ---------------------------------------------------------------------------

patterns_lines = []


def txn_to_pattern_row(txn):
    # Same 11 fields, comma-separated, no header - exactly what
    # train_model_real.py's Patterns.txt parser expects per line.
    return ",".join(str(txn[k]) for k in [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering",
    ])


complaint_like_rows = []  # small extra summary file, handy for a dashboard demo

for ring_idx in range(N_LAUNDERING_ATTEMPTS):
    fraud_type = random.choice(FRAUD_TYPES)

    victim_bank = random_bank()
    victim_acc = new_account_number()
    account_bank[victim_acc] = victim_bank
    account_rows.append({
        "Bank ID": victim_bank["bank_id"], "Account Number": victim_acc,
        "Bank Name": victim_bank["bank_name"], "Country": victim_bank["country"],
    })

    n_hops = random.randint(MIN_HOPS, MAX_HOPS)
    t = random_time(START_DATE, END_DATE - timedelta(days=2))
    chain_amount = round(random.uniform(15000, 500000), 2)

    prev_bank, prev_acc = victim_bank["bank_id"], victim_acc
    ring_txn_rows = []

    for hop in range(n_hops):
        is_last_hop = hop == n_hops - 1
        bank_id, acc = pre_seeded_fraud_accounts[fraud_account_cursor]
        bank = account_bank[acc]
        fraud_account_cursor += 1

        t = t + timedelta(minutes=random.randint(10, 600))
        chain_amount = round(chain_amount * random.uniform(0.75, 0.98), 2)

        txn = make_txn(t, prev_bank, prev_acc, bank["bank_id"], acc, chain_amount, is_laundering=True)
        trans_rows.append(txn)
        ring_txn_rows.append(txn)

        if is_last_hop:
            complaint_like_rows.append({
                "complaint_id": f"NCRP-{100000 + ring_idx}",
                "ring_id": ring_idx,
                "fraud_type": fraud_type,
                "victim_bank_id": victim_bank["bank_id"],
                "victim_account": victim_acc,
                "cashout_bank_id": bank["bank_id"],
                "cashout_account": acc,
                "cashout_bank_name": bank["bank_name"],
                "cashout_country": bank["country"],
                "amount_withdrawn": chain_amount,
                "hops": n_hops,
                "filed_at": fmt_ts(t + timedelta(hours=random.randint(1, 48))),
            })

        prev_bank, prev_acc = bank["bank_id"], acc

    patterns_lines.append(f"BEGIN LAUNDERING ATTEMPT - STACK")
    for txn in ring_txn_rows:
        patterns_lines.append(txn_to_pattern_row(txn))
    patterns_lines.append(f"END LAUNDERING ATTEMPT - STACK")

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

trans_df = pd.DataFrame(trans_rows)[[
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]]
trans_path = f"{OUT_DIR}/{DATASET_PREFIX}_Trans.csv"
trans_df.to_csv(trans_path, index=False)

accounts_df = pd.DataFrame(account_rows).drop_duplicates(subset=["Bank ID", "Account Number"])
accounts_path = f"{OUT_DIR}/{DATASET_PREFIX}_accounts.csv"
accounts_df.to_csv(accounts_path, index=False)

patterns_path = f"{OUT_DIR}/{DATASET_PREFIX}_Patterns.txt"
with open(patterns_path, "w") as f:
    f.write("\n".join(patterns_lines) + "\n")

complaints_df = pd.DataFrame(complaint_like_rows)
complaints_path = f"{OUT_DIR}/{DATASET_PREFIX}_complaints_demo.csv"
complaints_df.to_csv(complaints_path, index=False)

print(f"wrote {trans_path}       ({len(trans_df):,} transactions, "
      f"{trans_df['Is Laundering'].sum():,} laundering)")
print(f"wrote {accounts_path}    ({len(accounts_df):,} accounts)")
print(f"wrote {patterns_path}    ({N_LAUNDERING_ATTEMPTS} laundering attempts)")
print(f"wrote {complaints_path}  (demo/dashboard summary, not read by train_model_real.py)")
