import os
import numpy as np
import pandas as pd
import networkx as nx
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = r"C:\Users\stacy\Downloads\financial_synthetic_dataset\financial_synthetic"
OUT_DIR = DATA_DIR  # save artifacts next to the CSVs

users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
monthly = pd.read_csv(os.path.join(DATA_DIR, "monthly_financials.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "labels_goal12.csv"))

# ---------- P2P (Mobile Money) Layer Helpers ----------
def pick_peer_ids(users_df, user_id: int, k: int = 3):
    """
    Deterministically pick k peer user_ids for each user_id,
    so the same peers are used across all months.
    """
    all_ids = users_df["user_id"].tolist()
    all_ids = [i for i in all_ids if int(i) != int(user_id)]
    rng = np.random.RandomState(int(user_id))  # deterministic
    peers = rng.choice(all_ids, size=k, replace=False)
    return [int(p) for p in peers]

def apply_p2p_to_cashflow(row, total_out: float, total_in: float):
    """
    Optionally make P2P transfers affect the user’s cashflow:
      - incoming transfers increase effective income
      - outgoing transfers increase effective expenses (or reduce discretionary)
    Returns adjusted income, adjusted expenses, adjusted savings.
    """
    income = float(row["income_eur"])
    expenses = float(row["total_expenses_eur"])

    income_adj = income + total_in
    expenses_adj = expenses + total_out

    savings_adj = max(0.0, income_adj - expenses_adj)
    return income_adj, expenses_adj, savings_adj
# ------------------------------------------------------
def simulate_p2p_for_month(users_df, monthly_df, user_id: int, month: int, k: int = 3):
    """
    Simulate P2P transfers for one user-month.
    Returns:
      peer_amounts_out: dict {Peer1..PeerK: amount_sent}
      peer_amounts_in:  dict {Peer1..PeerK: amount_received}
      total_out, total_in
    Notes:
      - Deterministic (repeatable) using seeded RNG.
      - Makes transfers more dramatic during shock months.
    """
    # Base row
    row = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"] == month)].iloc[0]
    income = float(row["income_eur"])
    discretionary = float(row.get("discretionary_eur", 0.0))

    # User shock info (if present)
    u = users_df.loc[users_df["user_id"] == user_id].iloc[0]
    shock_type = str(u.get("shock_type", "none")).lower()
    shock_start = int(u.get("shock_start_month", 99)) if not np.isnan(u.get("shock_start_month", 99)) else 99

    shock_active = (month >= shock_start) and (shock_type != "none")

    # Deterministic RNG per (user, month)
    rng = np.random.RandomState(int(user_id) * 1000 + int(month))

    # How much total out/in? (simple but plausible)
    # Outgoing: small fraction of income, capped by discretionary
    out_frac = rng.uniform(0.00, 0.05)  # 0–5% of income
    total_out = min(discretionary * rng.uniform(0.2, 0.8), income * out_frac)

    # Incoming: usually smaller; increases if shock_active (support network)
    in_base = rng.uniform(0.00, 0.03) * income
    if shock_active:
        # “Friends/family help” effect during shock
        in_base *= rng.uniform(1.5, 3.0)
    total_in = in_base

    # Split totals across K peers (Dirichlet proportions)
    k = int(k)
    out_parts = rng.dirichlet(alpha=np.ones(k)) * total_out
    in_parts = rng.dirichlet(alpha=np.ones(k)) * total_in

    peer_amounts_out = {f"Peer{i+1}": float(out_parts[i]) for i in range(k)}
    peer_amounts_in  = {f"Peer{i+1}": float(in_parts[i])  for i in range(k)}

    return peer_amounts_out, peer_amounts_in, float(total_out), float(total_in)


def build_financial_graph(
    users_df, monthly_df, user_id, month,
    k_peers=3, include_p2p=True, p2p_affects_cashflow=True,
    include_bank=True
):
    row = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"] == month)].iloc[0]
    G = nx.DiGraph()

    # --- Nodes ---
    base_nodes = ["User","Income","Savings","Goal","Rent","Food","Transport","Utilities","Debt","Discretionary"]
    G.add_nodes_from(base_nodes)

    if include_bank:
        G.add_node("Bank")

    peer_nodes = [f"Peer{i}" for i in range(1, int(k_peers) + 1)]
    if include_p2p:
        G.add_nodes_from(peer_nodes)

    # --- Expenses by category (keep for explanations) ---
    expense_map = {
        "Rent": "rent_eur",
        "Food": "food_eur",
        "Transport": "transport_eur",
        "Utilities": "utilities_eur",
        "Debt": "debt_eur",
        "Discretionary": "discretionary_eur"
    }
    for node, col in expense_map.items():
        G.add_edge("User", node, weight=float(row[col]))

    # Starting cashflow
    income_w = float(row["income_eur"])
    savings_w = float(row["savings_eur"])
    total_out = 0.0
    total_in = 0.0

    # --- P2P Layer ---
    if include_p2p:
        peer_out, peer_in, total_out, total_in = simulate_p2p_for_month(
            users_df, monthly_df, int(user_id), int(month), k=int(k_peers)
        )

        for peer, amt in peer_out.items():
            G.add_edge("User", peer, weight=float(amt))
        for peer, amt in peer_in.items():
            G.add_edge(peer, "User", weight=float(amt))

        if p2p_affects_cashflow:
            income_adj, expenses_adj, savings_adj = apply_p2p_to_cashflow(row, total_out, total_in)
            income_w = float(income_adj)
            savings_w = float(savings_adj)

    # --- Banking abstraction ---
    if include_bank:
        # Income arrives to bank account
        G.add_edge("Income", "Bank", weight=float(income_w))
        # Bank makes funds available to user
        G.add_edge("Bank", "User", weight=float(income_w))

        # Aggregate outgoing payments leaving the account
        # (category edges remain for explanation)
        total_exp = float(row["total_expenses_eur"]) + float(total_out)  # include p2p out as money leaving user
        G.add_edge("User", "Bank", weight=float(total_exp))
    else:
        # Original simpler model
        G.add_edge("Income", "User", weight=float(income_w))

    # Savings edge (possibly adjusted)
    G.add_edge("User", "Savings", weight=float(savings_w))

    # Optional: savings stored in bank
    if include_bank:
        G.add_edge("Savings", "Bank", weight=float(savings_w))

    # Goal progress (kept as-is for MVP)
    goal = float(users_df.loc[users_df["user_id"] == user_id, "goal_amount_eur"].iloc[0])
    progress = float(row["cumulative_savings_eur"]) / goal if goal > 0 else 0.0
    G.add_edge("Savings", "Goal", weight=float(progress))

    return G

def build_user_graph_sequence(users_df, monthly_df, user_id, months=range(1, 7)):
    return {m: build_financial_graph(users_df, monthly_df, user_id, m) for m in months}

def edge_timeseries_from_graphs(graphs):
    rows = []
    for m in sorted(graphs.keys()):
        G = graphs[m]
        for u, v, d in G.edges(data=True):
            rows.append({"month": m, "edge": f"{u}->{v}", "weight": float(d["weight"])})
    return pd.DataFrame(rows)

def extract_graph_features(edge_ts):
    feats = {}
    for edge, g in edge_ts.groupby("edge"):
        g = g.sort_values("month")
        w = g["weight"].values
        feats[f"{edge}_mean"] = float(np.mean(w))
        feats[f"{edge}_std"] = float(np.std(w))
        feats[f"{edge}_trend"] = float(np.polyfit(np.arange(len(w)), w, 1)[0])
    return feats

# Build dataset
X_rows = []
y = []

for uid in users["user_id"].tolist():
    graphs_u = build_user_graph_sequence(users, monthly, uid, months=range(1, 7))
    edge_ts = edge_timeseries_from_graphs(graphs_u)
    feats = extract_graph_features(edge_ts)
    X_rows.append(feats)
    y.append(int(labels.loc[labels["user_id"] == uid, "will_reach_goal"].iloc[0]))

X = pd.DataFrame(X_rows).fillna(0.0)
y = np.array(y)

feature_cols = X.columns.tolist()

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, os.path.join(OUT_DIR, "model.joblib"))
joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_cols.joblib"))

print("Saved:", os.path.join(OUT_DIR, "model.joblib"))
print("Saved:", os.path.join(OUT_DIR, "feature_cols.joblib"))
