import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from app import apply_p2p_to_cashflow, simulate_p2p_for_month

# ====== CONFIG ======
DATA_DIR = r"C:\Users\stacy\Downloads\financial_synthetic_dataset\financial_synthetic"
OUT_DIR = os.path.join(DATA_DIR, "graphs")

SHOCK_TYPE = "income_drop"   # <-- change to: "rent_spike", "debt_spike", "expense_volatility", "none"
MAX_USERS = 50               # <-- cap how many users to generate (set None for all)
MONTHS = [1, 6, 12]          # months to render
SEED = 7                     # layout seed for consistent positions
# ====================

os.makedirs(OUT_DIR, exist_ok=True)

users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
monthly = pd.read_csv(os.path.join(DATA_DIR, "monthly_financials.csv"))

NODES = ["User","Income","Savings","Goal","Rent","Food","Transport","Utilities","Debt","Discretionary"]
EXPENSE_MAP = {
    "Rent": "rent_eur",
    "Food": "food_eur",
    "Transport": "transport_eur",
    "Utilities": "utilities_eur",
    "Debt": "debt_eur",
    "Discretionary": "discretionary_eur",
}

def build_financial_graph(users_df, monthly_df, user_id, month, k_peers=3, include_p2p=True, p2p_affects_cashflow=True):
    row = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"] == month)].iloc[0]

    G = nx.DiGraph()

    # Base nodes
    base_nodes = ["User","Income","Savings","Goal","Rent","Food","Transport","Utilities","Debt","Discretionary"]
    G.add_nodes_from(base_nodes)

    # Peer nodes (fixed labels keep feature columns consistent across users)
    peer_nodes = [f"Peer{i}" for i in range(1, int(k_peers) + 1)]
    if include_p2p:
        G.add_nodes_from(peer_nodes)

    # Expense edges
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

    # Start with dataset income/savings
    income_w = float(row["income_eur"])
    savings_w = float(row["savings_eur"])

    # ---- P2P layer (Mobile Money abstraction) ----
    if include_p2p:
        peer_out, peer_in, total_out, total_in = simulate_p2p_for_month(
            users_df, monthly_df, int(user_id), int(month), k=int(k_peers)
        )

        # Outgoing: User -> PeerX
        for peer, amt in peer_out.items():
            G.add_edge("User", peer, weight=float(amt))

        # Incoming: PeerX -> User
        for peer, amt in peer_in.items():
            G.add_edge(peer, "User", weight=float(amt))

        # Optionally let P2P affect income/expenses/savings
        if p2p_affects_cashflow:
            income_adj, expenses_adj, savings_adj = apply_p2p_to_cashflow(row, total_out, total_in)
            income_w = float(income_adj)
            savings_w = float(savings_adj)

    # Income edge (possibly adjusted)
    G.add_edge("Income", "User", weight=float(income_w))

    # Savings edge (possibly adjusted)
    G.add_edge("User", "Savings", weight=float(savings_w))

    # Goal progress edge (kept as-is for MVP)
    goal = float(users_df.loc[users_df["user_id"] == user_id, "goal_amount_eur"].iloc[0])
    progress = float(row["cumulative_savings_eur"]) / goal if goal > 0 else 0.0
    G.add_edge("Savings", "Goal", weight=float(progress))

    return G

def draw_and_save_graph(G, out_path, title):
    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=SEED)

    nx.draw(G, pos, with_labels=True, node_size=2200, arrows=True)

    weights = nx.get_edge_attributes(G, "weight")
    edge_labels = {}
    for (u, v), w in weights.items():
        if (u, v) == ("Savings", "Goal"):
            edge_labels[(u, v)] = f"{w*100:.1f}%"
        else:
            edge_labels[(u, v)] = f"€{w:,.0f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---- Select users by shock type ----
shock_mask = users["shock_type"].astype(str).str.lower() == str(SHOCK_TYPE).lower()
selected_users = users.loc[shock_mask, "user_id"].tolist()

if not selected_users:
    raise ValueError(f"No users found with shock_type='{SHOCK_TYPE}'. Check spelling/case.")

if MAX_USERS is not None:
    selected_users = selected_users[:MAX_USERS]

print(f"Selected {len(selected_users)} users with shock_type='{SHOCK_TYPE}'")

# ---- Generate images ----
count = 0
for uid in selected_users:
    for m in MONTHS:
        G = build_financial_graph(users, monthly, uid, m)
        out_path = os.path.join(OUT_DIR, f"graph_user{uid}_month{m}.png")
        draw_and_save_graph(G, out_path, f"{SHOCK_TYPE} — User {uid} — Month {m}")
        count += 1

print(f"Done. Saved {count} images to: {OUT_DIR}")
