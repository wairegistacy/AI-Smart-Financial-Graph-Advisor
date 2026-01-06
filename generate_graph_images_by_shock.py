import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

def build_financial_graph(users_df, monthly_df, user_id, month):
    row = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"] == month)]
    if row.empty:
        raise ValueError(f"No row for user_id={user_id}, month={month}")
    row = row.iloc[0]

    G = nx.DiGraph()
    G.add_nodes_from(NODES)

    G.add_edge("Income", "User", weight=float(row["income_eur"]))

    for node, col in EXPENSE_MAP.items():
        G.add_edge("User", node, weight=float(row[col]))

    G.add_edge("User", "Savings", weight=float(row["savings_eur"]))

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
