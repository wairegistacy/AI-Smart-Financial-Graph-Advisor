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

def build_financial_graph(users_df, monthly_df, user_id, month):
    row = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"] == month)].iloc[0]

    G = nx.DiGraph()
    nodes = ["User","Income","Savings","Goal","Rent","Food","Transport","Utilities","Debt","Discretionary"]
    G.add_nodes_from(nodes)

    G.add_edge("Income", "User", weight=float(row["income_eur"]))

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

    G.add_edge("User", "Savings", weight=float(row["savings_eur"]))

    goal = float(users_df.loc[users_df["user_id"] == user_id, "goal_amount_eur"].iloc[0])
    progress = float(row["cumulative_savings_eur"]) / goal if goal > 0 else 0.0
    G.add_edge("Savings", "Goal", weight=progress)

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
