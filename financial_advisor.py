# pandas: data inspection, numpy: numerical checks
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# Load the data
users = pd.read_csv("C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/users.csv")
monthly = pd.read_csv("C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/monthly_financials.csv")
labels = pd.read_csv("C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/labels_goal12.csv")

# First rows
# Users: user_id, goal, debt level, shock type
x_users = users.head(10)
print("USERS: ", x_users)
# Monthly: user_id, month, income/expenses/savings columns
x_monthly = monthly.head(10)
print("MONTHLY: ", x_monthly)
# Labels: user_id, will_reach_goal
x_labels = labels.head(10)
print("LABELS: ", x_labels)

# Confirm dataset size (sanity check)
print("Users rows: ", users.shape)
print("Monthly rows: ", monthly.shape)
print("Labels rows: ", labels.shape)

# Check columns & data types
print(users.dtypes)
print(monthly.dtypes)
print(labels.dtypes)

# Check missing values
print("Missing values in users:\n", users.isna().sum())
print("Missing values in monthly:\n", monthly.isna().sum())
print("Missing values in labels:\n", labels.isna().sum())

# Check duplicate keys
# Users should have one row per user
dup_users = users["user_id"].duplicated().sum()
print("Duplicate users: ", dup_users)
# Labels should have one row per user
dup_labels = labels["user_id"].duplicated().sum()
print("Duplicate users: ", dup_labels)
# Monthly should have one row per (user_id, month)
dup_users = monthly.duplicated(subset=["user_id", "month"]).sum()
print("Duplicates monthly ", dup_users)

# Check month coverage (are all months present?)
print("Unique months:", sorted(monthly["month"].unique()))

# Check each user has 12 months
months_per_user = monthly.groupby("user_id")["month"].nunique()
print("Min months per user:", months_per_user.min())
print("Max months per user:", months_per_user.max())

# Basic summary stats
print(monthly.describe())

# No negative money values
money_cols = ["income_eur","rent_eur","food_eur","transport_eur","utilities_eur",
              "debt_eur","discretionary_eur","total_expenses_eur","savings_eur","cumulative_savings_eur"]
print((monthly[money_cols] < 0).sum())

# Total expenses should approximately equal sum of categories
cat_sum = (
    monthly["rent_eur"] + monthly["food_eur"] + monthly["transport_eur"] +
    monthly["utilities_eur"] + monthly["debt_eur"] + monthly["discretionary_eur"]
)

diff = (monthly["total_expenses_eur"] - cat_sum).abs()
print("Mean abs diff:", diff.mean())
print("Max abs diff:", diff.max())

# Savings should be income - expenses (or 0 if overspend)
expected_savings = (monthly["income_eur"] - monthly["total_expenses_eur"]).clip(lower=0)
savings_diff = (monthly["savings_eur"] - expected_savings).abs()

print("Mean abs savings diff:", savings_diff.mean())
print("Max abs savings diff:", savings_diff.max())

# How many users reach their goal: 1 means reached goal, 0 did not reach goal, mean of 0/1 = % reached
counts = labels["will_reach_goal"].value_counts()
print("Counts:\n", counts)

reach_rate = labels["will_reach_goal"].mean()
print("Reach rate:", reach_rate)

# Which shock types fail more: this requires merging user shock info with labels
shock_perf = users.merge(labels[["user_id","will_reach_goal"]], on="user_id", how="inner")
shock_summary = (
    shock_perf.groupby("shock_type")
    .agg(
        users=("user_id","count"),
        reach_rate=("will_reach_goal","mean")
    )
    .reset_index()
)
shock_summary["fail_rate"] = 1 - shock_summary["reach_rate"]
print(shock_summary.sort_values("fail_rate", ascending=False))

# How volatile is income vs expenses: volatile here= standard deviation across months per user, then summarize
vol = (
    monthly.groupby("user_id")
    .agg(
        income_std=("income_eur","std"),
        expenses_std=("total_expenses_eur","std"),
    )
    .reset_index()
)
print("Income volatility (mean, median):", vol["income_std"].mean(), vol["income_std"].median())
print("Expense volatility (mean, median):", vol["expenses_std"].mean(), vol["expenses_std"].median())
vol2 = vol.merge(users[["user_id","shock_type"]], on="user_id")
print(
    vol2.groupby("shock_type")[["income_std","expenses_std"]]
    .mean()
    .sort_values("income_std", ascending=False)
)

# Save your inspection results (for your README)
shock_summary.sort_values("fail_rate", ascending=False).to_csv("C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/shock_summary.csv", index=False)
vol.to_csv("C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/volatility_per_user.csv", index=False)

# Build your first graph (Single user, Single month)
# Select one user and one month
def build_financial_graph(users_df, monthly_df, user_id, month):
    # select the row for this user and month
    row = monthly_df[
        (monthly_df["user_id"] == user_id) &
        (monthly_df["month"] == month)
    ].iloc[0]

    # create graph instance
    G = nx.DiGraph()

    # add nodes
    nodes = [
        "User", "Income", "Savings", "Goal",
        "Rent", "Food", "Transport", "Utilities",
        "Debt", "Discretionary"
    ]
    G.add_nodes_from(nodes)

    # income edge
    G.add_edge("Income", "User", weight=float(row["income_eur"]))

    # expense edges
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

    # savings edge
    G.add_edge("User", "Savings", weight=float(row["savings_eur"]))

    # goal progress edge
    goal = float(users_df.loc[users_df["user_id"] == user_id, "goal_amount_eur"].iloc[0])
    progress = float(row["cumulative_savings_eur"]) / goal if goal > 0 else 0.0
    G.add_edge("Savings", "Goal", weight=progress)

    # Useful metadata (optional)
    G.graph['user_id'] = user_id
    G.graph['month'] = month

    return G

# Draw a graph snapshot
def draw_graph(G: nx.DiGraph, title: str):
    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=7)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=2200, arrows=True)

    # Edge labels (euros except progress edge)
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
    # save images instead of showing
    plt.savefig(f"graph_user{G.graph['user_id']}_month{G.graph['month']}.png", dpi=200)
    plt.close()

# Pick a user and build graphs for ALL months (graph evolution)
user_id = 1  # change this to any user_id 1..800

# Print user profile info (useful)
urow = users.loc[users["user_id"] == user_id].iloc[0]
print("\nUser profile:")
print(urow[["user_id", "base_income_eur", "income_growth_rate_annual", "saving_propensity",
           "debt_level", "goal_amount_eur", "shock_type", "shock_start_month"]])

graphs = {}
for m in range(1, 13):
    graphs[m] = build_financial_graph(users, monthly, user_id, m)

print(f"\nBuilt {len(graphs)} graph snapshots for user {user_id} (months 1..12).")

# Inspect edges for a few months
for m in [1, 6, 12]:
    print(f"\n--- Month {m} edges (with weights) ---")
    for u, v, d in graphs[m].edges(data=True):
        print(f"{u:>12} -> {v:<14}  weight={d['weight']:.4f}")

# Visualize graphs for months 1, 6, 12
for m in [1, 6, 12]:
    draw_graph(graphs[m], f"Financial Graph Snapshot — User {user_id}, Month {m}")

# Convert your dynamic graphs into an "edge time series" table
# -------------------------
# A) Build edge-weight time series table for one user
# -------------------------
def graphs_to_edge_timeseries(graphs_dict):
    rows = []
    for m in sorted(graphs_dict.keys()):
        G = graphs_dict[m]
        for u, v, d in G.edges(data=True):
            rows.append({"month": m, "edge": f"{u}->{v}", "weight": float(d.get("weight", 0.0))})
    return pd.DataFrame(rows)

edge_ts = graphs_to_edge_timeseries(graphs)

print(edge_ts.head(15))
print("\nUnique edges:", edge_ts["edge"].nunique())

# Pivot into a "wide" table (edges as columns): perfect for analyzing evolution
edge_wide = (
    edge_ts.pivot_table(index="month", columns="edge", values="weight", aggfunc="mean")
    .sort_index()
)
print(edge_wide.head())

# Compute graph evolution: month-to-month changes (deltas)
edge_delta = edge_wide.diff()  # month t - month (t-1)
# Tell which financial relationships matter most over time
abs_change = edge_delta.abs().sum().sort_values(ascending=False)
print(abs_change.head(10))

# Find the biggest changes (Top "drivers" of evolution)
# Biggest absolute changes over the whole year
abs_change = edge_delta.abs().sum().sort_values(ascending=False)
print("\nTop 10 edges by total change over time:")
print(abs_change.head(10))

# Biggest changes in a specific month
m = 6
big_month = edge_delta.loc[m].abs().sort_values(ascending=False)
print(f"\nTop 10 changes at month {m}:")
print(big_month.head(10))

# Generate a human-readable "what changed?" summary (mini explanation)
def summarize_month_changes(edge_delta_df, month, top_k=5):
    if month not in edge_delta_df.index:
        return f"No delta info for month {month}"

    changes = edge_delta_df.loc[month].dropna()
    changes = changes.reindex(changes.abs().sort_values(ascending=False).index)

    lines = []
    for edge, delta in changes.head(top_k).items():
        direction = "increased" if delta > 0 else "decreased"
        if "Savings->Goal" in edge:
            lines.append(f"- {edge} {direction} by {delta*100:.2f} percentage points")
        else:
            lines.append(f"- {edge} {direction} by €{abs(delta):,.0f}")
    return "\n".join(lines)

print("\nMonth 6 changes summary:")
print(summarize_month_changes(edge_delta, 6, top_k=6))

print("\nMonth 12 changes summary:")
print(summarize_month_changes(edge_delta, 12, top_k=6))

def build_user_graph_sequence(users, monthly, user_id, months=range(1,7)):
    return {m: build_financial_graph(users, monthly, user_id, m) for m in months}

# Extract edge time series for one user
def edge_timeseries_from_graphs(graphs):
    rows = []
    for m, G in graphs.items():
        for u, v, d in G.edges(data=True):
            rows.append({
                "month": m,
                "edge": f"{u}->{v}",
                "weight": d["weight"]
            })
    return pd.DataFrame(rows)

# Compute features from edge time series
def extract_graph_features(edge_ts):
    feats = {}
    for edge, g in edge_ts.groupby("edge"):
        weights = g.sort_values("month")["weight"].values
        feats[f"{edge}_mean"] = weights.mean()
        feats[f"{edge}_std"] = weights.std()
        feats[f"{edge}_trend"] = np.polyfit(range(len(weights)), weights, 1)[0]
    return feats

# Building the modeling dataset
X_rows = []
y = []

for user_id in users["user_id"]:
    graphs_u = build_user_graph_sequence(users, monthly, user_id)
    edge_ts = edge_timeseries_from_graphs(graphs_u)
    feats = extract_graph_features(edge_ts)
    X_rows.append(feats)
    y.append(labels.loc[labels["user_id"] == user_id, "will_reach_goal"].iloc[0])

X = pd.DataFrame(X_rows).fillna(0)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Target distribution:", np.bincount(y))
# you now have x: graph-based feature matrix, and y: prediction target

# Train an explainable model (logistic regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Interpret the model: tells you which graph relationships most influence success or failure
coef = pd.Series(model.coef_[0], index=X.columns)
top_features = coef.reindex(coef.abs().sort_values(ascending=False).index)

print("\nTop 10 most influential graph features:")
print(top_features.head(10))

# Save results
top_features.head(15).to_csv(
    "C:/Users/stacy/Downloads/financial_synthetic_dataset/financial_synthetic/top_graph_features.csv"
)

# Retrain model multiple times
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

coef_list = []

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    m = LogisticRegression(max_iter=3000)
    m.fit(X_tr, y_tr)

    coef_list.append(pd.Series(m.coef_[0], index=X.columns))

coef_df = pd.concat(coef_list, axis=1)
coef_df.columns = [f"fold_{i}" for i in range(1, 6)]

# Analyze stability
coef_df["mean_abs"] = coef_df.abs().mean(axis=1)
coef_df["std"] = coef_df.std(axis=1)

stable_features = coef_df.sort_values("mean_abs", ascending=False).head(10)
print(stable_features[["mean_abs", "std"]])

# Simple counterfactual experiment: take one failing user and reduce rent pressure
uid = users[users["shock_type"] == "rent_spike"]["user_id"].iloc[0]

graphs_u = build_user_graph_sequence(users, monthly, uid)
edge_ts = edge_timeseries_from_graphs(graphs_u)
feats = extract_graph_features(edge_ts)

x_orig = pd.DataFrame([feats])

# Counterfactual: reduce rent by 20%
cf = x_orig.copy()
for col in cf.columns:
    if "User->Rent_mean" in col:
        cf[col] *= 0.8

print("Original prediction:", model.predict_proba(x_orig)[0][1])
print("Counterfactual prediction:", model.predict_proba(cf)[0][1])

#Failure case analysis: Pick users the model got wrong
errors = X_test.copy()
errors["y_true"] = y_test
errors["y_pred"] = y_pred

false_negatives = errors[(errors["y_true"] == 1) & (errors["y_pred"] == 0)]
false_positives = errors[(errors["y_true"] == 0) & (errors["y_pred"] == 1)]

print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))