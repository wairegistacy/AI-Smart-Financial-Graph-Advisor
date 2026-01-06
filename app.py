import os
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import streamlit as st

#CONFIG
DATA_DIR = r"C:\Users\stacy\Downloads\financial_synthetic_dataset\financial_synthetic"
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")

# LOAD DATA + ARTIFACTS
users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
monthly = pd.read_csv(os.path.join(DATA_DIR, "monthly_financials.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "labels_goal12.csv"))

model = joblib.load(os.path.join(DATA_DIR, "model.joblib"))
feature_cols = joblib.load(os.path.join(DATA_DIR, "feature_cols.joblib"))

# -------------------------
# Graph + feature functions (same as training)
# -------------------------
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

def apply_whatif_to_graphs(graphs, rent_reduce_pct=0, debt_reduce_pct=0, disc_reduce_pct=0):
    """
    Counterfactual: reduce certain edges (User->Rent, User->Debt, User->Discretionary)
    by a percent for ALL months in the observed window.
    """
    def scale_edge(G, u, v, pct):
        if G.has_edge(u, v) and pct > 0:
            w = float(G[u][v]["weight"])
            G[u][v]["weight"] = w * (1 - pct / 100.0)

    new_graphs = {}
    for m, G in graphs.items():
        H = G.copy()
        scale_edge(H, "User", "Rent", rent_reduce_pct)
        scale_edge(H, "User", "Debt", debt_reduce_pct)
        scale_edge(H, "User", "Discretionary", disc_reduce_pct)

        # If you reduce expenses, savings should increase (simple consistency step)
        # Recompute User->Savings edge as max(income - total_expenses, 0)
        income = float(H["Income"]["User"]["weight"])
        expenses = sum(float(H["User"][cat]["weight"]) for cat in ["Rent","Food","Transport","Utilities","Debt","Discretionary"])
        savings = max(income - expenses, 0.0)
        H["User"]["Savings"]["weight"] = savings

        # Note: Savings->Goal (progress) is still based on original cumulative savings in the dataset.
        # For MVP demo purposes, we keep it. In Day 6+ you can simulate forward cumulative savings.
        new_graphs[m] = H

    return new_graphs

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

def align_features(feats_dict, feature_cols):
    row = pd.DataFrame([feats_dict])
    for c in feature_cols:
        if c not in row.columns:
            row[c] = 0.0
    return row[feature_cols].fillna(0.0)

def top_contributors_lr(model, X_row, top_k=8):
    coefs = model.coef_[0]
    contrib = coefs * X_row.values[0]   # approximate contribution
    s = pd.Series(contrib, index=X_row.columns)
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return s.head(top_k)

def recommend_actions(drivers: pd.Series):
    """
    drivers: contribution series (positive helps success, negative hurts success)
    We'll look for the most negative drivers and propose actions.
    """
    neg = drivers[drivers < 0]
    actions = []

    # Heuristic mapping from feature names to user actions
    for feat, val in neg.head(6).items():
        name = feat.lower()

        if "user->rent" in name:
            actions.append(("Reduce housing pressure",
                            "Try renegotiating rent, moving, or getting a housemate. Even a 5–10% drop helps."))
        elif "user->debt" in name:
            actions.append(("Debt focus week",
                            "Pay down high-interest debt first, or consolidate/refinance if possible."))
        elif "user->discretionary" in name:
            actions.append(("Cut discretionary spending",
                            "Set a weekly cap for eating out/entertainment. Try a 10% cut and re-check the probability."))
        elif "user->savings" in name and "trend" in name:
            actions.append(("Protect your savings trend",
                            "Automate a small transfer right after income hits to stabilize savings."))
        elif "income->user" in name and "trend" in name:
            actions.append(("Income recovery plan",
                            "Explore side income or reduce fixed costs temporarily to offset income decline."))
        elif "user->utilities" in name or "user->transport" in name or "user->food" in name:
            actions.append(("Trim variable expenses",
                            "Pick one category (food/transport/utilities) and reduce it by 5–10% this month."))
        else:
            # generic fallback
            actions.append(("Improve cashflow",
                            "Reduce one expense category slightly and watch how it changes goal probability."))

    # Deduplicate by title
    seen = set()
    deduped = []
    for title, desc in actions:
        if title not in seen:
            deduped.append((title, desc))
            seen.add(title)

    # Return top 3
    return deduped[:3]

def compute_savings_streak(monthly_df, user_id, months=range(1, 7), tolerance=0.0):
    """
    Simple 'streak' definition for MVP:
    Count consecutive months (ending at last month in window) where savings >= previous_month_savings - tolerance
    """
    df = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"].isin(months))].sort_values("month")
    s = df["savings_eur"].astype(float).values
    if len(s) < 2:
        return len(s)

    streak = 1
    # count backwards from last month
    for i in range(len(s)-1, 0, -1):
        if s[i] >= s[i-1] - tolerance:
            streak += 1
        else:
            break
    return streak

def compute_consistency_score(monthly_df, user_id, months=range(1, 7)):
    """
    0–100 score combining:
    - lower income volatility (good)
    - lower expense volatility (good)
    - higher savings trend (good)
    """
    df = monthly_df[(monthly_df["user_id"] == user_id) & (monthly_df["month"].isin(months))].sort_values("month")
    if df.empty:
        return 0

    income = df["income_eur"].astype(float).values
    expenses = df["total_expenses_eur"].astype(float).values
    savings = df["savings_eur"].astype(float).values

    # volatility normalized (avoid divide by zero)
    inc_cv = np.std(income) / (np.mean(income) + 1e-9)
    exp_cv = np.std(expenses) / (np.mean(expenses) + 1e-9)

    # savings trend (positive is good)
    x = np.arange(len(savings))
    sav_trend = float(np.polyfit(x, savings, 1)[0]) if len(savings) >= 2 else 0.0

    # Convert to 0..100 components
    # Lower CV -> higher score
    inc_score = max(0, 1 - inc_cv) * 40          # up to 40 pts
    exp_score = max(0, 1 - exp_cv) * 40          # up to 40 pts

    # Trend scaled into 0..20 (cap)
    # If trend is negative, it reduces this component
    trend_score = 10 + (sav_trend / 200.0) * 10  # roughly: +€200/month slope -> +10 pts
    trend_score = min(20, max(0, trend_score))

    score = inc_score + exp_score + trend_score
    return int(round(min(100, max(0, score))))

# -------------------------
# SREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Smart Financial Advisor (Graph Demo)", layout="wide")
st.title("AI Smart Financial Advisor — Dynamic Graph Prediction + Explanation")

left, right = st.columns([1, 2])

with left:
    uid = st.selectbox("Select user_id", users["user_id"].tolist(), index=0)
    show_months = st.multiselect("Show graph snapshots", [1, 6, 12], default=[1, 6, 12])

    st.markdown("### What-if (counterfactual)")
    rent_reduce = st.slider("Reduce rent edges by (%)", 0, 30, 0)
    debt_reduce = st.slider("Reduce debt edges by (%)", 0, 30, 0)
    disc_reduce = st.slider("Reduce Discretionary by (%)", 0, 30, 0)

u = users[users["user_id"] == uid].iloc[0]
true_label = int(labels[labels["user_id"] == uid]["will_reach_goal"].iloc[0])

st.write("### User profile")
st.json({
    "goal_amount_eur": int(u["goal_amount_eur"]),
    "debt_level": u["debt_level"],
    "shock_type": u["shock_type"],
    "shock_start_month": int(u["shock_start_month"]),
    "saving_propensity": float(u["saving_propensity"]),
    "base_income_eur": float(u["base_income_eur"]),
})

# ----- Streak + Consistency (Chumz/Revolut feel) -----
streak = compute_savings_streak(monthly, uid, months=range(1, 7), tolerance=0.0)
consistency = compute_consistency_score(monthly, uid, months=range(1, 7))

st.subheader("Habits & Consistency")
h1, h2, h3 = st.columns(3)
h1.metric("Savings streak (months)", streak)
h2.metric("Consistency score", f"{consistency}/100")
h3.metric("Mode", "Chumz × Revolut ✨")

if consistency >= 80:
    st.success("🔥 Strong consistency — keep it up!")
elif consistency >= 60:
    st.info("✅ Decent consistency — small tweaks can boost success odds.")
else:
    st.warning("⚠️ Volatile cashflow — use the What-if sliders to stabilize.")

target = min(100, 20 + streak * 15)  # example
st.caption(f"Suggested target this week: {target}%")

# Build features from graphs (months 1–6)
graphs_u = build_user_graph_sequence(users, monthly, uid, months=range(1, 7))
edge_ts = edge_timeseries_from_graphs(graphs_u)
feats = extract_graph_features(edge_ts)

# Apply counterfactual on FEATURES (simple + consistent with this baseline)
# Since features are edge-based, we can adjust the relevant edge means/trends.
def apply_counterfactual(feats, rent_reduce, debt_reduce):
    feats = dict(feats)  # copy
    if rent_reduce > 0:
        for k in list(feats.keys()):
            if k.startswith("User->Rent_"):
                feats[k] *= (1 - rent_reduce/100)
    if debt_reduce > 0:
        for k in list(feats.keys()):
            if k.startswith("User->Debt_"):
                feats[k] *= (1 - debt_reduce/100)
    return feats

feats_cf = apply_counterfactual(feats, rent_reduce, debt_reduce)

X_row = align_features(feats_cf, feature_cols)

proba = float(model.predict_proba(X_row)[0][1])
pred = int(proba >= 0.5)

st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("P(reach goal)", f"{proba:.2f}")
c2.metric("Predicted", "Reach ✅" if pred == 1 else "Not reach ❌")
c3.metric("True label", "Reach ✅" if true_label == 1 else "Not reach ❌")

st.subheader("Top explanation drivers (approx.)")
drivers = top_contributors_lr(model, X_row, top_k=10)
st.subheader("Next Best Actions (Revolut-style insights)")
actions = recommend_actions(drivers)

if actions:
    for i, (title, desc) in enumerate(actions, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {title}**")
            st.write(desc)
else:
    st.success("You're on track — no urgent actions detected. Keep your current plan consistent.")

st.subheader("Weekly Challenge (Chumz-style)")
# Simple challenge based on the most negative driver
challenge_name = "7-Day Discretionary Cut"
challenge_desc = "Reduce discretionary spending by 10% this week."
if any("User->Rent" in f for f in drivers.index):
    challenge_name = "Rent Buffer Week"
    challenge_desc = "Try to reduce rent impact (or offset it) by setting aside an extra €50–€100."
elif any("User->Debt" in f for f in drivers.index):
    challenge_name = "Debt Knockdown Week"
    challenge_desc = "Make one extra debt payment (even small) to reduce future pressure."
elif any("User->Discretionary" in f for f in drivers.index):
    challenge_name = "No-Spend Weekend"
    challenge_desc = "Pick 2 days with €0 discretionary spend and track the streak."

with st.container(border=True):
    st.markdown(f"**Challenge:** {challenge_name}")
    st.write(challenge_desc)

    # Fake progress bar for demo (you can connect it to real tracking later)
    progress = st.slider("Progress (demo)", 0, 100, 40, help="For MVP demo only")
    st.progress(progress / 100)

    if progress >= 100:
        st.balloons()
        st.success("Challenge complete! 🎉 Your savings habit is getting stronger.")

st.dataframe(drivers.rename("contribution").to_frame())

with right:
    st.subheader("Graph snapshots")
    cols = st.columns(len(show_months)) if show_months else st.columns(1)
    for i, m in enumerate(show_months):
        img_path = os.path.join(GRAPH_DIR, f"graph_user{uid}_month{m}.png")
        if os.path.exists(img_path):
            cols[i].image(img_path, caption=f"User {uid} — Month {m}", use_container_width=True)
        else:
            cols[i].warning(f"Missing image:\n{img_path}")

st.caption("Demo uses the same edge-based graph feature extraction used during training (months 1–6).")
