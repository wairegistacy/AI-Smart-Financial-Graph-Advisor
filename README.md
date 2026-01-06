# Graph Evolution for Financial Prediction Explanation

This repository contains a research prototype that models personal financial behavior as a sequence of **dynamic graphs** in order to **predict and explain savings goal attainment**.
---

## Motivation

Most personal finance and banking applications provide descriptive analytics such as transaction histories and spending summaries. While useful, these tools do not answer forward-looking questions such as:

- Will a user reach their savings goal?
- Why is success or failure likely?
- What concrete actions would improve outcomes?

This project addresses this gap by representing personal finance as an **evolving relational system**, enabling **interpretable prediction and actionable explanations**.

---

## Core Idea

Each user’s financial behavior is modeled as a **sequence of weighted, directed graphs**, where:

- Nodes represent financial entities (income, expenses, savings, goal)
- Edges represent financial flows (e.g., income → user, user → rent)
- Edge weights evolve over time (graph evolution)

Temporal statistics extracted from evolving edges are used to predict goal attainment and explain financial risk.

## Key ideas
- Personal finance represented as evolving directed graphs
- Edge-level temporal features (mean, volatility, trend)
- Explainable prediction of goal success
- Counterfactual "what-if" analysis
- Interactive demo inspired by modern fintech apps

---

## Project Structure
- `financial_advisor.py` – data inspection, graph construction, analysis
- `train_and_save_model.py` – feature extraction and model training
- `app.py` – Streamlit demo for prediction and explanation
- `docs/` – research notes and proposal
- `graphs/` – generated graph snapshots (not committed)

## How to run
```bash
pip install -r requirements.txt
python train_and_save_model.py
python -m streamlit run app.py