# Graph Evolution for Financial Prediction Explanation

## Overview
The AI Smart Financial Advisor is a research-driven prototype that models personal finance as a dynamic, evolving graph in order to predict savings goal attainment and provide transparent, actionable explanations. The system unifies traditional banking flows, peer-to-peer (mobile money) transfers, and personal budgeting behavior within a single graph-based intelligence layer.

This project is designed as a PhD-ready research prototype, demonstrating how graph evolution and explainable machine learning can be used to move beyond descriptive financial dashboards toward predictive and interpretable financial decision support.

### Key Contributions (Deliverables)
#### Deliverable 1 — Core Graph Intelligence (Completed)

Models each user’s financial state as a monthly directed, weighted graph.

Captures graph evolution over time (months 1–6) using edge-level statistics (mean, standard deviation, trend).

Predicts whether a user will reach a long-term savings goal using an interpretable baseline model (logistic regression).

Generates human-readable explanations linking predictions to concrete financial behaviors (e.g. rent pressure, savings trend).

Supports counterfactual analysis (“what-if” reductions in rent, debt, or discretionary spending).

#### Deliverable 2 — Mobile Money (P2P) Abstraction (Completed)

Explicitly models peer-to-peer financial transfers inspired by mobile money systems such as M-Pesa.

Introduces fixed peer nodes (Peer1..PeerK) to represent a user’s social financial network.

Adds bidirectional edges:

User → PeerX (money sent)

PeerX → User (money received / support)

Simulates increased incoming transfers during financial shocks to reflect network-based resilience.

Integrates P2P flows into savings and income calculations, allowing explanations to highlight the role of social support.

#### Deliverable 3 — Banking Abstraction (Completed)

Introduces a first-class Bank node to represent traditional banks and neobanks (e.g. AIB, BOI, Revolut-style accounts).

Routes financial flows through the bank:

Income → Bank → User (salary crediting)

User → Bank (aggregate outgoing payments)

Savings → Bank (stored savings)

Maintains category-level expense edges for interpretability while enabling institutional-level analysis.

Demonstrates how banks, neobanks, and mobile money systems can be unified within a single graph framework.

### System Architecture

Each user is represented by a sequence of graph snapshots (one per month). Nodes represent entities (User, Bank, Income, Savings, Peers, Expense Categories), and edges represent financial relationships whose weights evolve over time.

From these evolving graphs, the system:

Extracts temporal edge features (mean, volatility, trend).

Trains a predictive model on partial observation windows (months 1–6).

Produces explanations grounded in specific financial relationships.

Evaluates counterfactual interventions for actionable guidance.

### Interactive Demo

A Streamlit-based demo (app.py) allows users to:

Select a user profile

View graph snapshots (months 1, 6, 12)

See predicted probability of reaching a savings goal

Explore top explanatory drivers

Run counterfactual scenarios (rent, debt, discretionary reductions)

Engage with habit metrics and challenges inspired by Revolut × Chumz-style UX

### Repository Structure
.
├── app.py                     # Interactive Streamlit demo
├── train_and_save_model.py    # Model training + feature extraction
├── financial_advisor.py       # Research analysis & experiments
├── generate_graph_images.py   # Graph snapshot generation
├── generate_graph_images_by_shock.py
├── requirements.txt
├── README.md
└── .gitignore

### Installation & Usage
pip install -r requirements.txt
python train_and_save_model.py
python -m streamlit run app.py

### Research Context
This prototype serves as the foundation for a PhD research project on:

Dynamic graph modeling for financial behavior

Explainable AI in personal finance

Robust prediction under financial shocks

Unifying banking and mobile money systems through graph abstractions

The system is intentionally designed as a research artifact, not a production financial application.

### License & Disclaimer

This project uses synthetic data only. It is intended for academic research and demonstration purposes and does not handle real financial transactions.