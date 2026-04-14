# Social Network Influencer Analysis and Community Detection

An end-to-end AI/ML project that identifies influential users and communities in a Twitter social graph, then delivers insights through an interactive Streamlit application.

This project demonstrates ownership across the full lifecycle:

- Business framing: converts a social influence question into measurable analytics and prediction tasks.
- Data and modeling: builds graph-based features, performs community detection, and trains a supervised ML classifier.
- Product delivery: ships results in a user-friendly dashboard for non-technical stakeholders.

In short, this project shows the ability to move from raw data to business-ready intelligence.

## Business Problem

Organizations need to answer:

- Who are the most influential users in a large social network?
- Which communities exist, and how are users grouped?
- Can we predict likely influencers from behavior and structure signals?

This is useful for:

- Marketing and creator partnerships
- Community strategy and audience segmentation
- Early identification of high-impact accounts

## Solution Overview

The system combines Social Network Analysis (SNA) and Machine Learning:

1. Build a directed graph from Twitter edge-list files.
2. Compute network metrics (PageRank, in-degree, out-degree, clustering coefficient).
3. Detect communities using Louvain clustering.
4. Define influencer labels from top 10% PageRank users.
5. Train a Random Forest classifier to predict influencer status.
6. Present rankings, model metrics, and ego-network visualizations in Streamlit.

## AI/ML Methodology

### Data

- Source: Twitter SNAP edge-list dataset
- Graph type: Directed graph
- Input format: multiple `.edges` files

### Feature Engineering

- `in_degree`: number of followers (incoming edges)
- `out_degree`: number of followings (outgoing edges)
- `clustering_coeff`: local neighborhood connectivity

### Labeling Strategy

- Influencer label (`is_influencer`) is generated from PageRank quantile threshold.
- Current threshold: top 10% (`PAGERANK_QUANTILE_THRESHOLD = 0.90`).

### Model

- Algorithm: Random Forest Classifier
- Train/test split: stratified (handles class imbalance)
- Imbalance handling: `class_weight='balanced'`
- Evaluation: classification report with Accuracy and F1-score

## Product Features

The Streamlit app provides 3 business-facing views:

1. Influencer Ranking

- Top users ranked by PageRank with key graph metrics.

2. Prediction Model

- ML performance summary (Accuracy, F1-score, class report).

3. User Explorer

- Detailed user-level metrics
- Community assignment
- Interactive ego-network visualization (Pyvis)

## Tech Stack

- Python
- Streamlit
- pandas
- NetworkX
- scikit-learn
- Pyvis

## Project Structure

- `app.py`: Streamlit entry point and UI orchestration
- `data_loader.py`: dataset extraction and graph construction
- `analysis.py`: SNA metrics and community detection
- `ML_model.py`: label creation, model training, evaluation
- `visualization.py`: ego-network rendering with Pyvis
- `config.py`: configurable paths and ML constants

## How to Run Locally

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install streamlit pandas networkx scikit-learn pyvis
```

3. Ensure dataset archive is available:

- Place `twitter.tar.gz` in the project root.
- The app will auto-extract to `data/twitter` on first run.

4. Start the app:

```powershell
streamlit run app.py
```

## Business-Ready Outcomes

This project delivers:

- A transparent ranking of high-impact users
- Community insights for segmentation decisions
- A predictive model to support proactive outreach strategy
- An interactive interface that HR, business, and technical teams can all review
