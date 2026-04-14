# app.py
import streamlit as st
import pandas as pd

# Import components from local modules
from data_loader import extract_data, build_full_graph
from analysis import perform_full_analysis
from ML_model import build_ml_model
from visualization import render_ego_network

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Social Network Influencer Analysis")
st.title("Influencer Analysis and Community Detection")

# --- Sidebar ---
st.sidebar.header("Project Information")
st.sidebar.markdown("""
Influence analysis project
on the Twitter SNAP dataset.
- **Author:** Nguyen Duc Trung
- **Class:** 64.HTTT
""")
extraction_status = extract_data()
st.sidebar.info(extraction_status)

# --- Main Logic ---
# 1. Load and build the graph
G = build_full_graph()

if G is not None:
    # 2. Run network analysis
    results_df, G_undirected = perform_full_analysis(G)
    
    # 3. Train and evaluate the ML model
    ml_model, ml_report, results_df = build_ml_model(results_df)

    # --- UI Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "Influencer Ranking", 
        "Prediction Model", 
        "User Explorer"
    ])

    # --- Tab 1: Ranking ---
    with tab1:
        st.header("Influencer Ranking (by PageRank)")
        st.write("Top 20 most influential users in the network.")
        st.dataframe(results_df.head(20).style.format({
            "pagerank": "{:.6f}",
            "clustering_coeff": "{:.4f}",
            "in_degree": "{:,.0f}",
            "out_degree": "{:,.0f}",
            "community_id": "{:,.0f}"
        }))

    # --- Tab 2: ML Results ---
    with tab2:
        st.header("Influence Prediction Model")
        st.write("""
        Train a Random Forest model to predict whether a user is an "Influencer" (top 10% by PageRank),
        using only: **Followers (in_degree), Following (out_degree), and Clustering Coefficient (clustering_coeff)**.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report (Class 1 = Influencer)")
            report_df = pd.DataFrame(ml_report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
        with col2:
            st.subheader("Summary")
            st.metric("Accuracy", f"{ml_report['accuracy']:.2%}")
            st.metric("F1-Score (Influencer)", f"{ml_report['1']['f1-score']:.3f}")
            st.info("The F1-Score indicates that the model identifies influencers reasonably well, despite strong class imbalance.")

    # --- Tab 3: User Explorer ---
    with tab3:
        st.header("User and Network Explorer")
        st.write("Enter a UserID from the ranking table above to view details.")
        
        # Default to the highest-ranked user.
        default_id = results_df.index[0]
        user_id = st.text_input("Enter UserID:", value=default_id)
        
        if user_id in results_df.index:
            st.subheader(f"Detailed analysis for user: {user_id}")
            user_data = results_df.loc[user_id]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("PageRank Rank", f"#{results_df.index.get_loc(user_id) + 1}")
            col2.metric("Community ID (Louvain)", f"Cluster #{user_data['community_id']:,.0f}")
            col3.metric("Followers (In-Degree)", f"{user_data['in_degree']:,.0f}")
            
            # Render the interactive ego network graph.
            render_ego_network(G_undirected, results_df, user_id)
            
        else:
            st.error("UserID not found. Try another ID from the table.")
else:
    st.error("Unable to load the graph. Check your data files and the 'data' folder.")