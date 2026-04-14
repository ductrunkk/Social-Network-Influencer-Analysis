# analysis.py
import networkx as nx
import pandas as pd
import streamlit as st
import time

@st.cache_resource
def perform_full_analysis(_G):
    """
    Run all heavy SNA analysis tasks: PageRank, degrees, clustering, and communities.
    Parameter _G is the pre-built directed graph.
    """
    with st.spinner("Running advanced SNA analysis (PageRank, communities)..."):
        start_time = time.time()
        
        # 1. Compute PageRank and degrees on the directed graph.
        pagerank_scores = nx.pagerank(_G, alpha=0.85)
        in_degree_scores = dict(_G.in_degree())
        out_degree_scores = dict(_G.out_degree())
        
        # 2. Convert to an undirected graph for clustering and communities.
        G_undirected = _G.to_undirected()
        
        # 3. Compute local clustering coefficients.
        clustering_scores = nx.clustering(G_undirected)
        
        # 4. Detect communities using the Louvain method.
        communities = nx.community.louvain_communities(G_undirected, seed=123)
        
        # Assemble all node-level metrics into a single DataFrame.
        df = pd.DataFrame.from_dict(pagerank_scores, orient='index', columns=['pagerank'])
        df['in_degree'] = df.index.map(in_degree_scores.get).fillna(0)
        df['out_degree'] = df.index.map(out_degree_scores.get).fillna(0)
        df['clustering_coeff'] = df.index.map(clustering_scores.get).fillna(0)
        
        # Assign a numeric community ID to each node.
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        
        df['community_id'] = df.index.map(node_to_community.get).fillna(-1)
        
        df_sorted = df.sort_values(by='pagerank', ascending=False)
        df_sorted.index.name = "UserID"
        
        total_time = time.time() - start_time
        st.success(f"Analysis completed in {total_time:.2f} seconds. Found {len(communities)} communities.")
    
    # Return both outputs so the expensive undirected conversion is reused.
    return df_sorted, G_undirected