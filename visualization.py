# visualization.py

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx

def render_ego_network(G_undirected, results_df, user_id):
    """
    Render an interactive ego-network graph with Pyvis.
    """
    with st.spinner(f"Rendering interaction network for {user_id}..."):
        
        # --- PART 1: VALIDATE GRAPH INPUT ---
        try:
            neighbors = list(G_undirected.neighbors(user_id))
        except nx.NetworkXError:
            st.error(f"Error: Node {user_id} was not found in the graph. (NetworkXError)")
            return
        except Exception as e:
            st.error(f"Unexpected error while retrieving node neighbors: {e}")
            return

        if len(neighbors) > 100:
            st.warning(f"This user has {len(neighbors)} connections. Rendering a sample of 100 connections.")
            neighbors = neighbors[:100]
        
        nodes_to_draw = neighbors + [user_id]
        sub_g = G_undirected.subgraph(nodes_to_draw)
        
        # --- PART 2: BUILD PYVIS GRAPH ---
        net = Network(height="500px", width="100%", cdn_resources='in_line')
        
        for node in sub_g.nodes():
            try:
                # Read node metrics from the analysis DataFrame.
                node_data = results_df.loc[node]
                title = f"""
                UserID: {node}
                Community: {node_data['community_id']}
                PageRank: {node_data['pagerank']:.6f}
                Followers: {node_data['in_degree']}
                """
                
                # Highlight the selected user node.
                color = 'lightblue'
                if node == user_id:
                    color = 'red'
                
                net.add_node(node, label=node, color=color, size=15, title=title)
            
            except KeyError:
                # Fallback for neighbor nodes missing detailed metrics.
                net.add_node(node, label=node, color='grey', size=5, title=f"UserID: {node}\n(No detailed data available)")

        for edge in sub_g.edges():
            net.add_edge(edge[0], edge[1])

        net.repulsion(node_distance=100, spring_length=200)
        
        # --- PART 3: RENDER IN STREAMLIT ---
        try:
            # Generate standalone HTML from the configured Pyvis network.
            html_content = net.generate_html() 
            
            # Render the generated HTML directly in the app.
            components.html(html_content, height=510, width="100%")
            
        except Exception as e:
            st.error(f"Error while rendering the Pyvis graph: {e}")