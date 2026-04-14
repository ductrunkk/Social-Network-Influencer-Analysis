# data_loader.py
import os
import tarfile
import networkx as nx
import streamlit as st
from config import EDGES_DIR, TAR_FILE_PATH, DATA_DIR

@st.cache_resource
def extract_data():
    """Extract the tar.gz dataset archive (runs once thanks to caching)."""
    if not os.path.exists(EDGES_DIR):
        try:
            with st.spinner(f"Extracting {TAR_FILE_PATH}..."):
                with tarfile.open(TAR_FILE_PATH, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)
            return f"Extraction completed: {EDGES_DIR}"
        except FileNotFoundError:
            st.error(f"ERROR: File not found: {TAR_FILE_PATH}")
            return None
    return f"Data already exists at: {EDGES_DIR}"

@st.cache_resource
def build_full_graph():
    with st.spinner("Building the global network graph..."):
        G = nx.DiGraph()
        try:
            edge_files = [f for f in os.listdir(EDGES_DIR) if f.endswith(".edges")]
        except FileNotFoundError:
            st.error(f"ERROR: Directory not found: {EDGES_DIR}.")
            return None
        
        if not edge_files:
            st.error(f"ERROR: No .edges files found in {EDGES_DIR}.")
            return None

        # Merge all edge-list files into one directed graph.
        for file_name in edge_files:
            file_path = os.path.join(EDGES_DIR, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        u, v = line.strip().split()
                        G.add_edge(u, v)
                    except ValueError:
                        # Skip malformed lines that cannot be parsed into an edge.
                        pass
        
        st.success(f"Graph build completed! {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    return G