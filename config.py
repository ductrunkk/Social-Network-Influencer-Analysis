# config.py
import os

# --- Path Configuration ---
TAR_FILE_PATH = "twitter.tar.gz"
DATA_DIR = "data"
EXTRACTED_FOLDER_NAME = "twitter"

# Absolute path to the directory containing .edges files.
EDGES_DIR = os.path.join(DATA_DIR, EXTRACTED_FOLDER_NAME)

# --- ML Model Configuration ---
ML_TEST_SIZE = 0.3
ML_RANDOM_STATE = 42
PAGERANK_QUANTILE_THRESHOLD = 0.90 # Top 10%