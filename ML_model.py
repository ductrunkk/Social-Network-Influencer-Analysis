# ml_model.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from config import ML_TEST_SIZE, ML_RANDOM_STATE, PAGERANK_QUANTILE_THRESHOLD

@st.cache_resource
def build_ml_model(df):
    """
    Train an ML model to predict influencers.
    Parameter df is the analyzed feature DataFrame.
    """
    with st.spinner("Training machine learning model..."):
        # Step 1: Create the target label (Y).
        quantile_threshold = df['pagerank'].quantile(PAGERANK_QUANTILE_THRESHOLD)
        df['is_influencer'] = (df['pagerank'] > quantile_threshold).astype(int)
        
        # Step 2: Build the feature matrix (X).
        features = ['in_degree', 'out_degree', 'clustering_coeff']
        X = df[features]
        y = df['is_influencer']
        
        # Step 3: Split and train.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=ML_TEST_SIZE, 
            random_state=ML_RANDOM_STATE, 
            stratify=y # Critical for imbalanced target classes.
        )
        
        model = RandomForestClassifier(
            random_state=ML_RANDOM_STATE, 
            n_jobs=-1, 
            class_weight='balanced' # Automatically compensates for class imbalance.
        )
        model.fit(X_train, y_train)
        
        # Step 4: Evaluate on the holdout set.
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        st.success("ML model training completed!")
    
    # Return all three outputs because df now includes 'is_influencer'.
    return model, report, df