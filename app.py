import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Motor Bearing Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# App Header
st.title("⚙️ Electrical Motor Bearing Predictive Maintenance AI")
st.markdown("""
This application analyzes vibration feature sets extracted from motor bearings to monitor health, 
predict remaining operational life, classify bearing health stages, and train custom machine learning models.
""")

# Sidebar Controls for Data Input
st.sidebar.header("🛠️ Model & View Controls")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, low_memory=False)
        
    # Define possible dataset paths (checking Kaggle directory first, then local paths)
    possible_paths = [
        '/kaggle/input/bearing-dataset/2nd_test/2nd_test/ims_bearing_features.csv',
        '/kaggle/input/bearing-dataset/2nd_test/2nd_test',
        '/kaggle/input/bearing-dataset/ims_bearing_features.csv',
        'ims_bearing_features.csv',
        'data/ims_bearing_features.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                sub_path = os.path.join(path, 'ims_bearing_features.csv')
                if os.path.exists(sub_path):
                    return pd.read_csv(sub_path, low_memory=False)
            else:
                return pd.read_csv(path, low_memory=False)
                
    raise FileNotFoundError("Dataset CSV file not found.")

# File uploader fallback in sidebar
uploaded_file = st.sidebar.file_uploader("Upload 'ims_bearing_features.csv' (Optional)", type=['csv'])

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.warning("⚠️ Dataset not found automatically. Please upload your `ims_bearing_features.csv` file using the sidebar uploader.")
    st.stop()

view_mode = st.sidebar.radio(
    "Select View", 
    ["Dataset Explorer", "Remaining Life Prediction (RUL)", "Bearing Health Classification", "Train Custom ML Model", "Health Degradation Trends"]
)

# Filter by Test ID or Bearing if available
if 'test_id' in df.columns:
    selected_test = st.sidebar.selectbox("Select Test ID", df['test_id'].unique())
    filtered_df = df[df['test_id'] == selected_test]
else:
    filtered_df = df

if view_mode == "Dataset Explorer":
    st.subheader("📊 Dataset Overview")
    st.write(f"Total Records: {filtered_df.shape[0]} | Total Features: {filtered_df.shape[1]}")
    
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    st.subheader("Summary Statistics")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    st.dataframe(filtered_df[numeric_cols].describe().T, use_container_width=True)

elif view_mode == "Remaining Life Prediction (RUL)":
    st.subheader("🤖 Machine Learning: Remaining Useful Life (RUL) Estimator")
    
    feature_candidates = [col for col in filtered_df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean'])]
    target_col = 'remaining_life_fraction'
    
    if target_col not in filtered_df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
    else:
        model_df = filtered_df.dropna(subset=[target_col] + feature_candidates[:5])
        
        if len(model_df) < 50:
            st.warning("Not enough labeled samples with remaining life fraction in this view. Using broader dataset rows.")
            model_df = df.dropna(subset=[target_col])
            
        if len(model_df) > 0:
            X = model_df[feature_candidates[:10]]
            y = model_df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            col1, col2 = st.columns(2)
            col1.metric("Model R² Score", f"{r2:.2f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
            
            st.subheader("🔍 Feature Importance for RUL Prediction")
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(8), ax=ax, palette='viridis')
            ax.set_title("Top Vibration Features Driving Bearing Aging")
            st.pyplot(fig)
        else:
            st.error("Insufficient valid data points to train the predictive model.")

elif view_mode == "Bearing Health Classification":
    st.subheader("🛡️ Machine Learning: Bearing Health Outcome Classifier")
    st.markdown("Predicts the current operational degradation stage (`healthy`, `incipient`, `developing`, `severe`) based on vibration statistics.")
    
    target_col = 'stage'
    feature_candidates = [col for col in filtered_df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean', 'bandpower'])]
    
    if target_col not in filtered_df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
    else:
        class_df = filtered_df.dropna(subset=[target_col])
        
        if len(class_df) > 50:
            X = class_df[feature_candidates[:12]]
            y = class_df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            st.metric("Health Classification Accuracy", f"{acc * 100:.2f}%")
            
            st.subheader("📊 Class Distribution in Dataset")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=y, ax=ax, palette='Set2')
            ax.set_title("Count of Samples per Bearing Health Stage")
            st.pyplot(fig)
        else:
            st.error("Insufficient samples available for classification model training.")

elif view_mode == "Train Custom ML Model":
    st.subheader("🏋️ Interactive Model Trainer & Hyperparameter Tuning")
    st.markdown("Configure and train a custom machine learning model using your dataset, then download the trained artifact.")
    
    with st.form("training_form"):
        st.sidebar.markdown("---")
        st.subheader("Training Configuration")
        
        task_type = st.selectbox("Select ML Task", ["Remaining Useful Life (Regression)", "Bearing Health Stage (Classification)"])
        test_size = st.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.slider("Number of Trees (n_estimators)", 10, 300, 100, 10)
        max_depth = st.slider("Max Tree Depth", 2, 30, 10, 1)
        
        submit_button = st.form_submit_button(label="🚀 Train Model Now")

    if submit_button:
        with st.spinner("Training model on dataset... Please wait."):
            if task_type == "Remaining Useful Life (Regression)":
                target_col = 'remaining_life_fraction'
                feature_cols = [col for col in filtered_df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean'])][:10]
                
                train_df = filtered_df.dropna(subset=[target_col] + feature_cols)
                if len(train_df) == 0:
                    train_df = df.dropna(subset=[target_col] + feature_cols)
                
                X = train_df[feature_cols]
                y = train_df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                
                st.success("✅ Regression Model Successfully Trained!")
                c1, c2 = st.columns(2)
                c1.metric("Trained R² Score", f"{r2:.4f}")
                c2.metric("Trained RMSE", f"{rmse:.4f}")
                
            else:
                target_col = 'stage'
                feature_cols = [col for col in filtered_df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean', 'bandpower'])][:12]
                
                train_df = filtered_df.dropna(subset=[target_col] + feature_cols)
                X = train_df[feature_cols]
                y = train_df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                st.success("✅ Classification Model Successfully Trained!")
                st.metric("Trained Accuracy", f"{acc * 100:.2f}%")
            
            # Save model to bytes for download
            model_bytes = pickle.dumps(model)
            st.download_button(
                label="📥 Download Trained Model (.pkl)",
                data=model_bytes,
                file_name="bearing_maintenance_model.pkl",
                mime="application/octet-stream"
            )

elif view_mode == "Health Degradation Trends":
    st.subheader("📈 Vibration & Health Degradation Over Time")
    
    if 'bearing' in filtered_df.columns:
        selected_bearing = st.selectbox("Select Bearing Identifier", filtered_df['bearing'].unique())
        bearing_data = filtered_df[filtered_df['bearing'] == selected_bearing]
    else:
        bearing_data = filtered_df

    if 'snapshot_index' in bearing_data.columns and 'axis1_rms' in bearing_data.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(bearing_data['snapshot_index'], bearing_data['axis1_rms'], label='Axis 1 RMS', color='royalblue')
        if 'axis2_rms' in bearing_data.columns:
            ax.plot(bearing_data['snapshot_index'], bearing_data['axis2_rms'], label='Axis 2 RMS', color='darkorange', alpha=0.7)
        ax.set_title("Vibration RMS Trend Over Time")
        ax.set_xlabel("Snapshot Index (Run-time progression)")
        ax.set_ylabel("RMS Value")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Required time-series columns (`snapshot_index` or `axis1_rms`) missing from filtered view.")
