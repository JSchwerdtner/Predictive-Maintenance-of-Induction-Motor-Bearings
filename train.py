import pandas as pd
import numpy as np
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def load_dataset(data_path):
    possible_paths = [
        data_path,
        '/kaggle/input/bearing-dataset/2nd_test/2nd_test/ims_bearing_features.csv',
        '/kaggle/input/bearing-dataset/2nd_test/2nd_test',
        '/kaggle/input/bearing-dataset/ims_bearing_features.csv',
        'ims_bearing_features.csv',
        'data/ims_bearing_features.csv'
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            if os.path.isdir(path):
                sub_path = os.path.join(path, 'ims_bearing_features.csv')
                if os.path.exists(sub_path):
                    print(f"Loading dataset from: {sub_path}")
                    return pd.read_csv(sub_path, low_memory=False)
            else:
                print(f"Loading dataset from: {path}")
                return pd.read_csv(path, low_memory=False)
                
    raise FileNotFoundError("Dataset CSV file not found. Please specify a valid path with --data.")

def main():
    parser = argparse.ArgumentParser(description="Train Bearing Predictive Maintenance Model")
    parser.add_argument("--data", type=str, default="", help="Path to ims_bearing_features.csv")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"], help="Training task type")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in RandomForest")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth of trees")
    parser.add_argument("--output", type=str, default="bearing_model.pkl", help="Output path for saved model")
    
    args = parser.parse_args()
    
    # Load data
    df = load_dataset(args.data)
    
    if args.task == "regression":
        target_col = 'remaining_life_fraction'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        feature_cols = [col for col in df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean'])][:10]
        train_df = df.dropna(subset=[target_col] + feature_cols)
        if len(train_df) == 0:
            train_df = df.dropna(subset=[target_col])
            feature_cols = [col for col in df.columns if col in train_df.select_dtypes(include=[np.number]).columns and col != target_col][:10]
            
        X = train_df[feature_cols]
        y = train_df[target_col]
        
        print(f"Training Regression Model (RUL) with {len(X)} samples and {len(feature_cols)} features...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"--- Model Evaluation ---")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
    else:
        target_col = 'stage'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        feature_cols = [col for col in df.columns if any(k in col for k in ['rms', 'kurtosis', 'peak_to_peak', 'crest_factor', 'mean', 'bandpower'])][:12]
        train_df = df.dropna(subset=[target_col])
        
        X = train_df[feature_cols]
        y = train_df[target_col]
        
        print(f"Training Classification Model (Health Stage) with {len(X)} samples and {len(feature_cols)} features...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"--- Model Evaluation ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        
    # Save model
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {args.output}")

if __name__ == "__main__":
    main()