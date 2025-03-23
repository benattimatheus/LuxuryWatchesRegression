# In application/services.py
from model.model.data_repository import load_data
from model.model.data_preprocessor import preprocess_data
from model.model_service import build_model
from model.model.feature_service import analyze_features
import pandas as pd


def run_pipeline(file_path: str, target: str):
    # Load raw data
    data = load_data(file_path)
    
    # Preprocess data
    X_preprocessed, y, preprocessor = preprocess_data(data, target)
    
    # Create a new DataFrame with preprocessed features
    X_df = pd.DataFrame(X_preprocessed)
    data_preprocessed = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
    
    # Build and train the model using the preprocessed data.
    model, le = build_model(data_preprocessed, target)
    
    # Analyze features with the trained model.
    shap_summary = analyze_features(model, data_preprocessed, target)
    
    return model, shap_summary
