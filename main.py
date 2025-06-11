import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utils.preprocess import load_and_preprocess_data
from models.classifier import rf_model, xgb_model
from models.anomaly_detection import iso_forest
from models.explainability import model as shap_model
from models.graph_model import model as gnn_model
from utils.evaluation import *

# Load data in chunks to handle large datasets efficiently
def load_data_in_chunks(filepath, chunk_size=10000):
    chunk_list = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunk_list.append(chunk)
    return pd.concat(chunk_list, ignore_index=True)

# Load and preprocess the dataset using chunking
dataset_path = "C:/Users/admin/OneDrive/Desktop/Self/fraud_detection_system/data/Synthetic_Financial_datasets_log.csv"
df = load_data_in_chunks(dataset_path)
_,X, y = load_and_preprocess_data(dataset_path)

# Train-test split with balanced dataset sampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Run models efficiently using parallel execution
def run_evaluation():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(evaluate_classification, rf_model, X_test, y_test)
        executor.submit(evaluate_classification, xgb_model, X_test, y_test)
        executor.submit(evaluate_anomaly_detection, iso_forest, X_test, y_test)
        executor.submit(evaluate_explainability, shap_model, X_test)
        executor.submit(evaluate_graph_model, gnn_model, X_test)

if __name__ == "__main__":
    print("⚡ Running Fraud Detection Models on Large Dataset...")
    run_evaluation()
    print("✅ Model Evaluation Completed!")