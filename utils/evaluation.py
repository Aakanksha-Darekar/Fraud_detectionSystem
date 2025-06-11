import numpy as np
import shap
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

def evaluate_classification(model, X_test, y_test):
    """Evaluates fraud detection classifiers."""
    y_pred = model.predict(X_test)
    
    print("📊 Classification Performance:")
    print(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

def evaluate_anomaly_detection(model, X_test, y_test):
    """Evaluates anomaly detection models."""
    anomaly_scores = model.decision_function(X_test)
    fraud_predictions = (anomaly_scores < -0.1).astype(int)
    
    print("🔍 Anomaly Detection Performance:")
    print(classification_report(y_test, fraud_predictions))
    
    return {
        "accuracy": accuracy_score(y_test, fraud_predictions),
        "precision": precision_score(y_test, fraud_predictions),
        "recall": recall_score(y_test, fraud_predictions),
        "f1_score": f1_score(y_test, fraud_predictions)
    }

def evaluate_explainability(model, X_test):
    """Generates SHAP explainability plots."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    print("🧐 Generating SHAP Explainability Plots...")
    shap.summary_plot(shap_values, X_test)

def evaluate_graph_model(model, graph_data):
    """Evaluates Graph Neural Network (GNN) model."""
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
        predictions = torch.argmax(output, dim=1).cpu().numpy()
    
    print("🖥️ Graph Neural Network Evaluation:")
    print(f"Predictions: {predictions[:10]} (Sample)")
