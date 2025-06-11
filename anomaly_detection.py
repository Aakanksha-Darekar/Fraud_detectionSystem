from sklearn.ensemble import IsolationForest
from utils.preprocess import load_and_preprocess_data

_,X, y = load_and_preprocess_data("data/financial_fraud_dataset.csv")

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X)

# Predict anomalies
anomaly_scores = iso_forest.decision_function(X)
fraud_predictions = (anomaly_scores < -0.1).astype(int)

print("Anomaly Detection Completed")