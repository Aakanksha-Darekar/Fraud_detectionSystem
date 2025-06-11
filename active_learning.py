import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from utils.preprocess import load_and_preprocess_data

X, y = load_and_preprocess_data("data/financial_fraud_dataset.csv")

# Select initial labeled samples
initial_sample_size = int(0.1 * len(X))
indices = np.random.choice(range(len(X)), size=initial_sample_size, replace=False)

X_initial = X[indices]
y_initial = y.iloc[indices]

# Active learning setup
active_learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    query_strategy=uncertainty_sampling,
    X_training=X_initial,
    y_training=y_initial
)

# Adaptive learning loop
for query_idx in range(5):
    query_idx, query_instance = active_learner.query(X)
    y_new_label = y.iloc[query_idx]  # Simulated expert labeling
    active_learner.teach(query_instance, y_new_label)

print("Adaptive Learning Process Completed")