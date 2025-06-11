import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from utils.preprocess import load_and_preprocess_data

_,X, y = load_and_preprocess_data("data/financial_fraud_dataset.csv")

# Train model
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Explain predictions
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Visualization
shap.summary_plot(shap_values, X)