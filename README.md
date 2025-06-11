# Fraud Detection System Using Machine Learning & Graph Neural Networks

# 📌 Overview
This project implements a **fraud detection system** using **machine learning models and graph-based analysis** to identify fraudulent transactions in financial datasets. 

## 🚀 Features
✅ **ML-based Fraud Classification** (Random Forest, XGBoost)  
✅ **Anomaly Detection** (Isolation Forest for rare fraud cases)  
✅ **Explainability** (SHAP for model transparency)  
✅ **Graph Neural Network (GNN)** (Analyzes transaction relationships)  
✅ **Self-Adaptive Learning** (Active learning for continuous improvement)  

## Dataset
The system uses **Kaggle's Financial Fraud Dataset** for training:  
- Transaction details (`step`, `type`, `amount`, `nameOrig`, `nameDest`)  
- Fraud indicators (`isFraud`, `isFlaggedFraud`)  
- Sender-receiver network for **graph-based fraud detection**  

## 🛠️ Installation & Requirements
### **1️⃣ Install Dependencies**
```bash
pip install pandas numpy scikit-learn xgboost modAL networkx torch_geometric matplotlib

