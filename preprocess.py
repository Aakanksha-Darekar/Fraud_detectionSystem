import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath, sample_size=1500):
    """Loads the dataset and processes only the first 1500 entries."""
    df = pd.read_csv("C:/Users/admin/OneDrive/Desktop/Self/fraud_detection_system/data/Synthetic_Financial_datasets_log.csv", nrows=sample_size)  # Load first 1500 rows only

    # Convert categorical `type` to numerical encoding
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])

    # Handling missing values: Apply only to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Select relevant features & target
    features = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    target = "isFraud"

    X = df[features]
    y = df[target]

    # Normalize numerical data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y  # Returning full dataframe for use in other modules