import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import networkx as nx

# Load dataset independently
df = pd.read_csv("C:/Users/admin/OneDrive/Desktop/Self/fraud_detection_system/data/Synthetic_Financial_datasets_log.csv", nrows=1500)  # Limit to first 1500 entries

# Convert categorical `type` to numerical encoding
type_encoder = LabelEncoder()
df["type"] = type_encoder.fit_transform(df["type"])

# Convert `nameOrig` and `nameDest` to numeric IDs
node_encoder = LabelEncoder()
df["nameOrig"] = node_encoder.fit_transform(df["nameOrig"])
df["nameDest"] = node_encoder.fit_transform(df["nameDest"])

# Select relevant features
features = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
target = "isFraud"

X = df[features]
y = df[target]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create transaction graph using sender-receiver relationships (now numeric IDs)
edges = [(i, j) for i, j in zip(df["nameOrig"], df["nameDest"])]
G = nx.Graph(edges)

# Convert to PyTorch Geometric format (ensuring numeric node IDs)
edge_index = torch.tensor(edges, dtype=torch.long).T  # Use `.long()` for integer IDs
node_features = torch.tensor(X_scaled, dtype=torch.float)

graph_data = Data(x=node_features, edge_index=edge_index)

# Define Graph Neural Network (GNN) model
class FraudGCN(torch.nn.Module):
    def __init__(self):
        super(FraudGCN, self).__init__()
        self.conv1 = GCNConv(graph_data.num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)  # Binary classification

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = self.fc(x)
        return x

# Initialize model
model = FraudGCN()
print("✅ Graph Neural Network Initialized Successfully with Numeric Node IDs")

# Visualize the graph structure
plt.figure(figsize=(10, 8))
nx.draw(G, node_size=20, edge_color="gray", with_labels=False)
plt.title("Transaction Graph (Sender ↔ Receiver)")
plt.show()

