import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, AttentionalAggregation
from torch_geometric.utils import degree
from sklearn.model_selection import KFold

# Dataset Preparation 
# We define a custom dataset to load graphs from a saved file.
# Each graph also adds node degree information (how many neighbors each node has).
class GraphDataset(GeometricDataset):
    def __init__(self, filepath):
        super().__init__(root=".")
        self.graphs = torch.load(filepath, weights_only=False)
        for g in self.graphs:
            deg_feature = degree(g.edge_index[0], dtype=torch.float).unsqueeze(1)
            g.x = torch.cat([g.x, deg_feature], dim=1)  # add degree as extra feature

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

#  Define one Graph Neural Network layer 
# This layer tells how nodes exchange information with their neighbors.
class GraphLayer(MessagePassing):
    def __init__(self, in_feats, out_feats):
        super().__init__(aggr='add')  # Add messages from neighboring nodes
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_feats + out_feats, out_feats),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(out_feats)  # Normalize for stability

    def forward(self, x, edge_index):
        res = x  # Keep a copy of the input for skip connection
        x = self.propagate(edge_index, x=x)  # Gather messages from neighbors
        x = self.update_mlp(torch.cat([res, x], dim=-1))  # Update the node features
        return self.norm(x)

    def message(self, x_i, x_j):
        return self.message_mlp(torch.cat([x_i, x_j], dim=-1))  # Create message from neighbors

#  Build the Full GNN Model 
# Stack multiple GraphLayers together and predict a final number for the whole graph.
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.initial_proj = nn.Linear(input_dim, hidden_dim)  # Project initial features
        self.gnn1 = GraphLayer(hidden_dim, hidden_dim)
        self.gnn2 = GraphLayer(hidden_dim, hidden_dim)
        self.gnn3 = GraphLayer(hidden_dim, hidden_dim)
        self.gnn4 = GraphLayer(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(0.2)  # Regularization to prevent overfitting
        self.global_pool = AttentionalAggregation(  # Collect info from all nodes smartly
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  

    def forward(self, data):
        x = F.relu(self.initial_proj(data.x))
        x = self.gnn1(x, data.edge_index)
        x = self.gnn2(x, data.edge_index)
        x = self.gnn3(x, data.edge_index)
        x = self.gnn4(x, data.edge_index)
        x = self.dropout_layer(x)
        x = self.global_pool(x, data.batch)  # Combine node features into graph feature
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(-1)  # Final prediction

# Setup seeds to make the results repeatable 
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load Training and Testing Data 
train_file = "/ocean/projects/cis250010p/shanmuga/GraphNN/train.pt"
test_file = "/ocean/projects/cis250010p/shanmuga/GraphNN/test.pt"

train_dataset = GraphDataset(train_file)
test_dataset = GraphDataset(test_file)

test_loader = DataLoader(test_dataset, batch_size=64)

# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3-Fold Cross Validation Training
kf = KFold(n_splits=3, shuffle=True, random_state=42)
all_fold_preds = []



# Go through each fold
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"\n[Fold {fold_idx+1}] in progress...")
    
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)

    model = GNNModel(train_dataset[0].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train for 100 epochs
    for ep in range(1, 101):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = F.l1_loss(output, batch.y.view(-1))  # L1 loss = mean absolute error
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        
        if ep % 20 == 0:
            print(f"Fold {fold_idx+1} | Epoch {ep} | Loss: {epoch_loss / len(train_loader.dataset):.4f}")

    # make predictions on test data
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            fold_preds.extend(preds.cpu().numpy())

    all_fold_preds.append(fold_preds)

# Save Final Predictions
# Take the average prediction across all folds (ensemble)
final_preds = np.mean(all_fold_preds, axis=0)

submission_df = pd.read_csv("sample_submission.csv")
submission_df["labels"] = final_preds

submission_df.to_csv("submission.csv", index=False)

print("Saved submission.csv")
