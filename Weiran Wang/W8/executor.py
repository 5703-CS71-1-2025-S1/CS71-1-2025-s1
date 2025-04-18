# executor.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_and_evaluate(X_train, y_train, X_val, y_val, 
                       hidden_dim=64, lr=0.001, dropout=0.2, l2=1e-4, 
                       batch_size=64, num_epochs=20, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Build datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device)).cpu().numpy().flatten()
    auc = roc_auc_score(y_val, val_preds)

    return auc, val_preds