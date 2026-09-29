import numpy as np
import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# A. Giả lập Dataset Churn (1000 mẫu, 4 features)

np.random.seed(42)
X = np.random.rand(1000, 4)
y = (X[:, 1] * 0.5 + X[:, 3] * 0.3 + np.random.rand(1000) > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# B. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")

# C Model & Training
X_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)


class ChurnModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


model = ChurnModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epochs in range(10):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "model.pt")
print(" Trained & Saved model.pt and scaler.pkl successfully!")
