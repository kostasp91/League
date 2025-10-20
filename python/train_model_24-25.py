import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1️⃣ Load the prepared dataset
# -----------------------------
data = pd.read_csv("../TrainingModel/processed_train_dataset.csv")

# Features to train on
features = ["CR", "Minutes", "FPT", "Week"]
target = "NextFantasyPoints"

if not all(f in data.columns for f in features):
    raise ValueError(f"Missing required columns in dataset. Found: {data.columns}")

X = data[features].values
y = data[target].values

# -----------------------------
# 2️⃣ Normalize + split
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 3️⃣ Define model architecture
# -----------------------------
class PlayerModel(nn.Module):
    def __init__(self, input_size):
        super(PlayerModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 4️⃣ Train model
# -----------------------------
model = PlayerModel(input_size=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        val_loss = criterion(model(X_val_t), y_val_t).item()
        print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")

# -----------------------------
# 5️⃣ Save model + scaler
# -----------------------------
torch.save(model.state_dict(), "../player_model.pth")
import joblib
joblib.dump(scaler, "scaler.pkl")

print("\n Model saved to player_model.pth")
print(" Scaler saved to scaler.pkl")
