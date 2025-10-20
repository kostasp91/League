from flask import Flask, jsonify, request
import os, json
import pandas as pd
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)

OUT_DIR = os.environ.get("TRAINING_OUT_DIR", "../TrainingModel")
MODEL_PATH = os.path.join(OUT_DIR, "player_model.pth")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(OUT_DIR, "features_used.json")
LATEST_PATH = os.path.join(OUT_DIR, "latest_week_data.csv")

# Load features from training (order matters). Fallback covers extras if JSON missing.
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f).get("features", [])
else:
    features = [
        "CR", "MIN", "avgFPT", "lastFPT", "adjEfficiency",
        "avg_REB", "avg_AST", "avg_STL", "avg_BLK", "avg_TOV", "avg_PF",
        "FG_pct", "TP_pct", "FT_pct", "roll3FPT",
        "avgMIN", "roll3MIN", "availability", "avgFPM",
        # p_play might not be in the model features; fine—it's for output.
    ]

id_cols = ["Player", "Pos", "Team"]

class PlayerModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

scaler = joblib.load(SCALER_PATH)
model = PlayerModel(input_size=len(features))

state = torch.load(MODEL_PATH, map_location="cpu")
try:
    model.load_state_dict(state, strict=True)
except RuntimeError:
    # Compatibility if checkpoint used "model." instead of "net."
    fixed = {k.replace("model.", "net."): v for k, v in state.items()}
    model.load_state_dict(fixed, strict=False)
model.eval()

latest_df = pd.read_csv(LATEST_PATH)
keep_cols = [c for c in id_cols + features + ["p_play","avgMIN","roll3MIN","MIN"] if c in latest_df.columns]
latest_features = latest_df[keep_cols].copy()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True) or {}
        incoming = pd.DataFrame(body.get("players", []))
        if incoming.empty:
            return jsonify({"error": "No players payload"}), 400

        incoming = incoming.rename(columns={
            "Minutes": "MIN", "minute": "MIN",
            "AvgFP": "avgFPT", "avgFP": "avgFPT",
        })
        for c in id_cols:
            if c not in incoming.columns:
                incoming[c] = ""

        joined = pd.merge(incoming, latest_features, on=id_cols, how="left", suffixes=("", "_feat"))

        # Ensure all model features exist & numeric
        for col in features:
            if col not in joined.columns: joined[col] = 0.0
            joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)

        X_scaled = scaler.transform(joined[features].values)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_t).squeeze().tolist()

        # Soft expected minutes for downstream (optimizer/UI)
        def nz(col): return pd.to_numeric(joined.get(col, 0), errors="coerce").fillna(0)
        expected_min = (0.6 * nz("roll3MIN") + 0.4 * nz("avgMIN"))
        expected_min = expected_min.where(expected_min > 0, nz("MIN"))
        joined["ExpectedMIN"] = expected_min

        # Output: include P(play) for efficiency weighting client-side
        out = joined[id_cols].copy()
        out["PredictedNextFP"] = preds
        out["CR"] = pd.to_numeric(joined.get("CR", 0.0), errors="coerce").fillna(0.0)
        out["ExpectedMIN"] = joined["ExpectedMIN"]
        out["PPlay"] = pd.to_numeric(joined.get("p_play", 1.0), errors="coerce").fillna(1.0)

        return jsonify(out.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
