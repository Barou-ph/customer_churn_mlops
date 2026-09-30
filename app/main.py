import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# 1. Định nghĩa lại đúng kiến trúc Model (16 nodes ẩn)
class ChurnModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),  # Khớp với model.pt
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# 2. Khởi tạo App
app = FastAPI(title="Customer Churn Prediction API")

# Load Scaler và Model
try:
    scaler = joblib.load("scaler.pkl")
    model = ChurnModel()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
except Exception as e:
    print(f"⚠️ Lỗi khi load artifacts: {e}")


# 3. Schema Pydantic
class CustomerData(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    age: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is Running!"}


@app.post("/predict")
async def predict(data: CustomerData):
    if data.age <= 0 or data.tenure < 0:
        raise HTTPException(
            status_code=400, detail="Age and Tenure must be positive value!"
        )

    try:
        raw_features = [
            [data.tenure, data.monthly_changes, data.total_changes, data.age]
        ]
        scaled_features = scaler.transform(raw_features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()

        return {
            "churn_prob": round(prob, 4),
            "is_churn": bool(prob > 0.5),
            "status": "Success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferene error: {str(e)}")
