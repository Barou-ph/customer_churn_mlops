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
def predict(data: CustomerData):
    try:
        # Chuyển dữ liệu đầu vào thành Mảng 2D [[tenure, monthly_charges, total_charges, age]]
        raw_features = [
            [data.tenure, data.monthly_charges, data.total_charges, data.age]
        ]

        # Standardize dữ liệu
        scaled_features = scaler.transform(raw_features)

        # Chuyển sang Tensor PyTorch
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        # Dự đoán
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()

        churn_prediction = int(prob > 0.5)

        return {
            "churn_probability": round(prob, 4),
            "is_churn": bool(churn_prediction),
            "status": "Success",
        }
    except Exception as e:
        # Nếu có lỗi sẽ trả về chi tiết thay vì lỗi 500 chung chung
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
