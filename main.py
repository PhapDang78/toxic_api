import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------
# 1. Setup & Config
# -------------------------
# Load file .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Label mapping
# Tùy thuộc vào model của bạn, bạn có thể chỉ có 0: clean, 1: toxic
LABELS = {0: "clean", 1: "toxic"} 

# Biến chứa model global
ml_models = {}

# -------------------------
# 2. Lifespan (Quản lý load model)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model Text khi khởi động server."""
    print("\n[STARTUP] Đang load model Text AI...")
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, "rb") as f:
                ml_models["model"] = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                ml_models["vectorizer"] = pickle.load(f)
            print("[STARTUP] ✅ Model & Vectorizer đã load thành công!")
        else:
            print(f"[STARTUP] ⚠️ Không tìm thấy {MODEL_PATH} hoặc {VECTORIZER_PATH}. Server sẽ hoạt động nhưng chức năng dự đoán sẽ báo lỗi.")
    except Exception as e:
        print(f"[STARTUP] ❌ Lỗi load model: {e}")
        ml_models["model"] = None
        ml_models["vectorizer"] = None
    
    yield
    
    ml_models.clear()
    print("[SHUTDOWN] Đã giải phóng tài nguyên.")

# -------------------------
# 3. Khởi tạo FastAPI
# -------------------------
app = FastAPI(
    title="Toxic Text Moderation API",
    description="API đơn giản hóa, chỉ xử lý phân loại Ngôn ngữ Độc hại.",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 4. Models Request & Response
# -------------------------
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    input: str
    label_id: int
    label: str
    scores: dict
    processing_time: float # Thêm thời gian xử lý

# -------------------------
# 5. Hàm xử lý Logic (Core)
# -------------------------

def predict_text_logic(text: str):
    """Xử lý text classification."""
    start_time = time.time()
    model = ml_models.get("model")
    vectorizer = ml_models.get("vectorizer")

    if not model or not vectorizer:
        raise RuntimeError("Model phân loại ngôn ngữ chưa được load.")

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    try:
        # Lấy xác suất của từng lớp
        proba = model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None
        
    scores = {}
    if proba:
        for i, p in enumerate(proba):
            label_name = LABELS.get(i, str(i))
            scores[label_name] = round(float(p), 4) # Làm tròn 4 chữ số
    else:
        # Trường hợp không có predict_proba (rất hiếm)
        for i in LABELS:
            scores[LABELS[i]] = float(1.0 if i == int(pred) else 0.0)

    total_time = time.time() - start_time
    
    return {
        "input": text,
        "label_id": int(pred),
        "label": LABELS.get(int(pred), "unknown"),
        "scores": scores,
        "processing_time": round(total_time, 4)
    }

# -------------------------
# 6. Endpoints
# -------------------------

@app.get("/health")
def health():
    """Kiểm tra trạng thái server và model."""
    return {
        "status": "ok",
        "model_loaded": ml_models.get("model") is not None
    }

@app.post("/predict", response_model=TextResponse) # Đổi tên endpoint thành /predict
def predict_text_endpoint(body: TextRequest):
    """
    Endpoint chính để dự đoán nhãn độc hại của văn bản.
    Body: { "text": "..." }
    """
    try:
        return predict_text_logic(body.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi không xác định: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Chạy trên port 5000 (hoặc PORT trong .env)
    port = int(os.getenv("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)