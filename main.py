import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- THƯ VIỆN TRANSFORMER ---
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------------------------
# 1. Setup & Config
# -------------------------
# Load file .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)

# Mô hình Transformer mạnh hơn cho Tiếng Việt
MODEL_NAME = "jesse-tong/vietnamese_hate_speech_detection_phobert"

# Cấu hình nhãn (Cần kiểm tra lại nếu mô hình có nhiều hơn 2 nhãn)
LABELS = {0: "clean", 1: "toxic"} 
ml_models = {}

# -------------------------
# 2. Lifespan (Quản lý load model)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load mô hình PhoBERT khi khởi động server."""
    print("\n[STARTUP] Đang load mô hình PhoBERT Transformer...")
    try:
        # Tải Tokenizer và Model
        ml_models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Sử dụng AutoModelForSequenceClassification cho nhiệm vụ phân loại
        ml_models["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        ml_models["model"].eval() # Đặt model ở chế độ đánh giá
        print(f"[STARTUP] ✅ PhoBERT ({MODEL_NAME}) đã load thành công!")
    except Exception as e:
        print(f"[STARTUP] ❌ Lỗi load mô hình Transformer: {e}")
        ml_models["model"] = None
        ml_models["tokenizer"] = None
    
    yield
    
    ml_models.clear()
    print("[SHUTDOWN] Đã giải phóng tài nguyên.")

# -------------------------
# 3. Khởi tạo FastAPI
# -------------------------
app = FastAPI(
    title="Toxic Text Moderation API (PhoBERT)",
    description="API chỉ xử lý phân loại Ngôn ngữ Độc hại tiếng Việt bằng mô hình Transformer.",
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
    processing_time: float 

# -------------------------
# 5. Hàm xử lý Logic (Core)
# -------------------------

def predict_text_logic(text: str):
    """Xử lý text classification bằng mô hình PhoBERT."""
    start_time = time.time()
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")

    if not model or not tokenizer:
        raise RuntimeError("Mô hình phân loại ngôn ngữ chưa được load.")

    # 1. Mã hóa văn bản (Tokenize)
    inputs = tokenizer(text, 
                       return_tensors="pt", 
                       truncation=True, 
                       padding=True, 
                       max_length=128) # Giới hạn độ dài token

    # 2. Dự đoán (không cần tính gradient)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Tính xác suất (Softmax)
    # Lấy xác suất của từng lớp
    probs = F.softmax(outputs.logits, dim=1)[0]
    
    # 4. Lấy nhãn dự đoán
    pred_idx = torch.argmax(probs).item()

    scores = {}
    for i, p in enumerate(probs.tolist()):
        label_name = LABELS.get(i, str(i))
        scores[label_name] = round(float(p), 4)

    total_time = time.time() - start_time
    
    return {
        "input": text,
        "label_id": pred_idx,
        "label": LABELS.get(pred_idx, "unknown"),
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

@app.post("/predict", response_model=TextResponse)
def predict_text_endpoint(body: TextRequest):
    """
    Endpoint chính để dự đoán nhãn độc hại của văn bản.
    Body: { "text": "..." }
    """
    try:
        return predict_text_logic(body.text)
    except RuntimeError as e:
        # Lỗi 503 nếu model chưa load thành công
        raise HTTPException(status_code=503, detail=str(e)) 
    except Exception as e:
        # Lỗi 500 cho các lỗi không xác định
        print(f"Lỗi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ khi dự đoán: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)