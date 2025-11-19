import os
import base64
import io
import typing
import pickle
import time  # Dùng để đo thời gian
import httpx # Client bất đồng bộ
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image  # Thư viện xử lý ảnh

# -------------------------
# 1. Setup & Config
# -------------------------
# Load file .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Label mapping
LABELS = {0: "clean", 1: "toxic", 2: "very_toxic"}

# Biến chứa model global
ml_models = {}

# -------------------------
# 2. Lifespan (Quản lý load model)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models khi khởi động server."""
    print("\n[STARTUP] Đang load model AI...")
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, "rb") as f:
                ml_models["model"] = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                ml_models["vectorizer"] = pickle.load(f)
            print("[STARTUP] ✅ Model & Vectorizer đã load thành công!")
        else:
            print(f"[STARTUP] ⚠️ Không tìm thấy {MODEL_PATH} hoặc {VECTORIZER_PATH}. Chức năng text sẽ lỗi.")
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
    title="Moderation API Optimized",
    description="Text (Local) + Image (Mock/Google) - High Performance",
    version="2.0",
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

class ImageResponse(BaseModel):
    safe_search: dict
    decision: str
    reason: typing.Optional[str] = None

class CombinedResponse(BaseModel):
    text: typing.Optional[TextResponse]
    image: typing.Optional[ImageResponse]
    processing_time: float 

# -------------------------
# 5. Hàm xử lý Logic (Core)
# -------------------------

def predict_text_logic(text: str):
    """Xử lý text classification."""
    model = ml_models.get("model")
    vectorizer = ml_models.get("vectorizer")

    if not model or not vectorizer:
        # Để tránh crash nếu chưa có model, trả về kết quả giả định
        return {
            "input": text,
            "label_id": -1,
            "label": "model_not_loaded",
            "scores": {}
        }

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    try:
        proba = model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None

    scores = {}
    if proba:
        for i, p in enumerate(proba):
            label_name = LABELS.get(i, str(i))
            scores[label_name] = float(p)
    else:
        for i in LABELS:
            scores[LABELS[i]] = float(1.0 if i == int(pred) else 0.0)

    return {
        "input": text,
        "label_id": int(pred),
        "label": LABELS.get(int(pred), "unknown"),
        "scores": scores
    }


LIKELIHOOD_MAP = {
    "UNKNOWN": 0.0, "VERY_UNLIKELY": 0.05, "UNLIKELY": 0.2,
    "POSSIBLE": 0.5, "LIKELY": 0.8, "VERY_LIKELY": 0.95
}

def resize_image_for_api(image_bytes: bytes, max_size=800) -> bytes:
    """Resize ảnh xuống 800px."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width <= max_size and img.height <= max_size:
            return image_bytes

        img.thumbnail((max_size, max_size))
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception as e:
        print(f"[WARN] Lỗi resize ảnh: {e}. Dùng ảnh gốc.")
        return image_bytes


async def call_google_vision_async(image_bytes: bytes):
    """
    Gọi Google Vision. 
    LƯU Ý: ĐANG BẬT CHẾ ĐỘ GIẢ LẬP (MOCK) ĐỂ TRÁNH LỖI BILLING.
    """
    
    # ==========================================
    # CHẾ ĐỘ GIẢ LẬP (MOCK) - Code mới
    # ==========================================
    print("[MOCK] Đang giả lập kết quả Google (Không tốn tiền/Không cần Billing)...")
    return {
        "safe_search": {
            "adult": {"score": 0.05, "raw": "VERY_UNLIKELY"},
            "violence": {"score": 0.05, "raw": "VERY_UNLIKELY"},
            "racy": {"score": 0.05, "raw": "VERY_UNLIKELY"}
        },
        "decision": "clean",
        "reason": None
    }

    # ==========================================
    # CODE THẬT (Khi nào bạn nạp tiền Google thì mở đoạn dưới này ra và xóa đoạn trên)
    # ==========================================
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("Chưa cấu hình GOOGLE_API_KEY.")

    optimized_bytes = resize_image_for_api(image_bytes)
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    image_base64 = base64.b64encode(optimized_bytes).decode("utf-8")
    
    payload = {
        "requests": [{
            "image": {"content": image_base64},
            "features": [{"type": "SAFE_SEARCH_DETECTION"}]
        }]
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload, timeout=30.0)

    if res.status_code != 200:
        raise RuntimeError(f"Google API trả lỗi {res.status_code}: {res.text}")

    data = res.json()
    try:
        annotation = data["responses"][0]["safeSearchAnnotation"]
    except (KeyError, IndexError):
        annotation = {"adult": "UNKNOWN", "violence": "UNKNOWN", "racy": "UNKNOWN"}

    converted = {k: {"raw": v, "score": LIKELIHOOD_MAP.get(v, 0.0)} 
                 for k, v in annotation.items()}

    reasons = []
    for cat in ["adult", "violence", "racy"]:
        if converted.get(cat, {}).get("score", 0) >= 0.8:
            reasons.append(cat)

    decision = "reject" if reasons else "clean"

    return {
        "safe_search": converted,
        "decision": decision,
        "reason": ", ".join(reasons) if reasons else None
    }
    """

# -------------------------
# 6. Endpoints
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_models.get("model") is not None
    }

@app.post("/predict_text", response_model=TextResponse)
def predict_text_endpoint(body: TextRequest):
    try:
        return predict_text_logic(body.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict_image", response_model=ImageResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh.")
    
    content = await file.read()
    try:
        return await call_google_vision_async(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=CombinedResponse)
async def predict_combined(
    text: typing.Optional[str] = Form(None),
    file: typing.Optional[UploadFile] = File(None)
):
    start_time = time.time()
    print(f"\n--- [REQ] Bắt đầu request ---")
    
    text_res = None
    image_res = None

    # Xử lý Text
    if text:
        t0 = time.time()
        try:
            text_res = predict_text_logic(text)
            print(f"   [TIMER] Text AI: {time.time() - t0:.3f}s")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"Text Error: {e}")

    # Xử lý Image
    if file:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
             raise HTTPException(status_code=400, detail="File không đúng định dạng ảnh.")

        t1 = time.time()
        content = await file.read()
        print(f"   [TIMER] Đọc file ({len(content)/1024:.1f} KB): {time.time() - t1:.3f}s")
        
        t2 = time.time()
        try:
            image_res = await call_google_vision_async(content)
            print(f"   [TIMER] AI Image (Mock/Google): {time.time() - t2:.3f}s")
        except Exception as e:
            print(f"   [ERROR] Image Error: {e}")
            raise HTTPException(status_code=500, detail=f"Image Error: {str(e)}")

    total_time = time.time() - start_time
    print(f"--- [END] Tổng thời gian: {total_time:.3f}s ---\n")

    if not text_res and not image_res:
        raise HTTPException(status_code=400, detail="Vui lòng gửi 'text' hoặc 'file'.")

    return {
        "text": text_res, 
        "image": image_res,
        "processing_time": total_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)