import os
import base64
import io
import typing
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import requests
from PIL import Image

# -------------------------
# Config
# -------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional, dùng cho image moderation
# nếu bạn muốn dùng HuggingFace/DeepAI thay cho Google Vision, có thể chỉnh hàm check_image_google()

# Label mapping (đổi theo dataset bạn train)
LABELS = {
    0: "clean",
    1: "toxic",
    2: "very_toxic"
}

# -------------------------
# Load model + vectorizer
# -------------------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"[WARN] Không load được model từ {MODEL_PATH}: {e}")

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    vectorizer = None
    print(f"[WARN] Không load được vectorizer từ {VECTORIZER_PATH}: {e}")

# -------------------------
# FastAPI init
# -------------------------
app = FastAPI(title="Moderation API (text + image)",
              description="Text toxicity + Image SafeSearch moderation",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: thay bằng domain của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response models
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

# -------------------------
# Helpers: text moderation
# -------------------------
def predict_text(text: str):
    """Dùng model local để predict label + probabilities."""
    if model is None or vectorizer is None:
        raise RuntimeError("Model hoặc vectorizer chưa được load. Hãy chắc chắn đã upload model.pkl và vectorizer.pkl.")

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    # Nếu model không hỗ trợ predict_proba -> trả confidence = 1.0 cho nhãn dự đoán
    try:
        proba = model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None

    # build scores dict (map theo LABELS)
    scores = {}
    if proba is not None:
        for i, p in enumerate(proba):
            label_name = LABELS.get(i, str(i))
            scores[label_name] = float(p)
    else:
        # fallback: đặt 1.0 cho nhãn dự đoán
        for i in LABELS:
            scores[LABELS[i]] = float(1.0 if i == int(pred) else 0.0)

    return {
        "input": text,
        "label_id": int(pred),
        "label": LABELS.get(int(pred), str(int(pred))),
        "scores": scores
    }

# -------------------------
# Helpers: image moderation (Google Vision SafeSearch)
# -------------------------
LIKELIHOOD_MAP = {
    "UNKNOWN": 0.0,
    "VERY_UNLIKELY": 0.05,
    "UNLIKELY": 0.2,
    "POSSIBLE": 0.5,
    "LIKELY": 0.8,
    "VERY_LIKELY": 0.95
}

def call_google_vision_safe_search(image_bytes: bytes):
    """Gọi Google Vision REST API SafeSearch (cần GOOGLE_API_KEY)."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY không được đặt. Thiết lập biến môi trường GOOGLE_API_KEY để dùng Google Vision SafeSearch.")

    endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    body = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "SAFE_SEARCH_DETECTION"}]
            }
        ]
    }
    res = requests.post(endpoint, json=body, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Google Vision API lỗi: {res.status_code} {res.text}")

    data = res.json()
    try:
        annotation = data["responses"][0]["safeSearchAnnotation"]
    except Exception as e:
        raise RuntimeError(f"Không parse được response từ Google Vision: {e}")

    # Convert to numeric scores
    converted = {}
    for k, v in annotation.items():
        converted[k] = {
            "raw": v,
            "score": LIKELIHOOD_MAP.get(v, 0.0)
        }

    # Decision rule (bạn có thể điều chỉnh ngưỡng)
    reasons = []
    decision = "clean"
    # nếu adult hoặc violence hoặc racy >= LIKELY => reject
    threshold = 0.8
    if converted.get("adult", {}).get("score", 0.0) >= threshold:
        decision = "reject"
        reasons.append("adult content")
    if converted.get("violence", {}).get("score", 0.0) >= threshold:
        decision = "reject"
        reasons.append("violence")
    if converted.get("racy", {}).get("score", 0.0) >= threshold:
        # racy thường là khoả thân nhẹ, tuỳ ứng dụng có thể accept/reject
        decision = "reject"
        reasons.append("racy")

    return {
        "safe_search": converted,
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else None
    }

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "google_vision": bool(GOOGLE_API_KEY)
    }

@app.post("/predict_text", response_model=TextResponse)
def predict_text_endpoint(body: TextRequest):
    try:
        r = predict_text(body.text)
        return r
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_image", response_model=ImageResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Upload file multipart/form-data (file field).
    Accepts common image types.
    """
    content = await file.read()
    try:
        r = call_google_vision_safe_search(content)
        return r
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.post("/predict", response_model=CombinedResponse)
async def predict_combined(
    text: typing.Optional[str] = Body(None),
    file: typing.Optional[UploadFile] = File(None)
):
    """
    Combined endpoint:
    - send JSON body field `text` (optional)
    - optionally upload image file as multipart (field name 'file')
    Example using Postman: form-data -> key 'text' (text), key 'file' (file)
    """
    text_res = None
    image_res = None

    if text:
        try:
            text_res = predict_text(text)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    if file:
        content = await file.read()
        try:
            image_res = call_google_vision_safe_search(content)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Phải gửi text hoặc file (hoặc cả hai).")

    return {
        "text": text_res,
        "image": image_res
    }

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
