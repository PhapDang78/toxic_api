from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI()

# Load model + vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: PredictRequest):
    text = data.input

    # Transform text → vector
    X = vectorizer.transform([text])

    # Predict class
    pred = model.predict(X)[0]

    # Predict probability
    proba = model.predict_proba(X)[0]
    
    return {
        "input": text,
        "label_id": int(pred),
        "score": {
            "clean": float(proba[0]),
            "toxic": float(proba[1]),
            "very_toxic": float(proba[2]) if len(proba) > 2 else None
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
