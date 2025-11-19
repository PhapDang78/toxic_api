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
def predict(data: InputText):
    X = vectorizer.transform([data.text])
    pred = model.predict(X)[0]

    return {
        "input": data.text,
        "label_id": int(pred)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
