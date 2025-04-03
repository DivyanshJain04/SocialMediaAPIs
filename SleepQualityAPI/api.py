from fastapi import FastAPI
import pickle
import numpy as np

with open("Social_Media_Impact_on_Sleep_Quality.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(features: list):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
