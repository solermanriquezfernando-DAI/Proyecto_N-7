from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Cargar modelo
modelo = joblib.load("../models/best_rf_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = modelo.predict(df)[0]
    return {"prediccion_installs": float(pred)}
