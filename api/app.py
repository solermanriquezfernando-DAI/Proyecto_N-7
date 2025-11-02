from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

# Ruta del modelo (funciona local y en GitHub + Colab)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")

# Cargar modelo
modelo = joblib.load(model_path)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = modelo.predict(df)[0]
    return {"prediccion_installs": float(pred)}
