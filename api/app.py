from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Ruta robusta del modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")
modelo = joblib.load(MODEL_PATH)

# Esquema de entrada (sin sentiment_score)
class Features(BaseModel):
    rating: float
    reviews: float
    price: float
    size: float
    days_since_update: float
    category: str

@app.post("/predict")
def predict(payload: Features):
    # DataFrame con las columnas originales
    df = pd.DataFrame([payload.dict()])

    # Si el modelo tiene nombres de features, alinear
    feat_names = getattr(modelo, "feature_names_in_", None)
    if feat_names is not None:
        # Completar faltantes con 0 si existieran
        for col in feat_names:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in feat_names if c in df.columns]]

    # Predicción
    yhat = float(modelo.predict(df)[0])
    return {"predicted_installs": yhat}
