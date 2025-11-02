from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Cargar modelo con ruta robusta
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")
modelo = joblib.load(MODEL_PATH)

# Esquema de entrada (sin sentiment_score)
class Features(BaseModel):
    rating: float
    reviews: float
    price: float
    size: float
    days_since_update: float
    category: str  # Texto: p.ej. "BUSINESS", "TOOLS", etc.

@app.post("/predict")
def predict(payload: Features):
    # Si el modelo conoce sus nombres de features, los usamos para alinear
    feat_names = getattr(modelo, "feature_names_in_", None)

    # Inicializa un vector con CERO en todas las columnas esperadas por el modelo
    if feat_names is not None:
        X = pd.DataFrame([[0] * len(feat_names)], columns=list(feat_names))
    else:
        # Fallback (poco probable): usa solo numéricas
        X = pd.DataFrame([{
            "rating": payload.rating,
            "reviews": payload.reviews,
            "price": payload.price,
            "size": payload.size,
            "days_since_update": payload.days_since_update
        }])

    # Carga de NUMÉRICAS (si existen en el modelo)
    for col, val in {
        "rating": payload.rating,
        "reviews": payload.reviews,
        "price": payload.price,
        "size": payload.size,
        "days_since_update": payload.days_since_update
    }.items():
        if col in X.columns:
            X.loc[0, col] = val

    # One-hot de CATEGORY:
    # Buscamos una columna que sea category_<valor> (ignorando mayúsculas/minúsculas y espacios)
    if feat_names is not None:
        cat_cols = [c for c in X.columns if c.lower().startswith("category_")]
        if cat_cols:
            target = payload.category.strip().lower().replace(" ", "_")
            # intenta match exacto
            matched = [c for c in cat_cols if c.lower() == f"category_{target}"]
            # si no hay exacto, intenta por sufijo (p.ej. nombres raros con símbolos)
            if not matched:
                matched = [c for c in cat_cols if c.lower().endswith(f"_{target}")]
            # prende 1 solo si encontramos columna correspondiente
            if matched:
                X.loc[0, matched[0]] = 1
        # Si el modelo espera una columna "category" numérica (poco común), la dejamos en 0 por defecto

    # Predicción
    yhat = float(modelo.predict(X)[0])
    return {"predicted_installs": yhat}
