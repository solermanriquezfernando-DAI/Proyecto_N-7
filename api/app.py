from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

app = FastAPI(title="GooglePlay Predictor")

# Carga robusta del modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")
modelo = joblib.load(MODEL_PATH)

class Features(BaseModel):
    rating: float
    reviews: float
    price: float
    size: float
    days_since_update: float
    category: str  # texto, ej: "BUSINESS"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model_features")
def model_features():
    names = getattr(modelo, "feature_names_in_", None)
    if names is None:
        return {"feature_names_in_": None, "note": "El modelo no expone feature_names_in_ (puede ser un Pipeline antiguo)."}
    return {"feature_names_in_": list(names)}

@app.post("/predict")
def predict(payload: Features):
    try:
        # 1) Si el modelo expone los nombres de features, armamos ese vector
        feat_names = getattr(modelo, "feature_names_in_", None)
        if feat_names is not None:
            X = pd.DataFrame([[0]*len(feat_names)], columns=list(feat_names))

            # Cargar numéricas si existen
            numericas = {
                "rating": payload.rating,
                "reviews": payload.reviews,
                "price": payload.price,
                "size": payload.size,
                "days_since_update": payload.days_since_update
            }
            for col, val in numericas.items():
                if col in X.columns:
                    X.loc[0, col] = val

            # Encender one-hot de category si existe alguna columna tipo category_*
            cat_cols = [c for c in X.columns if c.lower().startswith("category_")]
            if cat_cols:
                target = payload.category.strip().lower().replace(" ", "_")
                # match exacto
                match = [c for c in cat_cols if c.lower() == f"category_{target}"]
                if not match:
                    # fallback por sufijo
                    match = [c for c in cat_cols if c.lower().endswith(f"_{target}")]
                if match:
                    X.loc[0, match[0]] = 1
            # Si el preprocesamiento de category está dentro de un Pipeline y NO hay columnas category_*, el modelo se encargará (caso raro si feature_names_in_ es None)
        else:
            # 2) Fallback: el modelo NO expone features; mandamos las columnas crudas (posible si model = Pipeline)
            X = pd.DataFrame([{
                "rating": payload.rating,
                "reviews": payload.reviews,
                "price": payload.price,
                "size": payload.size,
                "days_since_update": payload.days_since_update,
                "category": payload.category
            }])

        yhat = float(modelo.predict(X)[0])
        return {"predicted_installs": yhat, "used_columns": list(X.columns)}
    except Exception as e:
        # Devolver el error real para depurar sin consola
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        raise HTTPException(status_code=500, detail={"error": err})
