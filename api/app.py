from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

app = FastAPI(title="GooglePlay Predictor")

# Ruta robusta del modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")
modelo = joblib.load(MODEL_PATH)

class Features(BaseModel):
    rating: float
    reviews: float
    price: float
    size: float
    days_since_update: float
    category: str  # p.ej. "BUSINESS"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model_features")
def model_features():
    names = getattr(modelo, "feature_names_in_", None)
    return {"feature_names_in_": None if names is None else list(names)}

@app.post("/predict")
def predict(payload: Features):
    try:
        # Construcción del vector según lo que el modelo espera
        feat_names = getattr(modelo, "feature_names_in_", None)
        if feat_names is not None:
            X = pd.DataFrame([[0] * len(feat_names)], columns=list(feat_names))

            num_map = {
                "rating": payload.rating,
                "reviews": payload.reviews,
                "price": payload.price,
                "size": payload.size,
                "days_since_update": payload.days_since_update,
            }
            for col, val in num_map.items():
                if col in X.columns:
                    X.loc[0, col] = val
                elif col.capitalize() in X.columns:
                    X.loc[0, col.capitalize()] = val
                elif col.upper() in X.columns:
                    X.loc[0, col.upper()] = val

            cat_cols = [c for c in X.columns if c.lower().startswith("category_")]
            if cat_cols:
                target = payload.category.strip().lower().replace(" ", "_")
                match = [c for c in cat_cols if c.lower() == f"category_{target}"]
                if not match:
                    match = [c for c in cat_cols if c.lower().endswith(f"_{target}")]
                if match:
                    X.loc[0, match[0]] = 1
        else:
            X = pd.DataFrame([payload.dict()])

        yhat = float(modelo.predict(X)[0])
        return {"predicted_installs": yhat, "used_columns": list(X.columns)}

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        return Response(content=f"ERROR:\n{err}", media_type="text/plain", status_code=500)



