from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

app = FastAPI(title="GooglePlay Predictor")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf_model.pkl")
modelo = joblib.load(MODEL_PATH)

class Features(BaseModel):
    rating: float
    reviews: float
    price: float
    size: float
    days_since_update: float
    category: str

@app.post("/predict")
def predict(payload: Features):
    try:
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

            cat_cols = [c for c in X.columns if c.lower().startswith("category_")]
            if cat_cols:
                target = payload.category.strip().lower().replace(" ", "_")
                match = [c for c in cat_cols if c.lower() == f"category_{target}"]
                if match:
                    X.loc[0, match[0]] = 1
        else:
            X = pd.DataFrame([payload.dict()])

        yhat = float(modelo.predict(X)[0])
        return {"predicted_installs": yhat}

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        return Response(content=f"ERROR:\n{err}", media_type="text/plain", status_code=500)




