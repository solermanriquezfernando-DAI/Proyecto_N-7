Proyecto: Predicción de descargas en Google Play
Objetivo

Construir un modelo predictivo para estimar la cantidad de instalaciones que tendrá una aplicación en Google Play Store usando variables numéricas y categóricas.
Entregar una API funcional para realizar predicciones.

Dataset

Fuente: Kaggle – Google Play Store Apps + User Reviews.

Variables usadas:

Rating

Reviews

Price

Size

Days since last update

Category
(Se agregó análisis de sentimiento para mejorar precisión)

Proceso

Carga y limpieza de datos.

EDA básico.

Entrenamiento modelo baseline.

Tuning e ingeniería simple.

Modelo final entrenado.

Exportación del modelo (joblib).

Construcción y prueba de API FastAPI.

Presentación final.

Modelo

Algoritmo: Random Forest Regressor
Métrica principal: R²

Versión	R²	RMSE	MAE
Baseline	0.72	1.85M	0.89M
Final (despliegue)	0.93	0.956 (log)	0.699 (log)

Interpretación: modelo con precisión alta. Estimaciones consistentes.

Requerimientos

Instalar dependencias:

pip install -r requirements.txt

Estructura del repo
notebooks/proyecto_google_play.ipynb
data/instrucciones.txt
models/best_rf_model.pkl
api/app.py
docs/presentacion.pdf
README.md
requirements.txt

Ejecutar localmente (API)

Desde la raíz del repositorio:

cd api
uvicorn app:app --host 127.0.0.1 --port 8000


Endpoints:

Salud: http://127.0.0.1:8000/health

Docs: http://127.0.0.1:8000/docs

Ejemplo de request POST /predict
{
  "Rating": 4.5,
  "Reviews": 12000,
  "Price_usd": 0,
  "Size_mb": 25,
  "days_since_update": 60,
  "Category": "GAME"
}

Ejemplo de respuesta
{
  "pred_installs_log": 13.65,
  "pred_installs": 850599
}

Pruebas rápidas (terminal)

Valida correcto:

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"Rating":4.5,"Reviews":12000,"Price_usd":0,"Size_mb":25,"days_since_update":60,"Category":"GAME"}'


Valida manejo error:

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"Rating":4.5,"Reviews":12000,"Price_usd":0,"Size_mb":25,"days_since_update":60,"Category":"XXXX"}'

Nota final

El modelo se carga desde models/best_rf_model.pkl.

La categoría debe estar en el vocabulario original del dataset.

La API está lista para despliegue en Render/Railway si se requiere.

El modelo entrenado se encuentra incluido localmente como `best_rf_model.pkl` (104MB).
No se sube por límite de GitHub.
Si se requiere para ejecutar, solicitarlo y se comparte por Drive.
Ruta esperada: /models/best_rf_model.pkl
