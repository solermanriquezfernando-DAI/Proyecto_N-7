📈 Predicción de Descargas de Apps en Google Play

Modelo predictivo para estimar la cantidad de instalaciones que tendrá una aplicación en Google Play Store.
Se utilizan variables numéricas, categóricas y análisis de sentimiento de reviews.
Incluye API para predicciones en producción.

🧠 Objetivo del Proyecto

Construir y desplegar un modelo capaz de predecir la cantidad estimada de descargas de una app en Google Play, integrando:

Datos de aplicaciones + reviews (Kaggle)

Feature engineering

Entrenamiento y optimización del modelo

Exportación del modelo final

API para inferencia

📂 Estructura del Proyecto
Proyecto_N-7/
│── api/                 # API FastAPI
│── data/                # Dataset + instrucciones de descarga
│── docs/                # PPT y documentación
│── models/              # Modelo entrenado (link Drive)
│── notebooks/           # Jupyter notebook del pipeline ML
│── requirements.txt     # Dependencias
└── README.md            # Este archivo

📊 Dataset

Fuente: Kaggle – Google Play Store Apps + User Reviews

Enlace dataset apps:
https://www.kaggle.com/datasets/lava18/google-play-store-apps

Enlace dataset reviews:
https://www.kaggle.com/datasets/lava18/google-play-store-user-reviews

🔧 Variables Utilizadas
Tipo	Variables
Numéricas	Rating, Reviews, Price, Size, Days since last update
Categóricas	Category
Texto	Sentiment score desde reviews
🔨 Proceso

Carga y limpieza de datos

Feature engineering

Pipeline ML

Entrenamiento modelo baseline

Optimización

Exportación del modelo

Integración API

✅ Modelo Final

Modelo: Random Forest Regressor
Métrica utilizada: RMSE / R²
Mejor resultado:

Se ajusta según resultados del notebook (completa tú aquí)

📁 Modelo (Drive):
https://drive.google.com/file/d/1W_geXAFiSmmeBYbKMRuz9RTwTN-VtatT/view?usp=drive_link

Guardar como:

models/best_rf_model.pkl

🚀 Cómo Ejecutar
1) Clonar repositorio
git clone https://github.com/solermanriquezfernando-DAI/Proyecto_N-7.git
cd Proyecto_N-7

2) Instalar dependencias
pip install -r requirements.txt

3) Ejecutar API
uvicorn api.app:app --reload

4) Endpoint de prueba
http://127.0.0.1:8000/predict

🛰 Ejemplo Request (JSON)
{
  "rating": 4.3,
  "reviews": 265000,
  "price": 0,
  "size": 25,
  "days_since_update": 30,
  "category": "TOOLS",
  "sentiment_score": 0.78
}

📦 Ejemplo Response
{
  "predicted_installs": 5200000
}

📒 Notebook del Proyecto

Ruta:

/notebooks/Proyecto_goggle_play.ipynb

📎 Presentación (PDF)

Ruta:

/docs/

🛠 Tecnologías

Python

Scikit-learn

Pandas / NumPy

FastAPI

Uvicorn

Google Colab / Jupyter

📌 Próximos pasos

Dockerfile

Deploy API (Render / Railway)

Dashboard de resultados

Monitoreo de drift

👤 Autor

Fernando Soler
Control de Proyectos | Machine Learning | Data Analytics

📜 Licencia

MIT
