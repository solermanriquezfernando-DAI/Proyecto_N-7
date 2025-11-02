📈 Predicción de Descargas de Apps en Google Play

Proyecto de Machine Learning para predecir la cantidad de instalaciones de una aplicación en Google Play Store, integrando datos de apps y análisis de sentimiento de reviews. Incluye API funcional para realizar predicciones en tiempo real.

🌐 API en Línea

URL Pública API:
https://pleurocarpous-wilbert-forwardly.ngrok-free.dev

Swagger Docs:
https://pleurocarpous-wilbert-forwardly.ngrok-free.dev/docs

Requiere mantener ngrok activo localmente (deploy demostrativo).

🧠 Objetivo

Construir y desplegar un modelo predictivo que estime la cantidad de instalaciones de apps usando:

Datos estructurados (Google Play)

Sentimiento de reviews

Feature engineering

Random Forest optimizado

API para inferencia

📊 Dataset

Fuente: Kaggle

Apps: https://www.kaggle.com/datasets/lava18/google-play-store-apps

Reviews: https://www.kaggle.com/datasets/lava18/google-play-store-user-reviews

🔧 Variables Utilizadas
Tipo	Variables
Numéricas	Rating, Reviews, Price, Size, Days_since_update
Categóricas	Category
Texto/Sentimiento	Sentiment score desde reviews
🏗 Pipeline

Limpieza y preparación de datos

Feature Engineering

Entrenamiento baseline

Optimización Random Forest

Exportación de modelo

API FastAPI + Uvicorn

✅ Modelo Final

Modelo: Random Forest Regressor
Métricas Finales:

R²: COMPLETAR
RMSE: COMPLETAR
MAE: COMPLETAR


Completa con tus valores del notebook antes de entregar.

📦 Modelo Entrenado

Ruta local esperada:

Proyecto_N-7/models/best_rf_model.pkl


Drive:
https://drive.google.com/file/d/1W_geXAFiSmmeBYbKMRuz9RTwTN-VtatT/view

📂 Estructura del Proyecto
Proyecto_N-7/
│── api/                  # API FastAPI
│── data/                 # Datasets / instrucciones descarga
│── docs/                 # PPT y documentación
│── models/               # best_rf_model.pkl
│── notebooks/            # Desarrollo ML
│── requirements.txt
└── README.md

🚀 Cómo Ejecutar Localmente
git clone https://github.com/solermanriquezfernando-DAI/Proyecto_N-7.git
cd Proyecto_N-7
pip install -r requirements.txt
uvicorn api.app:app --reload


Probar en Swagger:
http://127.0.0.1:8000/docs

🛰 Ejemplo Request
{
 "rating": 4.3,
 "reviews": 265000,
 "price": 0,
 "size": 25,
 "days_since_update": 30,
 "category": "TOOLS",
 "sentiment_score": 0.78
}

📎 Response
{
 "predicted_installs": 5200000
}

🎥 Demo de Funcionamiento

Incluye:

✅ Levantamiento API
✅ Predicción real
✅ Test vía navegador / Swagger

(Profesor: solicitar video si es necesario)

🧠 Reflexión Técnica

El modelo captura bien patrones generales de instalación. Presenta limitaciones en valores extremos por distribución heavy-tailed. Futuros pasos incluyen refinamiento log-transforms, ajuste Bayesian Optimization y deploy permanente en Railway/Render con contenedor Docker.

🛠 Tecnologías

Python

Scikit-learn

Pandas / NumPy

FastAPI

Uvicorn

Google Colab

Ngrok

👤 Autor

Fernando Soler Manriquez 
Control de Proyectos | Machine Learning | Data Analytics
Repositorio: https://github.com/solermanriquezfernando-DAI
