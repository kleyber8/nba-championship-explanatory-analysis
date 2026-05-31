# 🏀 NBA Playoffs Predictor – Dashboard Predictivo de Campeón

## 📌 Descripción del Proyecto

Dashboard interactivo con **Streamlit** que predice el campeón de los playoffs actuales de la NBA y **explica visualmente** las razones de esa predicción mediante métricas avanzadas y un modelo de machine learning.

Combina datos de la **API oficial de la NBA** y **web scraping** de Basketball Reference para construir características complejas como eficiencia defensiva en el pick & roll, rendimiento en clutch o impacto real de los jugadores.

## 🧠 Planteamiento del Problema

En playoffs, el rendimiento de temporada regular no es suficiente. Las rotaciones se acortan, el ritmo baja y el juego en momentos críticos (clutch) define las series. Los análisis tradicionales usan métricas superficiales (puntos, rebotes) que no capturan esas dinámicas. Necesitamos aislar variables contextuales para predecir y, sobre todo, **explicar** por qué un equipo será campeón.

## 🎯 Objetivos

**General**  
Desarrollar un dashboard en Streamlit que anticipe el campeón de los playoffs actuales mostrando las variables estadísticas clave.

**Específicos**  
1. Construir un pipeline híbrido de extracción (NBA API + scraping de Basketball Reference).  
2. Crear features que reflejen rendimiento en alta presión (net rating, clutch, defensa top‑10, etc.).  
3. Entrenar un modelo predictivo (XGBoost) que calcule probabilidades de victoria por ronda.  
4. Implementar el dashboard con gráficos interactivos (Plotly Express) y explicabilidad (SHAP).

## 🛠️ Tecnologías a utilizar

| Área | Librerías |
|------|------------|
| Extracción | `nba_api`, `bref-scraper`, `requests`, `BeautifulSoup` |
| Procesamiento | `pandas`, `numpy` |
| Feature engineering | `scikit-learn` |
| Modelado | `xgboost`, `scikit-learn`, `joblib` |
| Visualización | `streamlit`, `plotly.express`, `matplotlib`, `shap` |
| Entorno | `python-dotenv` |

## 📁 Estructura del Proyecto
```
nba_playoffs_predictor/
├── .env
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   └── 01_eda_and_modelling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── nba_api_connector.py
│   │   └── bref_scraper.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   └── visualization/
│       ├── __init__.py
│       ├── dashboard.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── header.py
│       │   ├── prediction_card.py
│       │   ├── feature_importance_plot.py
│       │   └── bracket_visual.py
│       ├── pages/
│       │   ├── 1_EDA.py
│       │   └── 2_Prediccion.py
│       ├── session_state.py
│       └── utils.py
└── models/
    └── playoff_predictor.pkl
```


## 📚 Antecedentes y proyectos similares

Referentes activos (2024‑2026):

- **NBA Analytics Dashboard (brycycle99)** – Dashboard en Streamlit con shot charts usando la NBA API.  
- **NBA Game Predictor (johntomlinsonn)** – Ensemble XGBoost + Random Forest con precisión >80%.  
- **Simulación de Playoffs con Monte Carlo (Sanjit Rijesh)** – Simulación del bracket con regresión logística.  
- **NBA Monte Carlo Betting Analyzer** – Combina simulación y análisis en producción.

## ✍️ Autor

**Kleyber Montoya** – estudiante de Estadística y Ciencias Actuariales de la Universidad Central de Venezuela (UCV)

---

**¡Que los datos te lleven al campeón!** 🏆           
