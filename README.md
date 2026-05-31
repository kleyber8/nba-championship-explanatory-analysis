# 🏆 NBA Playoffs 2026: ¿Qué Hace a un Campeón? Un Análisis Explicativo con Machine Learning

## 📖 Contexto y Motivación

Las finales de la NBA de 2026 nos regalaron un duelo inédito y cargado de simbolismo: los **San Antonio Spurs**, una dinastía que renació de sus cenizas liderada por el prodigio Victor Wembanyama, se enfrentaron a los **New York Knicks**, una franquicia histórica que volvía a unas finales 27 años después. Este escenario plantea una pregunta fascinante: *¿Qué factores estadísticos transformaron a estos dos equipos, que llevaban años sin pisar las finales, en el campeón y subcampeón de la liga?*

La respuesta no se encuentra en las estadísticas tradicionales. Mientras que los Knicks, comandados por Jalen Brunson, destacaron en ataque, los Spurs construyeron su éxito sobre una defensa histórica. **El verdadero valor de este proyecto reside en pasar de la especulación a la explicación basada en datos.** No buscamos predecir quién ganó (eso ya es historia), sino **descomponer y jerarquizar las métricas avanzadas que definieron al campeón**, ofreciendo una herramienta analítica para entender la ciencia detrás de un título de la NBA.

## 🧠 Planteamiento del Problema

En los playoffs de la NBA, el contexto del juego cambia drásticamente: las rotaciones se acortan, el ritmo baja, y la eficiencia en situaciones de *clutch* se vuelve crítica. Las métricas de la temporada regular, como los puntos por partido o el porcentaje de victorias, son predictores débiles del éxito en postemporada. Este proyecto aborda el problema de identificar y cuantificar qué variables avanzadas (por ejemplo, la eficiencia defensiva en el pick & roll, el *net rating* en playoffs o el rendimiento en momentos críticos) son las que realmente separan a un campeón de un finalista.

## 🎯 Objetivos

### 🥇 Objetivo General

Construir un pipeline analítico completo que, partiendo de datos históricos de la NBA (2020-2025), entrene un modelo de Machine Learning (XGBoost) para determinar, de forma explicativa, el conjunto de métricas avanzadas que definieron al campeón de los playoffs de 2026 (San Antonio Spurs), y presentar estos hallazgos a través de un dashboard interactivo en Streamlit.

### 📋 Objetivos Específicos

1.  **Ingesta y Preparación de Datos (Pipeline de Datos):** Implementar un proceso de extracción de la NBA API (datos estructurados de 2020 a 2025), guardando los datos crudos en la carpeta `data/raw/` para su posterior procesamiento.
2.  **Ingeniería de Características (*Feature Engineering*):** Limpiar, transformar y agregar los datos para construir un conjunto robusto de variables predictivas que reflejen el rendimiento real en playoffs, como el *Net Rating*, la eficiencia defensiva en el *pick & roll*, el *EFG%* en *clutch*, y la experiencia en postemporada.
3.  **Modelado Estadístico Explicativo:** Entrenar y optimizar un modelo de clasificación (XGBoost) con los datos históricos (2020-2025). El objetivo del modelo no es predecir el futuro, sino aprender a identificar patrones de campeón. Posteriormente, se aplicará sobre los datos de los finalistas de 2026 (Spurs y Knicks) para, utilizando SHAP (*SHapley Additive exPlanations*), descomponer y jerarquizar los factores que más influyeron en la clasificación del modelo a favor de los Spurs.
4.  **Desarrollo del Dashboard Explicativo:** Diseñar e implementar un dashboard modular en Streamlit que sirva como interfaz de usuario, mostrando visualizaciones interactivas (Plotly Express) que comparen las métricas clave de ambos equipos y expongan, de manera clara y pedagógica, el *porqué* del campeonato según el modelo.

## 🛠️ Tecnologías Utilizadas

El proyecto se construye sobre un stack moderno de ciencia de datos:

| Área | Librerías / Tecnologías |
|------|--------------------------|
| Extracción de Datos | `nba_api`, `requests`, `pandas` |
| Procesamiento y Análisis | `pandas`, `numpy` |
| Feature Engineering | `scikit-learn` |
| Modelado Predictivo | `xgboost`, `scikit-learn`, `joblib` |
| Explicabilidad del Modelo | `shap` |
| Visualización y Dashboard | `streamlit`, `plotly.express`, `matplotlib` |
| Gestión del Entorno | `python-dotenv`, `venv` |

## 📚 Antecedentes y Proyectos Similares

Este trabajo se apoya en la creciente literatura de *sports analytics* y en proyectos de código abierto que han abordado problemáticas similares. A continuación, se presentan los referentes más relevantes que inspiran y validan nuestro enfoque:

### Proyectos de Referencia en GitHub

*   **NBA-Championship-Prediction-Model (alexherb3):** Este proyecto sentó las bases al emplear un modelo de regresión logística para predecir la probabilidad de ganar el campeonato utilizando 18 variables de rendimiento de 27 temporadas. Su trabajo nos demuestra la viabilidad de modelos simples pero robustos para este tipo de análisis.
*   **NBA-Game-Predictor (johntomlinsonn):** Este repositorio es un referente en complejidad y buenas prácticas. Construye un sistema completo que integra web scraping, *feature engineering* con más de 130 variables estadísticas y un modelo de ensamble (Random Forest + XGBoost) con una precisión superior al 80%. Su enfoque en la interpretabilidad y su estructura de código son una inspiración directa.


### Investigaciones Relevantes

*   **"Long-Sequence LSTM Modeling for NBA Game Outcome Prediction" (Rios et al., 2025):** Esta investigación subraya la importancia de usar series temporales largas (9,840 juegos) para capturar la evolución de la dinámica de los equipos. Valida nuestra decisión de usar datos históricos (2020-2025), demostrando que el contexto multi-temporada es vital para cualquier modelo predictivo serio.
*   **"Multiple Machine Learning Algorithms-based NBA Team Playoffs Prediction" (Abdi et al.):** Este estudio concluye que los modelos de ensamble como Random Forest son los más efectivos para predecir la clasificación a los playoffs. Su hallazgo respalda nuestra elección del modelo XGBoost (otro método de ensamble) y la importancia de la validación cruzada rigurosa.
*   **"The Skills That Win Championships" (UCF, 2026):** Este análisis se enfoca en identificar las habilidades específicas que correlacionan con ganar campeonatos, usando variables como la eficiencia de tiro y el *plus/minus*. Esto nos da el marco para seleccionar las características avanzadas adecuadas para nuestro modelo.
*   **"Expected Points Above Average" (Williams et al., 2025):** Este artículo propone nuevas métricas avanzadas para evaluar a jugadores, como *expected points*. Esto es relevante porque muestra el estado del arte en la creación de nuevas variables, un proceso que emularemos en nuestra fase de *feature engineering*.
*   **"Winner Prediction Model for Basketball Matches" (IEEE, 2024):** Esta investigación compara siete modelos de ML, encontrando que un *Multilayer Perceptron* logró la mayor precisión (85.3%). Aunque usamos XGBoost, su análisis comparativo de modelos justifica nuestra exploración y optimización de hiperparámetros.

---

## 📁 Estructura del Repositorio (Aquí colocas tu nuevo bloque de código con la estructura de carpetas)

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
│   │   └─── nba_api_connector.py
│   │   
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



---

## 📈 Hoja de Ruta

El desarrollo del proyecto se ha planificado en las siguientes fases clave:

- [ ] **Fase 1: Extracción de Datos:** Implementar scripts para descargar datos de NBA API (2020-2025) .
- [ ] **Fase 2: Feature Engineering:** Limpiar y crear variables en `build_features.py`.
- [ ] **Fase 3: Modelado:** Entrenar y optimizar el modelo XGBoost en `train_model.py`.
- [ ] **Fase 4: Dashboard:** Construir la aplicación modular en Streamlit con componentes.
- [ ] **Fase 5: Integración y Despliegue:** Subir a GitHub y, opcionalmente, desplegar en Streamlit Cloud.

## 👥 Autores

- **Kleyber Montoya** – Extracción de datos, Modelado predictivo (`src/data_collection`, `src/models`)
- **Alvaro Goncalves** – Interpretación de datos, Dashboard (`notebooks/`, `src/visualization`)

Ambos somos estudiantes de **Estadística y Ciencias Actuariales de la Universidad Central de Venezuela**.

---

🏆 **¡Que los datos nos permitan desentrañar la ciencia detrás del campeón!**