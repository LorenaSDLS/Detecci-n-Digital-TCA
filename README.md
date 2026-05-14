# Detección Digital TCA

Herramienta de procesamiento de lenguaje natural para la detección de indicios de Trastornos de la Conducta Alimentaria (TCA) — específicamente anorexia — en publicaciones de redes sociales en español.

## Estructura del repositorio

```
.
├── src/                    Código fuente de la herramienta
│   ├── data_loader.py      Ingesta y validación de datos
│   ├── preprocessor.py     Normalización de texto en español
│   ├── feature_extractor.py  Cuatro vistas: léxica, dominio, estilométrica, emocional
│   ├── feature_union.py    Fusión multi-vista + selección de atributos
│   ├── classifier.py       Comparador de LR / SVM / RF / XGBoost
│   ├── evaluator.py        Métricas, curva ROC, análisis de FN
│   ├── predictor.py        Serialización y generación del dictamen
│   └── main.py             Orquestador end-to-end
├── tests/                  Suite de 143 pruebas unitarias
├── data/                   Corpus de entrenamiento, prueba y lexicones
├── scripts/                Utilidades auxiliares
├── output/                 Artefactos generados (curva ROC, modelo, predicciones)
└── docs/                   Documentación complementaria
```

## Pipeline

El sistema implementa un pipeline `scikit-learn` de cinco etapas:

1. **Carga** (`DataLoader`) — lectura de CSV/XLSX, validación de esquema y mapeo de etiquetas.
2. **Preprocesamiento** (`Preprocessor`) — normalización, expansión de abreviaturas, eliminación de URLs/menciones/emojis, lematización con spaCy `es_core_news_sm`. Preserva pronombres de primera persona, negaciones y vocabulario corporal/alimenticio.
3. **Extracción multi-vista** (`MultiViewFeatureUnion`) — concatena cuatro representaciones:
   - **A. Léxica**: TF-IDF de n-gramas de palabras (1-3) reducido a 500 atributos vía `SelectKBest(chi2)`.
   - **B. Dominio**: conteos sobre lexicón pro-ana curado de Ramírez-Cifuentes et al. (2020) y Aguilera et al. (2021).
   - **C. Estilométrica**: pronombres, negaciones, métricas tipo LIWC sobre el texto crudo.
   - **D. Emocional**: bag-of-emotions sobre el NRC EmoLex en español (10 dimensiones).
4. **Clasificación** (`ClassifierComparator`) — compara Regresión Logística, Linear SVM calibrado, Random Forest y XGBoost mediante `GridSearchCV` con `StratifiedKFold(5)` optimizando AUC-ROC. El estimador ganador se promueve automáticamente.
5. **Evaluación y dictamen** (`evaluator`, `predictor`) — métricas, gráficos, reporte clínico de falsos negativos, serialización del pipeline completo (`.pkl`) y archivo de predicciones (`text_id, predicted_label, probability_yes`).

## Modelo seleccionado

Regresión Logística con regularización L2 (`solver=liblinear`, `C` optimizada por grid search). Ganó el comparador por margen marginal sobre Linear SVM (0.9558 vs 0.9546 en CV) y por ~2.5 puntos sobre los modelos basados en árboles. La elección se sostiene en tres argumentos:

1. **Empírico**: mejor AUC promedio en validación cruzada.
2. **Teórico**: los modelos lineales dominan en espacios dispersos de alta dimensionalidad como el TF-IDF.
3. **Operativo**: los coeficientes son directamente interpretables, habilitando el análisis clínico cualitativo de la Fase 3.

## Reproducibilidad

Requisitos: Python 3.13, [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m spacy download es_core_news_sm
uv run python scripts/download_nrc.py
uv run python -m src.main
```

El pipeline produce en `output/`:

- `modelo_anorexia_v1.pkl` — pipeline entrenado completo.
- `predicciones_finales.csv` — dictamen sobre el conjunto de prueba.
- `curva_roc.png`, `importancia_atributos.png` — diagnósticos visuales.
- `analisis_clinico_falsos_negativos.csv` — casos para revisión cualitativa.

## Pruebas

```bash
uv run pytest
```

143 pruebas unitarias sobre los seis módulos del pipeline. Cobertura: validación de esquemas, manejo de errores, normalización de texto, extracción de cada vista, fusión multi-vista, comparación de clasificadores, métricas y serialización.

## Optimizaciones aplicadas

| Técnica | Ubicación | Efecto |
|---|---|---|
| `SelectKBest(chi2, k=500)` | `feature_union.py` | Reduce Vista A de ~15 000 a 500 atributos |
| `nlp.pipe(batch_size=64)` | `preprocessor.py` | Lematización en lotes (≈10× más rápida) |
| Carga diferida de spaCy | `preprocessor.py` | Evita coste de import en tests |
| Matrices CSR dispersas | Vista A | Memoria proporcional a no-ceros |
| `n_jobs=-1` en GridSearch | `classifier.py` | Paraleliza folds sobre todos los cores |
| `tree_method="hist"` en XGBoost | `classifier.py` | Construcción de árboles ≈5× más rápida |
| Regex compilados a nivel de clase | `preprocessor.py` | Compilación única, no por llamada |
| Precomputo de stopwords efectivas | `preprocessor.py` | Diferencia de conjuntos en `__init__` |
| `CalibratedClassifierCV` sobre LinearSVC | `classifier.py` | Habilita `predict_proba` sin sacrificar velocidad |

## Equipo

| Componente | Autor |
|---|---|
| Ingesta y preprocesamiento (`data_loader.py`, `preprocessor.py`) | Andrea Blanco |
| Extracción multi-vista y clasificación (`feature_extractor.py`, `feature_union.py`, `classifier.py`) | Carlos Zamudio |
| Evaluación y predicción (`evaluator.py`, `predictor.py`) | Lorena Solís |

Autoría detallada por archivo en los docstrings de cada módulo.

## Referencias

- Aragón, M. E., López-Monroy, A. P., González, L. C., & Montes-y-Gómez, M. (2023). *Detecting mental disorders in social media through emotional patterns: The case of anorexia and depression*. IEEE Transactions on Affective Computing.
- Ramírez-Cifuentes, D., Freire, A., Baeza-Yates, R., et al. (2020). *Detection of suicidal ideation on social media: Multimodal, relational, and behavioral analysis*. JMIR.
- Aguilera, J., González, L. C., Montes-y-Gómez, M., & López-Monroy, A. P. (2021). *A new multimodal approach for the early detection of anorexia*. Information Processing & Management.
- Villa-Pérez, M. E., Trejo, L. A., Moin, M. B., & Stroulia, E. (2023). *Extracting mental health indicators from English and Spanish social media: A machine learning approach*. IEEE Access.
- Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a word-emotion association lexicon*. Computational Intelligence.
