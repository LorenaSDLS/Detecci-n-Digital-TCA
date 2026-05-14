"""
main.py — Orquestador del pipeline completo.

Une los módulos del proyecto en el flujo end-to-end descrito por el Protocolo
de Evaluación: carga, preprocesado, extracción multi-vista, comparación de
clasificadores, evaluación clínica, serialización y predicción final.

Todos los artefactos (gráficos, reportes, modelo serializado y predicciones)
se escriben en el directorio ``output/``.
"""

from pathlib import Path

from sklearn.pipeline import Pipeline

from src.classifier import ClassifierComparator
from src.data_loader import DataLoader
from src.evaluator import evaluate_model, export_clinical_errors
from src.feature_union import MultiViewFeatureUnion
from src.predictor import run_predictor, save_pipeline
from src.preprocessor import Preprocessor

OUTPUT_DIR = Path("output")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CARGA DE DATOS
    print("--- Fase 1: Ingesta de Datos ---")
    loader = DataLoader(filepath="data/data_train.csv")
    df = loader.load()
    train_df, val_df = loader.train_val_split(df)

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]

    # 2. CONSTRUCCIÓN DEL PIPELINE (Unión de todos los módulos)
    print("\n--- Fase 2: Configuración del Pipeline ---")
    preprocessor = Preprocessor(profile="full")
    union = MultiViewFeatureUnion(
        nrc_lexicon_path="data/lexicons/NRC-Emotion-Lexicon-v0.92-Spanish.txt",
        select_k=500,
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("features", union),
        ("clf", ClassifierComparator()),
    ])

    # 3. ENTRENAMIENTO
    print("\n--- Fase 3: Entrenamiento del Modelo ---")
    full_pipeline.fit(X_train, y_train)

    clf = full_pipeline.named_steps["clf"]
    print(f"\nGanador: {clf.best_name_} (CV AUC={clf.best_score_:.4f})")
    print("Ranking de modelos (AUC-ROC promedio 5-fold CV):")
    for name, score in clf.ranking():
        print(f"  {name:<8} {score:.4f}")

    # 4. EVALUACIÓN
    print("\n--- Fase 4: Evaluación Clínica ---")
    y_pred = full_pipeline.predict(X_val)
    y_probs = full_pipeline.predict_proba(X_val)[:, 1]

    feature_names = full_pipeline.named_steps["features"].get_feature_names_out()

    evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        y_probs=y_probs,
        model=full_pipeline.named_steps["clf"].best_estimator_,
        feature_names=feature_names,
        roc_path=OUTPUT_DIR / "curva_roc.png",
        importance_path=OUTPUT_DIR / "importancia_atributos.png",
    )

    export_clinical_errors(
        X_val,
        y_val,
        y_pred,
        y_probs,
        filename=OUTPUT_DIR / "analisis_clinico_falsos_negativos.csv",
    )

    # 5. SERIALIZACIÓN Y PREDICCIÓN FINAL
    print("\n--- Fase 5: Serialización y Dictamen Final ---")
    model_path = OUTPUT_DIR / "modelo_anorexia_v1.pkl"
    save_pipeline(full_pipeline, model_path)

    run_predictor(
        test_file_path="data/data_test_fold1.csv",
        model_path=model_path,
        output_path=OUTPUT_DIR / "predicciones_finales.csv",
    )

    print(f"\n¡Proceso completado! Artefactos en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
