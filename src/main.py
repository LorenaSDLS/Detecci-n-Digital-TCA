import pandas as pd
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from feature_union import MultiViewFeatureUnion
from classifier import AnorexiaClassifier
from evaluator import evaluate_model, export_clinical_errors
from predictor import save_pipeline, run_predictor
from sklearn.pipeline import Pipeline

def main():
    # 1. CARGA DE DATOS 
    print("--- Fase 1: Ingesta de Datos ---")
    loader = DataLoader()
    # Cargamos el dataset principal
    df = loader.load_data("datos_entrenamiento.xlsx")
    # Partición estratificada para validar
    train_df, val_df = loader.get_split(df)
    
    X_train, y_train = train_df['text'], train_df['label']
    X_val, y_val = val_df['text'], val_df['label']

    # 2. CONSTRUCCIÓN DEL PIPELINE (Unión de todos los módulos)
    print("\n--- Fase 2: Configuración del Pipeline ---")
    
    # Usamos el perfil 'full' para el pipeline principal
    preprocessor = TextPreprocessor(profile='full')
    
    # Fusión de las 4 vistas
    union = MultiViewFeatureUnion(
        nrc_lexicon_path="lexicons/NRC-Emotion-Lexicon.txt",
        select_k=500 # Optimización de dimensionalidad
    )

    # El pipeline empaqueta: Limpieza -> Extracción -> Modelo
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('features', union),
        ('clf', AnorexiaClassifier(model_type='xgb')) # Ejemplo con XGBoost
    ])

    # 3. ENTRENAMIENTO 
    print("\n--- Fase 3: Entrenamiento del Modelo ---")
    full_pipeline.fit(X_train, y_train)

    # 4. EVALUACIÓN 
    print("\n--- Fase 4: Evaluación Clínica ---")
    y_pred = full_pipeline.predict(X_val)
    y_probs = full_pipeline.predict_proba(X_val)[:, 1]
    
    # Obtenemos nombres de atributos para el gráfico de importancia
    feature_names = full_pipeline.named_steps['features'].get_feature_names_out()
    
    # Llamada a función de evaluación
    evaluate_model(
        y_true=y_val, 
        y_pred=y_pred, 
        y_probs=y_probs, 
        model=full_pipeline.named_steps['clf'].best_estimator_, # El mejor modelo hallado
        feature_names=feature_names,
        X_val_text=X_val
    )
    
    # Exportación para la Fase 3 (Análisis crítico)
    export_clinical_errors(X_val, y_val, y_pred, y_probs)

    # 5. SERIALIZACIÓN Y PREDICCIÓN FINAL 
    print("\n--- Fase 5: Serialización y Dictamen Final ---")
    
    # Guardamos el "artefacto" .pkl
    model_name = "modelo_anorexia_v1.pkl"
    save_pipeline(full_pipeline, model_name)
    
    # Ejecutamos el predictor sobre el archivo de prueba oficial del protocolo
    run_predictor(
        test_file_path="archivo_prueba_protocolo.xlsx", 
        model_path=model_name
    )

    print("\n¡Proceso completado! Revisa los archivos .xlsx y .png generados.")

if __name__ == "__main__":
    main()
