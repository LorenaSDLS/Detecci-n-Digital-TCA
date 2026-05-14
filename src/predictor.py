"""
predictor.py — Módulo de predicción y serialización

Autor: Lorena Solís

El pipeline completo, que incluye:
preprocesador, extractor multi-vista, fusión y clasificador
se serializa con joblib como un único artefacto .pkl. 

El predictor carga el artefacto y expone una función predict(texts) que retorna tanto las 
etiquetas predichas como las probabilidades asociadas a la clase positiva. Este módulo es el 
encargado de procesar el archivo .xlsx de prueba proporcionado por el Protocolo de Evaluación 
y generar el archivo de salida con el dictamen por texto.

"""

import joblib
import pandas as pd

def save_pipeline(pipeline, filename='modelo_anorexia.pkl'):
    """
    Serializa el pipeline completo.
    """
    # Guarda el objeto completo (incluye preprocesamiento y modelo)
    joblib.dump(pipeline, filename)
    print(f"Pipeline serializado exitosamente en: {filename}")


def run_predictor(test_file_path, model_path):
    """
    Carga el modelo y genera el archivo de salida oficial.
    """
    # 1. Cargar el artefacto .pkl (Pipeline completo)
    pipeline = joblib.load(model_path)
    
    # 2. Cargar datos de prueba (.xlsx)
    df_test = pd.read_excel(test_file_path)
    
    X_input = df_test['title'].fillna('') + " " + df_test['text'].fillna('')
    
    # 3. Realizar predicciones usando el pipeline serializado
    predictions = pipeline.predict(X_input)
    probabilities = pipeline.predict_proba(X_input)[:, 1] 
    
    # 4. Formato del Protocolo de Evaluación
    output_df = pd.DataFrame({
        'text_id': df_test['text_id'],      # Debe coincidir con el original
        'predicted_label': predictions,     # 0 o 1
        'probability_yes': probabilities    # Probabilidad de clase 1
    })
    
    output_df.to_excel('predicciones_finales.xlsx', index=False)
    print("Dictamen generado exitosamente en 'predicciones_finales.xlsx'.")