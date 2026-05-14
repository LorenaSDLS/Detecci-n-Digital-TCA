"""
evaulator.py — Módulo de evaluación

Autor: Lorena Solís

Calcula las métricas exigidas por el Protocolo de Evaluación, siendo la métrica primaria el AUC-ROC. 
Como métricas secundarias reporta F1, precisión, recall y la matriz de confusión completa con TP, TN, FP y FN. 
Incluye además un análisis cualitativo de los falsos negativos, que en el dominio clínico de detección de anorexia 
son considerablemente más costosos que los falsos positivos, ya que representan usuarios enfermos no identificados. 

El módulo genera también la curva ROC y un gráfico de importancia de atributos 
(coeficientes para modelos lineales, feature_importances_ para modelos basados en árboles).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import pandas as pd

def evaluate_model(y_true, y_pred, y_probs, model, feature_names, X_val_text):
    """
    Calcula métricas, genera la matriz de confusión y visualiza importancia de atributos.
    """
    # 1. Métricas Principales
    auc = roc_auc_score(y_true, y_probs)
    print(f"\nMétrica Primaria - AUC-ROC: {auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred))

    # 2. Matriz de Confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Resultados: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 3. Gráfico de Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("curva_roc.png")
    plt.close()

    # 4. Importancia de Atributos
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        importances = model.feature_importances_
    
    feat_imp = pd.Series(importances, index=feature_names).nlargest(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title('Top 20 Atributos más Discriminantes')
    plt.tight_layout()
    plt.savefig("importancia_atributos.png")
    plt.close()

def export_clinical_errors(X_val_text, y_true, y_pred, y_probs, filename="analisis_clinico_falsos_negativos.xlsx"):
    """
    Identifica y exporta los Falsos Negativos para la reflexión crítica de la Fase 3.
    """
    df_eval = pd.DataFrame({
        'texto_original': X_val_text,
        'etiqueta_real': y_true,
        'prediccion': y_pred,
        'probabilidad_anorexia': y_probs
    })
    
    # Filtramos los Falsos Negativos (Real=1, Predicho=0)
    falsos_negativos = df_eval[(df_eval['etiqueta_real'] == 1) & (df_eval['prediccion'] == 0)]
    
    falsos_negativos.to_excel(filename, index=False)
    print(f"--> Reporte clínico generado: {filename} ({len(falsos_negativos)} casos identificados).")