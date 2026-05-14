"""
evaluator.py — Módulo de evaluación

Autor: Lorena Solís

Calcula las métricas exigidas por el Protocolo de Evaluación, siendo la métrica primaria el AUC-ROC.
Como métricas secundarias reporta F1, precisión, recall y la matriz de confusión completa con TP, TN, FP y FN.
Incluye además un análisis cualitativo de los falsos negativos, que en el dominio clínico de detección de anorexia
son considerablemente más costosos que los falsos positivos, ya que representan usuarios enfermos no identificados.

El módulo genera también la curva ROC y un gráfico de importancia de atributos
(coeficientes para modelos lineales, feature_importances_ para modelos basados en árboles).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    y_true,
    y_pred,
    y_probs,
    model,
    feature_names,
    roc_path: str | Path = "curva_roc.png",
    importance_path: str | Path = "importancia_atributos.png",
):
    """
    Calcula métricas, genera la matriz de confusión y visualiza importancia de atributos.

    Devuelve un dict con las métricas principales para facilitar tests y agregación.
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
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(str(roc_path))
    plt.close()

    # 4. Importancia de Atributos
    # En modelos lineales los atributos más discriminantes son los de mayor
    # |coef_|, no los más positivos (los muy negativos discriminan la clase 0).
    if hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        importances = model.feature_importances_

    serie = pd.Series(importances, index=feature_names)
    top_idx = serie.abs().nlargest(20).index
    feat_imp = serie.loc[top_idx].sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Top 20 Atributos más Discriminantes")
    plt.tight_layout()
    plt.savefig(str(importance_path))
    plt.close()

    return {
        "auc_roc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def export_clinical_errors(
    X_val_text,
    y_true,
    y_pred,
    y_probs,
    filename: str | Path = "analisis_clinico_falsos_negativos.csv",
):
    """
    Identifica y exporta los Falsos Negativos para la reflexión crítica de la Fase 3.

    Soporta salida ``.csv`` (defecto) o ``.xlsx`` según la extensión de ``filename``.
    """
    df_eval = pd.DataFrame({
        "texto_original": list(X_val_text),
        "etiqueta_real": y_true,
        "prediccion": y_pred,
        "probabilidad_anorexia": y_probs,
    })

    # Filtramos los Falsos Negativos (Real=1, Predicho=0)
    falsos_negativos = df_eval[
        (df_eval["etiqueta_real"] == 1) & (df_eval["prediccion"] == 0)
    ]

    out_path = Path(filename)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        falsos_negativos.to_csv(out_path, index=False)
    elif suffix in (".xlsx", ".xls"):
        falsos_negativos.to_excel(out_path, index=False)
    else:
        raise ValueError(
            f"Extensión no soportada para reporte clínico: {suffix!r}. "
            "Use .csv o .xlsx."
        )
    print(
        f"--> Reporte clínico generado: {filename} "
        f"({len(falsos_negativos)} casos identificados)."
    )
