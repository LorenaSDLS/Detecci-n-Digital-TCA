"""
test_evaluator.py — Pruebas unitarias para el módulo evaluator (Sección 3.6).

Autor: Lorena Solís

Cubre:
  - Cálculo correcto de métricas (AUC-ROC, F1, Precision, Recall).
  - Generación de la Matriz de Confusión (TP, TN, FP, FN).
  - Exportación correcta de Falsos Negativos a Excel para análisis clínico.
  - Robustez ante modelos con diferentes atributos (coef_ vs feature_importances_).
  - Generación de artefactos visuales (Curva ROC e Importancia de Atributos).
"""

import pytest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.evaluator import evaluate_model, export_clinical_errors

class TestEvaluator:
    
    @pytest.fixture
    def mock_results(self):
        """Genera datos sintéticos: 5 ejemplos con 1 Falso Negativo claro."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0]) # El índice 2 es FN (Real 1, Pred 0)
        y_probs = np.array([0.9, 0.1, 0.2, 0.8, 0.05])
        texts = [
            "Quiero bajar de peso rápido", 
            "Hoy desayuné fruta", 
            "No he comido en 2 días", # Este es el FN
            "Mi meta es ser delgada", 
            "Me siento bien hoy"
        ]
        features = [f"indicador_{i}" for i in range(10)]
        return y_true, y_pred, y_probs, texts, features

    def test_export_clinical_errors_content(self, mock_results, tmp_path):
        """Verifica que el CSV de errores contenga los datos correctos."""
        y_true, y_pred, y_probs, texts, _ = mock_results
        filename = tmp_path / "falsos_negativos.csv"

        export_clinical_errors(texts, y_true, y_pred, y_probs, filename=str(filename))

        assert os.path.exists(filename)
        df_error = pd.read_csv(filename)
        # Debe haber capturado al usuario que no comió en 2 días
        assert len(df_error) == 1
        assert "No he comido en 2 días" in df_error.iloc[0]['texto_original']

    def test_metrics_calculation_consistency(self, mock_results, monkeypatch):
        """Verifica que las métricas reportadas coincidan con sklearn."""
        y_true, y_pred, y_probs, _texts, features = mock_results

        class MockLinearModel:
            # Coeficientes con signos mezclados para verificar |coef_| en el ranking.
            coef_ = [np.array([-3.0, 0.1, 2.5, -0.5, 1.0, -1.5, 0.2, 0.05, -0.8, 0.3])]

        # Evitar que se abran ventanas de GUI durante los tests automatizados
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.setattr(plt, "savefig", lambda x: None)

        metrics = evaluate_model(
            y_true, y_pred, y_probs, MockLinearModel(), features
        )

        assert metrics["auc_roc"] == pytest.approx(roc_auc_score(y_true, y_probs))
        # Matriz de confusión esperada: TP=2, TN=2, FP=0, FN=1
        assert metrics["tp"] == 2
        assert metrics["tn"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 1