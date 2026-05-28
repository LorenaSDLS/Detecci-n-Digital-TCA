"""test_compare.py — Pruebas de la lógica de comparación de Fase 3.

Verifica el cálculo de AUC y del análisis FP/FN sobre datos sintéticos, y la
omisión limpia de métodos sin predicciones. No requiere torch.

Autor: Carlos Zamudio (Fase 3)
"""

import pandas as pd

from src.models import comparison as compare


class TestEvaluatePredictions:
    def test_perfect_separation_auc_one(self):
        gold = pd.DataFrame({"text_id": ["a", "b", "c", "d"], "label": [1, 1, 0, 0]})
        preds = pd.DataFrame({
            "text_id": ["a", "b", "c", "d"],
            "predicted_label": [1, 1, 0, 0],
            "probability_yes": [0.9, 0.8, 0.2, 0.1],
        })
        res = compare.evaluate_predictions(gold, preds)
        assert res.auc == 1.0
        assert res.f1 == 1.0
        assert (res.tp, res.tn, res.fp, res.fn) == (2, 2, 0, 0)
        assert res.fp_ids == [] and res.fn_ids == []

    def test_reports_fp_and_fn_ids(self):
        gold = pd.DataFrame({"text_id": ["a", "b", "c", "d"], "label": [1, 1, 0, 0]})
        preds = pd.DataFrame({
            "text_id": ["a", "b", "c", "d"],
            "predicted_label": [0, 1, 1, 0],   # a=FN, c=FP
            "probability_yes": [0.4, 0.8, 0.7, 0.2],
        })
        res = compare.evaluate_predictions(gold, preds)
        assert res.fn_ids == ["a"]
        assert res.fp_ids == ["c"]
        assert (res.fp, res.fn) == (1, 1)
        assert res.f1 == 0.5  # P=R=0.5 → F1=0.5 (TP=1, FP=1, FN=1)

    def test_join_on_text_id_subset(self):
        # La verdad-terreno tiene más muestras que las predicciones disponibles.
        gold = pd.DataFrame({"text_id": ["a", "b", "c"], "label": [1, 0, 1]})
        preds = pd.DataFrame({
            "text_id": ["a", "b"],
            "predicted_label": [1, 0],
            "probability_yes": [0.9, 0.1],
        })
        res = compare.evaluate_predictions(gold, preds)
        assert res.n == 2  # sólo evalúa la intersección

    def test_no_common_ids_returns_none(self):
        gold = pd.DataFrame({"text_id": ["x"], "label": [1]})
        preds = pd.DataFrame({"text_id": ["y"], "predicted_label": [1], "probability_yes": [0.9]})
        assert compare.evaluate_predictions(gold, preds) is None


class TestCollectEvaluations:
    def test_skips_missing_prediction_files(self, tmp_path):
        gold = pd.DataFrame({"text_id": ["a", "b"], "label": [1, 0]})
        # Sólo el baseline tiene archivo de predicciones.
        pd.DataFrame({
            "text_id": ["a", "b"],
            "predicted_label": [1, 0],
            "probability_yes": [0.9, 0.1],
        }).to_csv(tmp_path / "predicciones_finales.csv", index=False)

        results = compare.collect_evaluations(gold, tmp_path)
        # Las tres variantes de Fase 3 se omiten (sin archivo); queda el baseline.
        assert [r.name for r in results] == ["baseline_2b"]
        assert results[0].auc == 1.0

    def test_only_filter(self, tmp_path):
        gold = pd.DataFrame({"text_id": ["a", "b"], "label": [1, 0]})
        pd.DataFrame({
            "text_id": ["a", "b"],
            "predicted_label": [1, 0],
            "probability_yes": [0.9, 0.1],
        }).to_csv(tmp_path / "predicciones_finales.csv", index=False)

        results = compare.collect_evaluations(gold, tmp_path, only={"robertuito_finetune"})
        assert results == []  # baseline excluido por el filtro --only
