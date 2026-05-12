"""
test_classifier.py — Pruebas unitarias para classifier (Sección 3.5).

Autor: Carlos Zamudio

Cubre:
  * GridSearchCV ejecuta los cuatro modelos (logreg, svm, rf, xgb).
  * Ranking ordenado por AUC-ROC descendente.
  * best_estimator_ predice probabilidades calibradas.
  * predict() y predict_proba() devuelven shapes correctos.

Las mallas de hiperparámetros se reducen a un solo valor por dimensión vía
``param_grids=...`` para mantener los tests rápidos (<60s en total).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.classifier import ClassifierComparator, ModelResult


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

@pytest.fixture
def small_binary_dataset():
    """Dataset binario sintético pequeño y bien separable."""
    X, y = make_classification(
        n_samples=80,
        n_features=15,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=42,
    )
    return X, y


@pytest.fixture
def fast_param_grids():
    """Mallas mínimas de un solo punto para acelerar tests."""
    return {
        "logreg": {"C": [1.0]},
        "svm": {"estimator__C": [1.0]},
        "rf": {"n_estimators": [50], "max_depth": [5]},
        "xgb": {"n_estimators": [50], "max_depth": [3], "learning_rate": [0.1]},
    }


@pytest.fixture
def fitted_comparator(small_binary_dataset, fast_param_grids):
    """Comparator ajustado, compartido entre tests para evitar repetir cómputo."""
    X, y = small_binary_dataset
    comp = ClassifierComparator(
        cv_folds=3,
        param_grids=fast_param_grids,
        n_jobs=1,
    )
    return comp.fit(X, y), X, y


# ---------------------------------------------------------------------------
# Clase 1: Ajuste y ranking
# ---------------------------------------------------------------------------

class TestFitAndRanking:

    def test_fit_returns_self(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        assert isinstance(comp, ClassifierComparator)

    def test_results_has_four_entries(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        assert len(comp.results_) == 4

    def test_results_entries_are_modelresult(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        for r in comp.results_:
            assert isinstance(r, ModelResult)

    def test_results_contain_all_four_models(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        names = {r.name for r in comp.results_}
        assert names == {"logreg", "svm", "rf", "xgb"}

    def test_ranking_sorted_desc(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        scores = [score for _, score in comp.ranking()]
        assert scores == sorted(scores, reverse=True)

    def test_best_score_matches_top_of_ranking(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        assert comp.best_score_ == comp.results_[0].best_score

    def test_best_name_matches_top_of_ranking(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        assert comp.best_name_ == comp.results_[0].name

    def test_auc_scores_in_valid_range(self, fitted_comparator):
        comp, _, _ = fitted_comparator
        for r in comp.results_:
            assert 0.0 <= r.best_score <= 1.0


# ---------------------------------------------------------------------------
# Clase 2: Predicciones del mejor modelo
# ---------------------------------------------------------------------------

class TestPredictions:

    def test_predict_returns_binary_labels(self, fitted_comparator):
        comp, X, _ = fitted_comparator
        preds = comp.predict(X)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_shape_matches_input(self, fitted_comparator):
        comp, X, _ = fitted_comparator
        preds = comp.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_proba_shape(self, fitted_comparator):
        comp, X, _ = fitted_comparator
        proba = comp.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_rows_sum_to_one(self, fitted_comparator):
        comp, X, _ = fitted_comparator
        proba = comp.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_unit_interval(self, fitted_comparator):
        comp, X, _ = fitted_comparator
        proba = comp.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# Clase 3: Estado antes de fit
# ---------------------------------------------------------------------------

class TestNotFitted:

    def test_predict_before_fit_raises(self, small_binary_dataset):
        X, _ = small_binary_dataset
        from sklearn.exceptions import NotFittedError
        comp = ClassifierComparator()
        with pytest.raises(NotFittedError):
            comp.predict(X)

    def test_ranking_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError
        comp = ClassifierComparator()
        with pytest.raises(NotFittedError):
            comp.ranking()
