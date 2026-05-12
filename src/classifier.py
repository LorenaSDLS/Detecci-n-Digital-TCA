"""
classifier.py — Módulo clasificador (Sección 3.5).

Autor: Carlos Zamudio

Entrena y compara cuatro algoritmos de aprendizaje supervisado sobre la matriz
fusionada producida por ``feature_union``:

  * Regresión Logística con regularización L2 (Aragón et al. 2023, F1=0.84).
  * Linear SVM envuelto en ``CalibratedClassifierCV`` para producir
    probabilidades calibradas (Aguilera et al. 2021).
  * Random Forest (Villa-Pérez et al. 2023, Benítez-Andrades et al. 2022).
  * XGBoost (Villa-Pérez et al. 2023, clasificador clásico más competitivo).

La selección de hiperparámetros se hace con ``GridSearchCV`` con validación
cruzada estratificada de 5 folds, optimizando AUC-ROC como función objetivo.
El modelo con mejor AUC de validación se promueve a ``best_estimator_``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    """Resumen de la búsqueda en cuadrícula de un único modelo.

    Attributes:
        name: Etiqueta del modelo (logreg, svm, rf, xgb).
        best_score: Mejor AUC-ROC promedio sobre los k folds.
        best_params: Hiperparámetros que produjeron ``best_score``.
        best_estimator: Estimador ajustado con ``best_params``.
    """

    name: str
    best_score: float
    best_params: dict[str, Any]
    best_estimator: Any


class ClassifierComparator(BaseEstimator):
    """Compara LR, SVM, RF y XGBoost mediante GridSearchCV sobre AUC-ROC.

    El método :meth:`fit` ejecuta los cuatro ``GridSearchCV`` y almacena el
    ranking ordenado en :attr:`results_`. El estimador ganador queda en
    :attr:`best_estimator_` y su nombre en :attr:`best_name_`.

    Attributes:
        cv_folds: Número de folds del ``StratifiedKFold`` (defecto 5).
        random_state: Semilla aleatoria para reproducibilidad.
        n_jobs: Trabajos paralelos por ``GridSearchCV``.
        verbose: Nivel de verbosidad del grid search.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
        param_grids: dict[str, dict[str, list]] | None = None,
    ) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.param_grids = param_grids

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, X, y) -> ClassifierComparator:
        """Ejecuta GridSearchCV para los cuatro modelos sobre ``(X, y)``."""
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        candidates = self._build_candidates()

        results: list[ModelResult] = []
        for name, (estimator, grid) in candidates.items():
            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
            )
            search.fit(X, y)
            results.append(ModelResult(
                name=name,
                best_score=float(search.best_score_),
                best_params=dict(search.best_params_),
                best_estimator=search.best_estimator_,
            ))

        # Ranking descendente por AUC-ROC promedio de CV
        results.sort(key=lambda r: r.best_score, reverse=True)
        self.results_ = results
        self.best_name_ = results[0].name
        self.best_estimator_ = results[0].best_estimator
        self.best_score_ = results[0].best_score
        return self

    def predict(self, X) -> np.ndarray:
        """Predice etiquetas con el mejor estimador."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predice probabilidades con el mejor estimador (todos calibrados)."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def ranking(self) -> list[tuple[str, float]]:
        """Devuelve ``[(nombre, auc), ...]`` ordenado de mayor a menor AUC."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "results_")
        return [(r.name, r.best_score) for r in self.results_]

    # ------------------------------------------------------------------
    # Construcción de candidatos
    # ------------------------------------------------------------------

    def _build_candidates(self) -> dict[str, tuple[Any, dict[str, list]]]:
        """Define los cuatro modelos y sus mallas de hiperparámetros."""
        seed = self.random_state

        # penalty="l2" es el default en sklearn; pasarlo explícito está deprecated en 1.10.
        logreg = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
        logreg_grid = {"C": [0.1, 1.0, 10.0]}

        # LinearSVC no produce probabilidades nativas → CalibratedClassifierCV
        # con cv=3 internamente para calibrar y mantener compatibilidad con AUC.
        svm = CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=2000, random_state=seed, dual="auto"),
            cv=3,
        )
        svm_grid = {"estimator__C": [0.1, 1.0, 10.0]}

        rf = RandomForestClassifier(
            n_jobs=1,  # n_jobs externo en GridSearchCV; evitar oversubscription
            random_state=seed,
        )
        rf_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10],
        }

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        xgb_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }

        defaults = {
            "logreg": (logreg, logreg_grid),
            "svm": (svm, svm_grid),
            "rf": (rf, rf_grid),
            "xgb": (xgb, xgb_grid),
        }
        # Override de grids para casos como tests rápidos o búsqueda extendida.
        if self.param_grids is not None:
            return {
                name: (estimator, self.param_grids.get(name, grid))
                for name, (estimator, grid) in defaults.items()
            }
        return defaults
