"""baseline_classic.py — Baseline clásico de Fase 2B como ``AnorexiaClassifier``.

Envuelve el pipeline multivista de Fase 2B (``Preprocessor`` +
``MultiViewFeatureUnion`` + ``ClassifierComparator``) detrás del contrato común
``fit``/``predict_proba``, para que el baseline sea entrenable y cacheable por
variante igual que las variantes de Fase 3.

Motivación: el archivo congelado ``predicciones_finales.csv`` sólo cubre la
variante CON hashtags. Para evaluar el baseline también SIN hashtags hay que
reentrenarlo sobre los datos limpios; este wrapper lo permite de forma uniforme
(y reproduce, sobre los datos originales, el baseline de Fase 2B).

Los imports pesados (sklearn de feature engineering, módulos de ``src``) se
difieren a ``fit`` para que importar el registro no los requiera.
"""

from __future__ import annotations

import numpy as np

from src.models.base import AnorexiaClassifier

_NRC_LEXICON = "data/lexicons/NRC-Emotion-Lexicon-v0.92-Spanish.txt"


class ClassicBaseline(AnorexiaClassifier):
    """Pipeline clásico multivista de Fase 2B (TF-IDF/BoW + clasificador).

    Attributes:
        nrc_lexicon_path: Ruta al léxico NRC EmoLex (Vista D del feature union).
        select_k: Nº de atributos retenidos por la selección del feature union.
        seed: Semilla del ``ClassifierComparator`` (GridSearchCV reproducible).
    """

    name = "baseline_clasico"

    def __init__(
        self,
        nrc_lexicon_path: str = _NRC_LEXICON,
        select_k: int = 500,
        seed: int = 42,
    ) -> None:
        self.nrc_lexicon_path = nrc_lexicon_path
        self.select_k = select_k
        self.seed = seed

        self._pipeline = None

    def fit(self, texts: list[str], labels: list[int]) -> "ClassicBaseline":
        """Construye y ajusta el pipeline de Fase 2B sobre los textos crudos."""
        import pandas as pd
        from sklearn.pipeline import Pipeline

        from src.classifier import ClassifierComparator
        from src.feature_union import MultiViewFeatureUnion
        from src.preprocessor import Preprocessor

        self._pipeline = Pipeline([
            ("preprocessor", Preprocessor(profile="full")),
            ("features", MultiViewFeatureUnion(
                nrc_lexicon_path=self.nrc_lexicon_path,
                select_k=self.select_k,
            )),
            ("clf", ClassifierComparator(random_state=self.seed)),
        ])

        self._pipeline.fit(pd.Series(list(texts)), np.asarray(labels, dtype=int))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) del mejor clasificador clásico calibrado."""
        if self._pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado: llama a fit() primero.")

        import pandas as pd

        return self._pipeline.predict_proba(pd.Series(list(texts)))[:, 1]
