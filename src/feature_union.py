"""
feature_union.py — Módulo de fusión y reducción multi-vista.

Autor: Carlos Zamudio

Concatena horizontalmente las cuatro vistas del módulo ``feature_extractor``
utilizando ``sklearn.pipeline.FeatureUnion``. Aplica ``StandardScaler`` sobre
las vistas densas (B, C y D) para homogeneizar sus escalas, mientras que la
Vista A se mantiene en formato disperso.

Opcionalmente aplica selección de atributos con ``SelectKBest(chi2)`` sobre la
porción TF-IDF para retener únicamente las K características más discriminantes
— palanca principal de optimización reportada en la Parte B del Protocolo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extractor import (
    DomainLexiconVectorizer,
    EmotionLexiconVectorizer,
    LexicalTfidfVectorizer,
    StylometricVectorizer,
)


class MultiViewFeatureUnion(BaseEstimator, TransformerMixin):
    """Fusión multi-vista A+B+C+D con escalado y selección opcional de atributos.

    Construye internamente un ``FeatureUnion`` con cuatro ramas:

        * Rama A (léxica): ``LexicalTfidfVectorizer`` → ``SelectKBest(chi2)``
          opcional. Permanece dispersa.
        * Rama B (dominio): ``DomainLexiconVectorizer`` → ``StandardScaler``.
        * Rama C (estilística): ``StylometricVectorizer`` → ``StandardScaler``.
        * Rama D (emocional): ``EmotionLexiconVectorizer`` → ``StandardScaler``.

    La salida es una matriz dispersa (CSR) por la presencia de la vista A. Las
    vistas densas se mantienen como bloques densos dentro de la concatenación.

    Attributes:
        use_char_ngrams: Activa char n-gramas en la Vista A.
        select_k: Si es un entero > 0, aplica ``SelectKBest(chi2, k=select_k)``
            sobre la Vista A. Si es ``None`` (defecto), no aplica selección.
        nrc_lexicon_path: Ruta al archivo del NRC EmoLex (Vista D).
    """

    def __init__(
        self,
        use_char_ngrams: bool = False,
        select_k: int | None = None,
        nrc_lexicon_path: str | Path | None = None,
    ) -> None:
        self.use_char_ngrams = use_char_ngrams
        self.select_k = select_k
        self.nrc_lexicon_path = nrc_lexicon_path

    def fit(self, X, y=None) -> MultiViewFeatureUnion:
        """Ajusta todas las ramas del ``FeatureUnion``.

        Args:
            X: Lista de textos (mismos pre-procesados por el caller).
            y: Etiquetas binarias; requeridas si ``select_k`` no es ``None``
                porque ``chi2`` necesita supervisión.
        """
        if self.select_k is not None and y is None:
            raise ValueError(
                "select_k requiere etiquetas 'y' (chi2 es un test supervisado)."
            )
        self._union_ = self._build_union()
        self._union_.fit(X, y)
        return self

    def transform(self, X):
        """Aplica las cuatro vistas y concatena horizontalmente."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_union_")
        return self._union_.transform(X)

    def fit_transform(self, X, y=None):
        """Equivalente a ``fit(X, y).transform(X)`` pero con caché interna."""
        if self.select_k is not None and y is None:
            raise ValueError(
                "select_k requiere etiquetas 'y' (chi2 es un test supervisado)."
            )
        self._union_ = self._build_union()
        return self._union_.fit_transform(X, y)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de columnas tras la fusión y selección."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_union_")
        return self._union_.get_feature_names_out()

    # ------------------------------------------------------------------
    # Construcción del pipeline
    # ------------------------------------------------------------------

    def _build_union(self) -> FeatureUnion:
        """Compone el ``FeatureUnion`` con las cuatro ramas configuradas."""
        # Rama A: TF-IDF + (opcional) SelectKBest(chi2)
        lexical_steps: list[tuple[str, object]] = [
            ("tfidf", LexicalTfidfVectorizer(use_char_ngrams=self.use_char_ngrams)),
        ]
        if self.select_k is not None:
            lexical_steps.append(("select", SelectKBest(chi2, k=self.select_k)))
        rama_a = Pipeline(lexical_steps)

        # Ramas B, C, D: vista densa + StandardScaler
        rama_b = Pipeline([
            ("domain", DomainLexiconVectorizer()),
            ("scale", StandardScaler()),
        ])
        rama_c = Pipeline([
            ("style", StylometricVectorizer()),
            ("scale", StandardScaler()),
        ])
        rama_d = Pipeline([
            ("emotion", EmotionLexiconVectorizer(lexicon_path=self.nrc_lexicon_path)),
            ("scale", StandardScaler()),
        ])

        return FeatureUnion([
            ("lexical", rama_a),
            ("domain", rama_b),
            ("stylometric", rama_c),
            ("emotion", rama_d),
        ])
