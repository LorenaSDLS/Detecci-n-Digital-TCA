"""
test_feature_union.py — Pruebas unitarias para feature_union (Sección 3.4).

Autor: Carlos Zamudio

Cubre:
  * Composición de las cuatro vistas en un FeatureUnion.
  * StandardScaler aplicado a vistas densas (B, C, D).
  * SelectKBest(chi2) opcional sobre la Vista A.
  * Compatibilidad con la API de scikit-learn (clone, fit_transform, etc.).
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

# spaCy es requerido por la Vista C: si falta el modelo, se hace skip.
pytest.importorskip("spacy", reason="spaCy no instalado")

from src.feature_union import MultiViewFeatureUnion  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture local: dataset balanceado pre-procesado y etiquetas
# ---------------------------------------------------------------------------

@pytest.fixture
def labeled_texts(sample_texts_minimal):
    """Texts + labels para tests que requieren supervisión (chi2)."""
    n = len(sample_texts_minimal)
    # Etiquetas alternadas balanceadas
    y = np.array([i % 2 for i in range(n)])
    return sample_texts_minimal, y


# ---------------------------------------------------------------------------
# Clase 1: Composición básica
# ---------------------------------------------------------------------------

class TestMultiViewFeatureUnion:

    @pytest.fixture(autouse=True)
    def setup(self, nrc_lexicon_path):
        try:
            self.union = MultiViewFeatureUnion(nrc_lexicon_path=nrc_lexicon_path)
        except RuntimeError as e:
            pytest.skip(str(e))

    def test_output_is_sparse(self, sample_texts_minimal):
        X = self.union.fit_transform(sample_texts_minimal)
        # La presencia de la vista A (sparse) hace que la salida sea sparse.
        assert sparse.issparse(X)

    def test_output_shape_rows_match_input(self, sample_texts_minimal):
        X = self.union.fit_transform(sample_texts_minimal)
        assert X.shape[0] == len(sample_texts_minimal)

    def test_output_includes_all_four_views(self, sample_texts_minimal):
        X = self.union.fit_transform(sample_texts_minimal)
        # B+C+D aportan al menos 30+12+10 = 52 columnas densas;
        # A aporta al menos algunas columnas tras min_df=2.
        assert X.shape[1] >= 52

    def test_fit_returns_self(self, sample_texts_minimal):
        assert self.union.fit(sample_texts_minimal) is self.union

    def test_no_nan_in_output(self, sample_texts_minimal):
        X = self.union.fit_transform(sample_texts_minimal).toarray()
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# Clase 2: SelectKBest sobre la Vista A
# ---------------------------------------------------------------------------

class TestSelectKBest:

    def test_select_k_requires_labels(self, sample_texts_minimal, nrc_lexicon_path):
        try:
            union = MultiViewFeatureUnion(
                select_k=5, nrc_lexicon_path=nrc_lexicon_path,
            )
        except RuntimeError as e:
            pytest.skip(str(e))
        with pytest.raises(ValueError, match="select_k"):
            union.fit(sample_texts_minimal, y=None)

    def test_select_k_reduces_lexical_dimension(self, labeled_texts, nrc_lexicon_path):
        texts, y = labeled_texts
        try:
            union_no_select = MultiViewFeatureUnion(nrc_lexicon_path=nrc_lexicon_path)
            X_full = union_no_select.fit_transform(texts).shape[1]

            union_select = MultiViewFeatureUnion(
                select_k=3, nrc_lexicon_path=nrc_lexicon_path,
            )
            X_reduced = union_select.fit_transform(texts, y).shape[1]
        except RuntimeError as e:
            pytest.skip(str(e))

        # Reducir K en la rama A debe reducir las columnas totales o igualarlas.
        assert X_reduced <= X_full


# ---------------------------------------------------------------------------
# Clase 3: Compatibilidad sklearn
# ---------------------------------------------------------------------------

class TestSklearnCompatibility:

    def test_can_be_cloned(self, nrc_lexicon_path):
        union = MultiViewFeatureUnion(
            use_char_ngrams=True,
            select_k=10,
            nrc_lexicon_path=nrc_lexicon_path,
        )
        cloned = clone(union)
        assert cloned.use_char_ngrams is True
        assert cloned.select_k == 10
        assert cloned.nrc_lexicon_path == nrc_lexicon_path

    def test_transform_before_fit_raises(self, sample_texts_minimal, nrc_lexicon_path):
        try:
            union = MultiViewFeatureUnion(nrc_lexicon_path=nrc_lexicon_path)
        except RuntimeError as e:
            pytest.skip(str(e))
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            union.transform(sample_texts_minimal)
