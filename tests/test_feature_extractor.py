"""
test_feature_extractor.py — Pruebas unitarias para el módulo feature_extractor.

Autor: Carlos Zamudio

Cubre las cuatro vistas multi-vista de la Sección 3.3:

  * LexicalTfidfVectorizer  (Vista A)
  * DomainLexiconVectorizer (Vista B)
  * StylometricVectorizer   (Vista C, requiere spaCy es_core_news_sm)
  * EmotionLexiconVectorizer (Vista D, usa mini-lexicón sintético)

Más una clase de compatibilidad con la API de scikit-learn (clone, FeatureUnion).
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion

from src.feature_extractor import (
    DomainLexiconVectorizer,
    EmotionLexiconVectorizer,
    LexicalTfidfVectorizer,
    StylometricVectorizer,
)


# ---------------------------------------------------------------------------
# Clase 1: Vista A — Léxica (TF-IDF)
# ---------------------------------------------------------------------------

class TestLexicalTfidfVectorizer:

    def test_output_is_sparse_matrix(self, sample_texts_full):
        vec = LexicalTfidfVectorizer()
        X = vec.fit_transform(sample_texts_full)
        assert sparse.issparse(X)

    def test_output_shape_matches_input(self, sample_texts_full):
        vec = LexicalTfidfVectorizer()
        X = vec.fit_transform(sample_texts_full)
        assert X.shape[0] == len(sample_texts_full)

    def test_fit_transform_equals_fit_then_transform(self, sample_texts_full):
        vec_a = LexicalTfidfVectorizer()
        X_a = vec_a.fit_transform(sample_texts_full).toarray()

        vec_b = LexicalTfidfVectorizer()
        vec_b.fit(sample_texts_full)
        X_b = vec_b.transform(sample_texts_full).toarray()

        np.testing.assert_array_almost_equal(X_a, X_b)

    def test_max_features_respected(self, sample_texts_full):
        vec = LexicalTfidfVectorizer(max_features_word=5, min_df=1)
        X = vec.fit_transform(sample_texts_full)
        assert X.shape[1] <= 5

    def test_min_df_filters_hapax(self):
        texts = ["palabra unica solamente", "otra palabra cosa"]
        vec = LexicalTfidfVectorizer(min_df=2, max_features_word=100)
        vec.fit(texts)
        # "palabra" aparece en ambos, "unica"/"otra"/"cosa" solo en uno → filtradas
        vocab = vec.vocabulary_
        assert "palabra" in vocab
        assert "unica" not in vocab
        assert "solamente" not in vocab

    def test_vocabulary_not_empty_after_fit(self, sample_texts_full):
        vec = LexicalTfidfVectorizer(min_df=1)
        vec.fit(sample_texts_full)
        assert len(vec.vocabulary_) > 0

    def test_char_ngrams_add_columns(self, sample_texts_full):
        word_only = LexicalTfidfVectorizer(min_df=1, use_char_ngrams=False)
        word_only.fit(sample_texts_full)
        X_word = word_only.transform(sample_texts_full)

        with_char = LexicalTfidfVectorizer(min_df=1, use_char_ngrams=True)
        with_char.fit(sample_texts_full)
        X_both = with_char.transform(sample_texts_full)

        assert X_both.shape[1] > X_word.shape[1]

    def test_transform_before_fit_raises(self, sample_texts_full):
        vec = LexicalTfidfVectorizer()
        with pytest.raises(NotFittedError):
            vec.transform(sample_texts_full)

    def test_get_feature_names_out_after_fit(self, sample_texts_full):
        vec = LexicalTfidfVectorizer(min_df=1)
        vec.fit(sample_texts_full)
        names = vec.get_feature_names_out()
        assert len(names) > 0


# ---------------------------------------------------------------------------
# Clase 2: Vista B — Léxico de dominio
# ---------------------------------------------------------------------------

class TestDomainLexiconVectorizer:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.vec = DomainLexiconVectorizer()

    def test_output_is_dense_ndarray(self, sample_texts_minimal):
        X = self.vec.fit_transform(sample_texts_minimal)
        assert isinstance(X, np.ndarray)

    def test_output_shape(self, sample_texts_minimal):
        X = self.vec.fit_transform(sample_texts_minimal)
        assert X.shape[0] == len(sample_texts_minimal)
        assert X.shape[1] >= 30  # ~40 términos curados

    def test_counts_thinspo(self):
        X = self.vec.fit_transform(["thinspo thinspo otra cosa"])
        names = self.vec.get_feature_names_out()
        col = list(names).index("thinspo")
        assert X[0, col] > 0

    def test_counts_ayuno(self):
        X = self.vec.fit_transform(["ayuno extremo hoy ayuno"])
        names = self.vec.get_feature_names_out()
        col = list(names).index("ayuno")
        assert X[0, col] > 0

    def test_counts_cw(self):
        X = self.vec.fit_transform(["cw 50 gw 45 ugw 40"])
        names = list(self.vec.get_feature_names_out())
        assert X[0, names.index("cw")] > 0
        assert X[0, names.index("gw")] > 0
        assert X[0, names.index("ugw")] > 0

    def test_zero_vector_for_unrelated_text(self):
        X = self.vec.fit_transform(["el cielo está azul y los pájaros vuelan"])
        assert X[0].sum() == 0.0

    def test_normalization_scale_invariant(self):
        """Texto repetido N veces produce el mismo vector normalizado."""
        vec = DomainLexiconVectorizer(normalize=True)
        short = "thinspo ayuno"
        long = " ".join([short] * 10)
        X = vec.fit_transform([short, long])
        np.testing.assert_array_almost_equal(X[0], X[1])

    def test_feature_names_match_columns(self, sample_texts_minimal):
        X = self.vec.fit_transform(sample_texts_minimal)
        names = self.vec.get_feature_names_out()
        assert len(names) == X.shape[1]

    def test_handles_empty_string(self):
        X = self.vec.fit_transform([""])
        assert X.shape[0] == 1
        assert X[0].sum() == 0.0

    def test_accepts_numpy_array_input(self, sample_texts_minimal):
        arr = np.array(sample_texts_minimal)
        X = self.vec.fit_transform(arr)
        assert X.shape[0] == len(sample_texts_minimal)


# ---------------------------------------------------------------------------
# Clase 3: Vista C — Estilística (requiere spaCy es_core_news_sm)
# ---------------------------------------------------------------------------

class TestStylometricVectorizer:

    @pytest.fixture(autouse=True)
    def setup(self):
        pytest.importorskip("spacy", reason="spaCy no instalado")
        try:
            self.vec = StylometricVectorizer()
            self.vec.fit([""])  # valida que el modelo está disponible
        except RuntimeError as e:
            pytest.skip(str(e))

    def test_output_shape_is_12_columns(self, sample_texts_raw):
        X = self.vec.transform(sample_texts_raw)
        assert X.shape == (len(sample_texts_raw), 12)

    def test_output_is_dense_ndarray(self, sample_texts_raw):
        X = self.vec.transform(sample_texts_raw)
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float64

    def test_feature_names_length(self):
        names = self.vec.get_feature_names_out()
        assert len(names) == 12

    def test_ttr_for_all_repeated_token(self):
        """type-token ratio de 'hola hola hola' ≈ 1/3."""
        X = self.vec.transform(["hola hola hola"])
        names = list(self.vec.get_feature_names_out())
        ttr_col = names.index("type_token_ratio")
        assert X[0, ttr_col] == pytest.approx(1 / 3, abs=0.05)

    def test_ttr_for_all_unique_tokens(self):
        X = self.vec.transform(["uno dos tres cuatro cinco"])
        names = list(self.vec.get_feature_names_out())
        ttr_col = names.index("type_token_ratio")
        assert X[0, ttr_col] == pytest.approx(1.0, abs=0.01)

    def test_empty_text_returns_zero_vector(self):
        X = self.vec.transform([""])
        assert X.shape == (1, 12)
        np.testing.assert_array_equal(X[0], np.zeros(12))

    def test_uppercase_ratio_detects_caps(self):
        X = self.vec.transform(["AAAA aaaa"])
        names = list(self.vec.get_feature_names_out())
        col = names.index("uppercase_ratio")
        assert X[0, col] > 0.3  # 4 mayúsculas de ~9 caracteres

    def test_uppercase_ratio_zero_for_lowercase(self):
        X = self.vec.transform(["hola mundo nada raro"])
        names = list(self.vec.get_feature_names_out())
        col = names.index("uppercase_ratio")
        assert X[0, col] == 0.0

    def test_first_person_ratio_detects_yo(self):
        X = self.vec.transform(["yo yo yo yo yo"])
        names = list(self.vec.get_feature_names_out())
        col = names.index("first_person_ratio")
        assert X[0, col] > 0.5

    def test_negation_ratio_detects_no(self):
        X = self.vec.transform(["no no no no no"])
        names = list(self.vec.get_feature_names_out())
        col = names.index("negation_ratio")
        assert X[0, col] > 0.5

    def test_sentence_count_three_sentences(self):
        X = self.vec.transform(["Uno. Dos. Tres."])
        names = list(self.vec.get_feature_names_out())
        col = names.index("sentence_count")
        assert X[0, col] == pytest.approx(3, abs=1)

    def test_question_ratio_detects_question_marks(self):
        X = self.vec.transform(["¿Por qué? ¿Cuándo?"])
        names = list(self.vec.get_feature_names_out())
        col = names.index("question_ratio")
        assert X[0, col] > 0


# ---------------------------------------------------------------------------
# Clase 4: Vista D — Emocional (NRC EmoLex)
# ---------------------------------------------------------------------------

class TestEmotionLexiconVectorizer:

    def test_output_shape_is_10_columns(self, nrc_lexicon_path, sample_texts_minimal):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform(sample_texts_minimal)
        assert X.shape == (len(sample_texts_minimal), 10)

    def test_output_is_dense_float64(self, nrc_lexicon_path, sample_texts_minimal):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform(sample_texts_minimal)
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float64

    def test_sadness_detected(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform(["triste triste triste"])
        names = list(vec.get_feature_names_out())
        assert X[0, names.index("sadness")] > 0
        assert X[0, names.index("negative")] > 0

    def test_joy_detected(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform(["feliz alegre"])
        names = list(vec.get_feature_names_out())
        assert X[0, names.index("joy")] > 0
        assert X[0, names.index("positive")] > 0

    def test_unknown_words_yield_zeros(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform(["xyz abc def ghi"])
        assert X[0].sum() == 0.0

    def test_feature_names_canonical(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        vec.fit([""])
        names = list(vec.get_feature_names_out())
        assert set(names) == {
            "anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust", "negative", "positive",
        }

    def test_missing_lexicon_raises_runtime_error(self, tmp_path):
        bad_path = tmp_path / "no_existe.txt"
        vec = EmotionLexiconVectorizer(lexicon_path=bad_path)
        with pytest.raises(RuntimeError, match="NRC|EmoLex|download_nrc"):
            vec.fit([""])

    def test_empty_text_yields_zero_row(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        X = vec.fit_transform([""])
        assert X[0].sum() == 0.0


# ---------------------------------------------------------------------------
# Clase 5: Compatibilidad con scikit-learn
# ---------------------------------------------------------------------------

class TestSklearnCompatibility:

    def test_lexical_can_be_cloned(self):
        vec = LexicalTfidfVectorizer(max_features_word=100, min_df=1)
        cloned = clone(vec)
        assert cloned.max_features_word == 100
        assert cloned.min_df == 1

    def test_domain_can_be_cloned(self):
        vec = DomainLexiconVectorizer(normalize=False)
        cloned = clone(vec)
        assert cloned.normalize is False

    def test_emotion_can_be_cloned(self, nrc_lexicon_path):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path, normalize=False)
        cloned = clone(vec)
        assert cloned.normalize is False
        assert cloned.lexicon_path == nrc_lexicon_path

    def test_lexical_fit_returns_self(self, sample_texts_full):
        vec = LexicalTfidfVectorizer(min_df=1)
        assert vec.fit(sample_texts_full) is vec

    def test_domain_fit_returns_self(self, sample_texts_minimal):
        vec = DomainLexiconVectorizer()
        assert vec.fit(sample_texts_minimal) is vec

    def test_emotion_fit_returns_self(self, nrc_lexicon_path, sample_texts_minimal):
        vec = EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)
        assert vec.fit(sample_texts_minimal) is vec

    def test_feature_union_lexical_plus_domain(self, sample_texts_minimal):
        """Smoke test: dos vistas se concatenan vía FeatureUnion (valida la Sección 3.4)."""
        union = FeatureUnion([
            ("lexical", LexicalTfidfVectorizer(min_df=1)),
            ("domain", DomainLexiconVectorizer()),
        ])
        X = union.fit_transform(sample_texts_minimal)
        assert X.shape[0] == len(sample_texts_minimal)

    def test_feature_union_with_emotion(self, nrc_lexicon_path, sample_texts_minimal):
        union = FeatureUnion([
            ("domain", DomainLexiconVectorizer()),
            ("emotion", EmotionLexiconVectorizer(lexicon_path=nrc_lexicon_path)),
        ])
        X = union.fit_transform(sample_texts_minimal)
        assert X.shape[0] == len(sample_texts_minimal)
        # ambas son densas → la suma de columnas debe coincidir
        assert X.shape[1] >= 10 + 30
