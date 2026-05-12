"""
feature_extractor.py — Módulo de extracción de atributos multi-vista.

Autor: Carlos Zamudio

Implementa cuatro sub-extractores independientes y paralelizables, inspirados
en el catálogo de vistas propuesto por Villa-Pérez et al. (2023):

  * Vista A — Léxica (TF-IDF de n-gramas de palabras y opcionalmente caracteres).
  * Vista B — Léxico de dominio (conteos normalizados sobre diccionario pro-ana
    curado de Ramírez-Cifuentes et al. (2020) y Aguilera et al. (2021)).
  * Vista C — Estilística (métricas estilométricas tipo LIWC sobre texto crudo).
  * Vista D — Emocional (Bag of Emotions sobre NRC EmoLex, replicando
    conceptualmente el enfoque BoSE de Aragón et al. (2023)).

Cada sub-extractor implementa la interfaz fit/transform de scikit-learn
(``BaseEstimator`` + ``TransformerMixin``), permitiendo su composición limpia
en un ``FeatureUnion`` para la fusión multi-vista de la Sección 3.4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# ---------------------------------------------------------------------------
# Constantes a nivel de módulo
# ---------------------------------------------------------------------------

# Pronombres de primera persona (singular y plural) en español
_FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset({
    "yo", "me", "mi", "mí", "mío", "mía", "míos", "mías", "conmigo",
    "nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras",
})

# Negaciones en español
_NEGATIONS: frozenset[str] = frozenset({
    "no", "ni", "nunca", "jamás", "tampoco", "nada", "nadie", "sin",
})

# Diccionario de dominio pro-ana / anorexia (mezcla EN/ES porque la jerga
# se comparte transversalmente en comunidades pro-ana hispanohablantes).
# Curado de Ramírez-Cifuentes et al. (2020) y Aguilera et al. (2021).
_PROANA_LEXICON: dict[str, frozenset[str]] = {
    "peso": frozenset({
        "cw", "gw", "ugw", "lbw", "bmi", "imc",
        "peso", "kilos", "kilogramos", "gramos", "libras",
    }),
    "comida": frozenset({
        "calories", "calorías", "calorias", "macros", "carbs", "carbohidratos",
        "comida", "comer", "hambre", "atracón", "binge",
    }),
    "restriccion": frozenset({
        "restrict", "restricción", "restriccion", "restringir",
        "fast", "ayuno", "ayunar", "skipmeal", "skipping",
    }),
    "purga": frozenset({
        "purge", "purga", "purgar", "vomit", "vomitar",
        "laxante", "laxantes", "diurético", "mia",
    }),
    "cuerpo": frozenset({
        "thinspo", "bonespo", "thigh", "thighgap", "thinspiration",
        "delgado", "delgada", "flaco", "flaca",
    }),
    "ideologia": frozenset({
        "ana", "proana", "pro-ana", "ed", "tca", "anorexia",
        "skinny", "underweight", "fatspo", "fitspo",
    }),
}

# Lista plana ordenada de todos los términos del lexicón (orden determinístico)
_PROANA_TERMS: tuple[str, ...] = tuple(
    sorted(term for terms in _PROANA_LEXICON.values() for term in terms)
)

# Orden canónico de las 10 dimensiones del NRC Emotion Lexicon
_NRC_EMOTIONS: tuple[str, ...] = (
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "negative", "positive",
)

# Ruta por defecto al lexicón NRC EmoLex en español
_DEFAULT_NRC_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "data" / "lexicons" / "NRC-Emotion-Lexicon-v0.92-Spanish.txt"
)


# ---------------------------------------------------------------------------
# Vista A — Léxica (TF-IDF)
# ---------------------------------------------------------------------------

class LexicalTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Vista A: matriz TF-IDF de n-gramas de palabras (1-3) con char opcional (3-5).

    Encapsula uno o dos ``TfidfVectorizer`` de scikit-learn. Cuando
    ``use_char_ngrams=True``, compone internamente un ``FeatureUnion`` con dos
    vectorizadores (palabra y caracter) cuyas matrices dispersas se concatenan
    horizontalmente.

    Attributes:
        ngram_range_word: Rango de n-gramas de palabras.
        max_features_word: Vocabulario máximo de la vista de palabras.
        min_df: Frecuencia mínima de documento.
        use_char_ngrams: Si ``True``, añade vectorizador de char n-gramas.
        ngram_range_char: Rango de n-gramas de caracteres.
        max_features_char: Vocabulario máximo de la vista de caracteres.
    """

    def __init__(
        self,
        ngram_range_word: tuple[int, int] = (1, 3),
        max_features_word: int = 15000,
        min_df: int = 2,
        use_char_ngrams: bool = False,
        ngram_range_char: tuple[int, int] = (3, 5),
        max_features_char: int = 5000,
    ) -> None:
        self.ngram_range_word = ngram_range_word
        self.max_features_word = max_features_word
        self.min_df = min_df
        self.use_char_ngrams = use_char_ngrams
        self.ngram_range_char = ngram_range_char
        self.max_features_char = max_features_char

    def fit(self, X: Iterable[str], y=None) -> LexicalTfidfVectorizer:
        """Aprende el vocabulario TF-IDF a partir de ``X``."""
        self._vectorizer_ = self._build_vectorizer()
        self._vectorizer_.fit(list(X))
        return self

    def transform(self, X: Iterable[str]) -> sparse.csr_matrix:
        """Transforma textos a matriz dispersa TF-IDF.

        Returns:
            Matriz CSR de dimensión n × (max_features_word [+ max_features_char]).

        Raises:
            sklearn.exceptions.NotFittedError: Si se llama antes de ``fit``.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_vectorizer_")
        return self._vectorizer_.transform(list(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las columnas del vocabulario aprendido."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_vectorizer_")
        return self._vectorizer_.get_feature_names_out()

    @property
    def vocabulary_(self) -> dict[str, int]:
        """Vocabulario aprendido (delegado al vectorizador interno)."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_vectorizer_")
        # FeatureUnion no expone vocabulary_; en ese caso recopilamos manualmente.
        if isinstance(self._vectorizer_, FeatureUnion):
            return {
                name: idx for idx, name in enumerate(
                    self._vectorizer_.get_feature_names_out()
                )
            }
        return self._vectorizer_.vocabulary_

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _build_vectorizer(self):
        """Construye un ``TfidfVectorizer`` o un ``FeatureUnion`` de dos."""
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range_word,
            max_features=self.max_features_word,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        if not self.use_char_ngrams:
            return word_vec
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range_char,
            max_features=self.max_features_char,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        return FeatureUnion([("word", word_vec), ("char", char_vec)])


# ---------------------------------------------------------------------------
# Vista B — Léxico de dominio
# ---------------------------------------------------------------------------

class DomainLexiconVectorizer(BaseEstimator, TransformerMixin):
    """Vista B: conteos normalizados sobre el diccionario pro-ana curado.

    Cuenta ocurrencias de cada término del lexicón por documento y normaliza
    por número de tokens del documento (frecuencia relativa), produciendo un
    vector denso e interpretable.

    Attributes:
        normalize: Si ``True`` (defecto), divide los conteos por el número
            de tokens del documento para producir frecuencias relativas.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def fit(self, X: Iterable[str], y=None) -> DomainLexiconVectorizer:
        """No-op; el lexicón es fijo. Existe sólo para cumplir la interfaz."""
        # Consumir el iterable por compatibilidad sklearn (no falla con generadores).
        _ = list(X)
        self._terms_ = _PROANA_TERMS
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Devuelve matriz densa n × |lexicón| con frecuencias relativas."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_terms_")

        texts = list(X)
        n_features = len(self._terms_)
        matrix = np.zeros((len(texts), n_features), dtype=np.float64)

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                continue
            tokens = text.split()
            n_tokens = len(tokens)
            if n_tokens == 0:
                continue
            token_set = tokens  # contamos por aparición exacta de token
            for j, term in enumerate(self._terms_):
                # Conteo de tokens iguales al término (insensible a may/min);
                # los textos preprocesados con perfil mínimo ya vienen en lower.
                count = sum(1 for t in token_set if t == term)
                matrix[i, j] = count
            if self.normalize:
                matrix[i] /= n_tokens

        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de los términos del lexicón en orden columna."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_terms_")
        return np.array(self._terms_)


# ---------------------------------------------------------------------------
# Vista C — Estilística
# ---------------------------------------------------------------------------

class StylometricVectorizer(BaseEstimator, TransformerMixin):
    """Vista C: métricas estilométricas tipo LIWC calculadas con spaCy.

    Procesa texto **crudo** (sin preprocesar) porque requiere mayúsculas y
    puntuación originales. Produce un vector denso de 12 features fijas.

    Raises:
        RuntimeError: Si el modelo ``es_core_news_sm`` de spaCy no está instalado.
    """

    FEATURE_NAMES: tuple[str, ...] = (
        "char_count",
        "token_count",
        "sentence_count",
        "avg_sentence_length",
        "avg_word_length",
        "type_token_ratio",
        "first_person_ratio",
        "negation_ratio",
        "punctuation_ratio",
        "uppercase_ratio",
        "exclamation_ratio",
        "question_ratio",
    )

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self._nlp = None

    def fit(self, X: Iterable[str], y=None) -> StylometricVectorizer:
        """Verifica disponibilidad de spaCy. Las métricas son cerradas, no se aprenden."""
        self._check_spacy_model()
        _ = list(X)
        self._fitted_ = True
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Calcula 12 métricas estilométricas por documento."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_fitted_")

        texts = [t if isinstance(t, str) else "" for t in X]
        n = len(texts)
        matrix = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float64)

        # nlp.pipe procesa en lote para eficiencia
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=self.batch_size)):
            matrix[i] = self._features_for_doc(doc, texts[i])

        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.FEATURE_NAMES)

    @property
    def nlp(self):
        """Carga diferida del modelo spaCy ``es_core_news_sm``."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("es_core_news_sm", disable=["ner"])
        return self._nlp

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _features_for_doc(self, doc, raw_text: str) -> np.ndarray:
        """Calcula el vector de 12 features para un único documento."""
        n_chars = len(raw_text)
        tokens = [t for t in doc if not t.is_space]
        n_tokens = len(tokens)

        if n_tokens == 0 or n_chars == 0:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)

        word_tokens = [t for t in tokens if not t.is_punct]
        n_words = len(word_tokens) or 1  # evitar div/0

        sentences = list(doc.sents)
        n_sents = len(sentences) or 1

        types = {t.lower_ for t in word_tokens}
        ttr = len(types) / n_words

        lowered = [t.lower_ for t in word_tokens]
        n_first_person = sum(1 for w in lowered if w in _FIRST_PERSON_PRONOUNS)
        n_negations = sum(1 for w in lowered if w in _NEGATIONS)
        n_punct = sum(1 for t in tokens if t.is_punct)
        n_upper = sum(1 for c in raw_text if c.isupper())
        n_excl = raw_text.count("!") + raw_text.count("¡")
        n_quest = raw_text.count("?") + raw_text.count("¿")

        avg_sentence_length = n_tokens / n_sents
        avg_word_length = sum(len(t.text) for t in word_tokens) / n_words

        return np.array([
            float(n_chars),
            float(n_tokens),
            float(n_sents),
            avg_sentence_length,
            avg_word_length,
            ttr,
            n_first_person / n_words,
            n_negations / n_words,
            n_punct / n_tokens,
            n_upper / n_chars,
            n_excl / n_chars,
            n_quest / n_chars,
        ], dtype=np.float64)

    @staticmethod
    def _check_spacy_model() -> None:
        """Verifica que ``es_core_news_sm`` esté instalado (mismo patrón que Preprocessor)."""
        try:
            import spacy
            if not spacy.util.is_package("es_core_news_sm"):
                raise RuntimeError(
                    "El modelo de spaCy 'es_core_news_sm' no está instalado. "
                    "Instálalo ejecutando:\n"
                    "    uv run python -m spacy download es_core_news_sm"
                )
        except ImportError as exc:
            raise RuntimeError(
                "spaCy no está instalado. Ejecuta: uv add spacy"
            ) from exc


# ---------------------------------------------------------------------------
# Vista D — Emocional (NRC EmoLex)
# ---------------------------------------------------------------------------

class EmotionLexiconVectorizer(BaseEstimator, TransformerMixin):
    """Vista D: Bag of Emotions sobre el NRC Emotion Lexicon en español.

    Produce un vector denso de 10 dimensiones (8 emociones básicas + 2
    sentimientos) por documento. Los conteos se normalizan por número de
    tokens del documento, replicando el enfoque BoSE de Aragón et al. (2023)
    pero omitiendo el clustering con FastText (régimen sin embeddings).

    Attributes:
        lexicon_path: Ruta al archivo TSV del NRC EmoLex Spanish. Si es ``None``,
            usa ``data/lexicons/NRC-Emotion-Lexicon-v0.92-Spanish.txt``.
        normalize: Si ``True`` (defecto), divide por número de tokens.
    """

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        self.lexicon_path = lexicon_path
        self.normalize = normalize
        self._lexicon: dict[str, np.ndarray] | None = None

    def fit(self, X: Iterable[str], y=None) -> EmotionLexiconVectorizer:
        """Verifica disponibilidad del archivo y carga el lexicón en memoria."""
        _ = self.lexicon  # fuerza carga (valida existencia del archivo)
        _ = list(X)
        self._fitted_ = True
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Devuelve matriz densa n × 10 con frecuencias relativas por emoción."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_fitted_")

        texts = list(X)
        n = len(texts)
        matrix = np.zeros((n, len(_NRC_EMOTIONS)), dtype=np.float64)
        lexicon = self.lexicon

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                continue
            tokens = text.split()
            n_tokens = len(tokens)
            if n_tokens == 0:
                continue
            for token in tokens:
                vec = lexicon.get(token)
                if vec is not None:
                    matrix[i] += vec
            if self.normalize:
                matrix[i] /= n_tokens

        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(_NRC_EMOTIONS)

    @property
    def lexicon(self) -> dict[str, np.ndarray]:
        """Carga diferida del lexicón NRC EmoLex desde el archivo TSV."""
        if self._lexicon is None:
            path = Path(self.lexicon_path) if self.lexicon_path else _DEFAULT_NRC_PATH
            self._lexicon = self._load_lexicon(path)
        return self._lexicon

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _load_lexicon(path: Path) -> dict[str, np.ndarray]:
        """Lee el TSV de NRC EmoLex y construye dict palabra → vector(10).

        El archivo oficial tiene tres columnas: ``word``, ``emotion``, ``value``.
        Cada palabra aparece 10 veces (una por emoción/sentimiento).

        Raises:
            RuntimeError: Si el archivo no existe en la ruta indicada.
        """
        if not path.exists():
            raise RuntimeError(
                f"Lexicón NRC EmoLex no encontrado en: {path}\n"
                "Descárgalo ejecutando:\n"
                "    uv run python -m scripts.download_nrc"
            )

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["word", "emotion", "value"],
            keep_default_na=False,
        )
        df = df[df["value"] == 1]

        emotion_idx: dict[str, int] = {e: i for i, e in enumerate(_NRC_EMOTIONS)}
        lexicon: dict[str, np.ndarray] = {}
        for word, emotion in zip(df["word"].astype(str), df["emotion"].astype(str)):
            word_lower = word.lower().strip()
            if not word_lower or emotion not in emotion_idx:
                continue
            if word_lower not in lexicon:
                lexicon[word_lower] = np.zeros(len(_NRC_EMOTIONS), dtype=np.float64)
            lexicon[word_lower][emotion_idx[emotion]] = 1.0

        if not lexicon:
            raise RuntimeError(
                f"El lexicón en {path} está vacío o tiene formato inesperado. "
                f"Se esperaban columnas 'word\\temotion\\tvalue'."
            )
        return lexicon
