"""
preprocessor.py — Módulo de preprocesamiento de texto.

Autor: Andrea Blanco

Provee dos perfiles de preprocesamiento para texto en español proveniente
de redes sociales:

  * Perfil "minimal" (vista emocional): normalización ligera que conserva
    stopwords, pronombres y negaciones para preservar la señal emocional
    (Aragón et al., 2023).

  * Perfil "full" (vista léxica): aplica todas las transformaciones del perfil
    mínimo más tokenización, eliminación de stopwords con lista personalizada
    y lematización mediante spaCy (es_core_news_sm).
"""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Constantes de la clase (definidas a nivel de módulo para claridad)
# ---------------------------------------------------------------------------

# Abreviaturas comunes en redes sociales en español
_ABBREV_MAP: dict[str, str] = {
    r"\bxq\b": "porque",
    r"\bpq\b": "porque",
    r"\btb\b": "también",
    r"\btbm\b": "también",
    r"\bq\b": "que",
    r"\bxfa\b": "por favor",
    r"\bpfv\b": "por favor",
    r"\bk\b": "que",
    r"\bmñn\b": "mañana",
    r"\bstas\b": "estás",
    r"\bsta\b": "está",
    r"\bsto\b": "esto",
    r"\bx\b": "por",
    r"\bxd\b": "",
    r"\bjajaja+\b": "jaja",
    r"\bhahaha+\b": "jaja",
}

# Términos que nunca se eliminan en el perfil completo aunque sean stopwords
_PRESERVED_TERMS: frozenset[str] = frozenset({
    # Pronombres de primera persona
    "yo", "me", "mi", "mío", "mía", "míos", "mías", "conmigo",
    # Negaciones
    "no", "ni", "nunca", "jamás", "tampoco", "sin", "nadie", "nada",
    # Términos corporales
    "cuerpo", "peso", "grasa", "caloria", "caloría", "kilo", "talla",
    "figura", "estómago", "abdomen", "muslo", "cintura", "hueso", "costilla",
    # Términos alimenticios
    "comer", "comida", "hambre", "ayuno", "dieta", "laxante", "purga",
    "vomitar", "atracón", "restricción", "binge", "purgar", "ayunar",
    # Adjetivos/estados relacionados
    "delgado", "gordo", "obeso", "flaco", "delgada", "gorda", "flaca",
})


class Preprocessor:
    """Normalizador de texto para detección de desórdenes alimenticios.

    Attributes:
        profile: ``'minimal'`` para el perfil emocional o ``'full'`` para el léxico.
    """

    # Expresiones regulares compiladas a nivel de clase para eficiencia
    _URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _MENTION = re.compile(r"@\w+")
    _HASHTAG = re.compile(r"#(\w+)")          # elimina # pero conserva la palabra
    _EMOJI = re.compile(
        "[\U00002600-\U000027BF]|[\U0001F300-\U0001FAFF]|"
        "[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|"
        "[\U00002702-\U000027B0]",
        flags=re.UNICODE,
    )
    _REPEATED = re.compile(r"(.)\1{2,}")      # jaaaaja → jaja (máximo 2)
    _SPECIAL = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")
    _WHITESPACE = re.compile(r"\s+")

    def __init__(self, profile: Literal["minimal", "full"] = "minimal") -> None:
        """Inicializa el preprocesador.

        Args:
            profile: ``'minimal'`` conserva stopwords/pronombres/negaciones;
                ``'full'`` aplica lematización y eliminación de stopwords.

        Raises:
            ValueError: Si ``profile`` no es ``'minimal'`` ni ``'full'``.
            RuntimeError: Si ``profile='full'`` y el modelo ``es_core_news_sm``
                de spaCy no está instalado.
        """
        if profile not in ("minimal", "full"):
            raise ValueError(f"profile debe ser 'minimal' o 'full', se recibió: {profile!r}")

        self.profile = profile
        self._nlp = None  # carga diferida (lazy loading)

        if profile == "full":
            self._check_spacy_model()
            # Precalcular el conjunto de stopwords con excepciones
            from spacy.lang.es.stop_words import STOP_WORDS as _ES_STOP_WORDS
            base_stopwords: set[str] = set(_ES_STOP_WORDS)
            self._effective_stopwords: frozenset[str] = frozenset(
                base_stopwords - _PRESERVED_TERMS
            )

        # Compilar los patrones de abreviatura una sola vez
        self._abbrev_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in _ABBREV_MAP.items()
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Normaliza un texto según el perfil configurado.

        Args:
            text: Texto crudo en español.

        Returns:
            Texto preprocesado como cadena de caracteres.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        if self.profile == "minimal":
            return self._minimal(text)
        return self._full(text)

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Normaliza una lista de textos.

        Para el perfil ``'full'``, utiliza ``nlp.pipe()`` en lotes para
        mayor eficiencia respecto a llamadas individuales.

        Args:
            texts: Lista de textos crudos.

        Returns:
            Lista de textos preprocesados en el mismo orden.
        """
        if self.profile == "minimal":
            return [self.preprocess(t) for t in texts]

        # Perfil full: aplicar minimal primero, luego pasar por spaCy en lote
        minimized = [self._minimal(t) for t in texts]
        results: list[str] = []
        for doc in self.nlp.pipe(minimized, batch_size=64):
            results.append(self._lemmatize_and_filter(doc))
        return results

    # ------------------------------------------------------------------
    # Propiedad lazy para el modelo spaCy
    # ------------------------------------------------------------------

    @property
    def nlp(self):
        """Carga el modelo spaCy ``es_core_news_sm`` en el primer acceso."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        return self._nlp

    # ------------------------------------------------------------------
    # Métodos privados de transformación
    # ------------------------------------------------------------------

    def _minimal(self, text: str) -> str:
        """Perfil mínimo: normalización ligera para preservar señal emocional.

        Pasos (en orden):
        1. Conversión a minúsculas.
        2. Expansión de abreviaturas de redes sociales.
        3. Eliminación de URLs.
        4. Eliminación de menciones (@usuario).
        5. Eliminación del símbolo # conservando la palabra del hashtag.
        6. Eliminación de emojis.
        7. Colapso de caracteres repetidos (3+ → 2).
        8. Eliminación de caracteres especiales (conservando diacríticos españoles).
        9. Normalización de espacios en blanco.
        """
        text = text.lower()

        for pattern, replacement in self._abbrev_patterns:
            text = pattern.sub(replacement, text)

        text = self._URL.sub(" ", text)
        text = self._MENTION.sub(" ", text)
        text = self._HASHTAG.sub(r"\1", text)    # "#fitness" → "fitness"
        text = self._EMOJI.sub(" ", text)
        text = self._REPEATED.sub(r"\1\1", text)  # "jaaaaja" → "jaaja"
        text = self._SPECIAL.sub(" ", text)
        text = self._WHITESPACE.sub(" ", text).strip()
        return text

    def _full(self, text: str) -> str:
        """Perfil completo: lematización y eliminación de stopwords tras el perfil mínimo."""
        minimized = self._minimal(text)
        if not minimized:
            return ""
        doc = self.nlp(minimized)
        return self._lemmatize_and_filter(doc)

    def _lemmatize_and_filter(self, doc) -> str:
        """Aplica lematización y filtra stopwords con excepciones.

        Un token se conserva si:
        - No es una stopword, O
        - Su lema (en minúsculas) pertenece a ``_PRESERVED_TERMS``.
        - No es puntuación ni espacio.

        Args:
            doc: Objeto ``Doc`` de spaCy.

        Returns:
            Cadena con lemas separados por espacios.
        """
        lemmas: list[str] = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            # Conservar si el lema está en términos preservados o no es stopword
            if lemma in _PRESERVED_TERMS or lemma not in self._effective_stopwords:
                lemmas.append(lemma)
        return " ".join(lemmas)

    # ------------------------------------------------------------------
    # Validación de dependencias
    # ------------------------------------------------------------------

    @staticmethod
    def _check_spacy_model() -> None:
        """Verifica que el modelo español de spaCy esté disponible.

        Raises:
            RuntimeError: Si ``es_core_news_sm`` no está instalado.
        """
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
