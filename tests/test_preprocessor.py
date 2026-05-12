"""
test_preprocessor.py — Pruebas unitarias para el módulo preprocessor.

Autor: Andrea Blanco

Cubre para el perfil "minimal":
  - Conversión a minúsculas.
  - Eliminación de URLs, menciones, emojis.
  - Conservación de la palabra en hashtags (remoción del #).
  - Expansión de abreviaturas de redes sociales.
  - Conservación de negaciones y pronombres de primera persona.
  - Conservación de caracteres especiales españoles (ñ, tildes).
  - Colapso de caracteres repetidos.
  - Manejo de cadenas vacías y solo espacios.

Cubre para el perfil "full":
  - Eliminación de stopwords comunes.
  - Conservación de negaciones y pronombres en texto lematizado.
  - Conservación de términos corporales y alimenticios.
  - Lematización de conjugaciones verbales.
  - Manejo de cadenas vacías.

Cubre procesamiento en lote:
  - Consistencia entre preprocess() individual y preprocess_batch().
"""

import pytest

from src.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Clase 1: Perfil mínimo
# ---------------------------------------------------------------------------

class TestMinimalProfile:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prep = Preprocessor(profile="minimal")

    def test_lowercases_text(self):
        assert self.prep.preprocess("HOLA Mundo") == "hola mundo"

    def test_removes_http_url(self):
        result = self.prep.preprocess("visita https://example.com hoy")
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_www_url(self):
        result = self.prep.preprocess("ve a www.ejemplo.org para más info")
        assert "www" not in result

    def test_removes_mention(self):
        result = self.prep.preprocess("hola @usuario123 que tal")
        assert "@usuario123" not in result
        assert "usuario123" not in result

    def test_strips_hashtag_symbol_keeps_word(self):
        result = self.prep.preprocess("me gusta el #fitness hoy")
        assert "#" not in result
        assert "fitness" in result

    def test_removes_emoji_face(self):
        result = self.prep.preprocess("me siento mal 😢")
        assert "😢" not in result

    def test_removes_emoji_heart(self):
        result = self.prep.preprocess("te quiero ❤️")
        assert "❤" not in result

    def test_expands_xq_to_porque(self):
        result = self.prep.preprocess("no como xq no quiero")
        assert "porque" in result

    def test_expands_pq_to_porque(self):
        result = self.prep.preprocess("pq estás triste")
        assert "porque" in result

    def test_expands_tb_to_tambien(self):
        result = self.prep.preprocess("tb quiero perder peso")
        assert "también" in result

    def test_expands_q_to_que(self):
        result = self.prep.preprocess("todo lo q como me hace daño")
        assert "que" in result

    def test_preserves_negation_no(self):
        result = self.prep.preprocess("no puedo comer nada hoy")
        assert "no" in result.split()

    def test_preserves_negation_nunca(self):
        result = self.prep.preprocess("nunca como bien")
        assert "nunca" in result.split()

    def test_preserves_negation_ni(self):
        result = self.prep.preprocess("ni agua ni comida")
        assert "ni" in result.split()

    def test_preserves_first_person_yo(self):
        result = self.prep.preprocess("yo me siento sola")
        assert "yo" in result.split()

    def test_preserves_first_person_me(self):
        result = self.prep.preprocess("me duele el estómago")
        assert "me" in result.split()

    def test_preserves_spanish_n_tilde(self):
        result = self.prep.preprocess("mañana comeré más")
        assert "mañana" in result

    def test_preserves_accented_vowels(self):
        result = self.prep.preprocess("también me siento así")
        assert "así" in result or "tambien" in result or "también" in result

    def test_collapses_repeated_chars(self):
        result = self.prep.preprocess("jaaaaja no puedo más")
        assert "jaaaaja" not in result

    def test_empty_string_returns_empty(self):
        assert self.prep.preprocess("") == ""

    def test_whitespace_only_returns_empty(self):
        assert self.prep.preprocess("   ") == ""

    def test_only_special_chars_produces_empty_or_whitespace(self):
        result = self.prep.preprocess("!!! ### @@@")
        assert result.strip() == ""

    def test_text_without_noise_unchanged_content(self):
        result = self.prep.preprocess("no como nada")
        assert "no" in result
        assert "como" in result
        assert "nada" in result


# ---------------------------------------------------------------------------
# Clase 2: Perfil completo (requiere spaCy es_core_news_sm)
# ---------------------------------------------------------------------------

class TestFullProfile:

    @pytest.fixture(autouse=True)
    def setup(self):
        pytest.importorskip("spacy", reason="spaCy no instalado")
        try:
            self.prep = Preprocessor(profile="full")
        except RuntimeError as e:
            pytest.skip(str(e))

    def test_removes_common_stopword_para(self):
        result = self.prep.preprocess("voy para la tienda con ella")
        assert "para" not in result.split()

    def test_removes_common_stopword_con(self):
        result = self.prep.preprocess("salí con mis amigos ayer")
        assert "con" not in result.split()

    def test_preserves_negation_no_in_full(self):
        result = self.prep.preprocess("no quiero comer nada hoy")
        assert "no" in result.split()

    def test_preserves_first_person_yo_in_full(self):
        result = self.prep.preprocess("yo no puedo comer")
        assert "yo" in result.split()

    def test_preserves_body_term_peso(self):
        result = self.prep.preprocess("mi peso es demasiado")
        tokens = result.split()
        assert "peso" in tokens or "pesar" in tokens

    def test_preserves_food_term_comer(self):
        result = self.prep.preprocess("no quiero comer nada")
        assert "comer" in result.split()

    def test_lemmatizes_verb_conjugation_comiendo(self):
        result = self.prep.preprocess("estoy comiendo muy poco")
        tokens = result.split()
        assert "comer" in tokens, f"Se esperaba lema 'comer', tokens obtenidos: {tokens}"

    def test_empty_string_returns_empty_in_full(self):
        assert self.prep.preprocess("") == ""

    def test_result_is_string(self):
        result = self.prep.preprocess("no como por las mañanas")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Clase 3: Perfil inválido
# ---------------------------------------------------------------------------

class TestInvalidProfile:

    def test_invalid_profile_raises_value_error(self):
        with pytest.raises(ValueError, match="profile"):
            Preprocessor(profile="ultra")

    def test_none_profile_raises_value_error(self):
        with pytest.raises((ValueError, TypeError)):
            Preprocessor(profile=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Clase 4: Procesamiento en lote
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    def test_batch_minimal_matches_individual(self):
        texts = [
            "no como xq estoy a dieta",
            "hola @user visita https://x.com hoy",
            "",
        ]
        prep = Preprocessor(profile="minimal")
        batch = prep.preprocess_batch(texts)
        individual = [prep.preprocess(t) for t in texts]
        assert batch == individual

    def test_batch_minimal_returns_list(self):
        prep = Preprocessor(profile="minimal")
        result = prep.preprocess_batch(["texto uno", "texto dos"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_batch_minimal_empty_list(self):
        prep = Preprocessor(profile="minimal")
        assert prep.preprocess_batch([]) == []

    def test_batch_full_matches_individual(self):
        pytest.importorskip("spacy", reason="spaCy no instalado")
        try:
            prep = Preprocessor(profile="full")
        except RuntimeError as e:
            pytest.skip(str(e))

        texts = ["no quiero comer", "me duele el cuerpo hoy"]
        batch = prep.preprocess_batch(texts)
        individual = [prep.preprocess(t) for t in texts]
        assert batch == individual
