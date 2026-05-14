"""
conftest.py — Fixtures compartidas para las pruebas unitarias.

Autor: Andrea Blanco (data_loader, preprocessor)
       Carlos Zamudio (feature_extractor)
"""

from pathlib import Path

import pandas as pd
import pytest
import os


@pytest.fixture
def df_schema_b():
    """DataFrame válido en Esquema B (datos reales: tweet_text / class)."""
    return pd.DataFrame({
        "user_id":    ["u1", "u2", "u3", "u4"],
        "tweet_id":   ["t1", "t2", "t3", "t4"],
        "tweet_text": [
            "no puedo comer nada hoy",
            "todo está bien me siento feliz",
            "mi peso es demasiado necesito perder kilos",
            "hoy comí con mi familia fue genial",
        ],
        "class": ["anorexia", "control", "anorexia", "control"],
    })


@pytest.fixture
def df_schema_a():
    """DataFrame válido en Esquema A (hipotético: title + text / label)."""
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "text_id": ["t1", "t2"],
        "title":   ["ayuno extremo", "desayuno saludable"],
        "text":    ["llevo tres días sin comer nada", "hoy comí tostadas con aguacate"],
        "label":   ["anorexia", "control"],
    })


@pytest.fixture
def df_balanced_100():
    """DataFrame balanceado de 100 filas (50 por clase) para pruebas de partición."""
    return pd.DataFrame({
        "user_id":    [f"u{i}" for i in range(100)],
        "tweet_id":   [f"t{i}" for i in range(100)],
        "tweet_text": ["texto de prueba número " + str(i) for i in range(100)],
        "class":      ["anorexia"] * 50 + ["control"] * 50,
    })


@pytest.fixture
def df_schema_b_with_empty(df_schema_b):
    """Esquema B con filas de texto vacío y solo espacios en blanco."""
    df = df_schema_b.copy()
    df.loc[0, "tweet_text"] = ""
    df.loc[1, "tweet_text"] = "   "
    return df


@pytest.fixture
def df_schema_b_with_nan(df_schema_b):
    """Esquema B con una fila de texto nulo (NaN)."""
    df = df_schema_b.copy()
    df.loc[0, "tweet_text"] = None
    return df


# ---------------------------------------------------------------------------
# Fixtures para tests de feature_extractor (Carlos Zamudio)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_texts_minimal() -> list[str]:
    """Textos preprocesados con perfil mínimo (lower, sin URLs/emojis)."""
    return [
        "no puedo comer nada hoy mi peso me agobia",
        "thinspo cw 50 gw 45 ugw 40 no puedo más",
        "hoy comí con mi familia y me siento muy feliz",
        "ayuno otra vez para perder kilos",
        "estoy contenta no necesito restringir nada",
    ]


@pytest.fixture
def sample_texts_full() -> list[str]:
    """Textos preprocesados con perfil completo (lemas, sin stopwords)."""
    return [
        "no poder comer nada peso agobiar",
        "thinspo cw gw ugw no poder",
        "comer familia sentir feliz",
        "ayuno perder kilo",
        "contento no restringir",
    ]


@pytest.fixture
def sample_texts_raw() -> list[str]:
    """Textos crudos sin preprocesar (para StylometricVectorizer)."""
    return [
        "No puedo comer nada hoy. Mi peso me agobia mucho.",
        "¡Otro día más sin comida! ¿Cuándo terminará?",
        "Hoy comí con mi familia. Fue maravilloso. Me sentí bien.",
        "AYUNO TOTAL HOY. NO PUEDO MÁS.",
        "Estoy contenta y tranquila con mi cuerpo hoy.",
    ]


@pytest.fixture
def nrc_lexicon_path(tmp_path) -> Path:
    """Mini-lexicón NRC sintético para tests (formato largo word/emotion/value).

    Permite que los tests de EmotionLexiconVectorizer corran sin depender del
    archivo real de ~600KB descargado por scripts/download_nrc.py.
    """
    stub = tmp_path / "nrc_stub.txt"
    rows = [
        # palabra      emoción          valor
        ("triste",     "sadness",       1),
        ("triste",     "negative",      1),
        ("feliz",      "joy",           1),
        ("feliz",      "positive",      1),
        ("alegre",     "joy",           1),
        ("alegre",     "positive",      1),
        ("miedo",      "fear",          1),
        ("miedo",      "negative",      1),
        ("enojo",      "anger",         1),
        ("enojo",      "negative",      1),
        ("confianza",  "trust",         1),
        ("confianza",  "positive",      1),
        ("sorpresa",   "surprise",      1),
        ("asco",       "disgust",       1),
        ("asco",       "negative",      1),
        ("esperar",    "anticipation",  1),
    ]
    stub.write_text(
        "\n".join(f"{w}\t{e}\t{v}" for w, e, v in rows),
        encoding="utf-8",
    )
    return stub

# ---------------------------------------------------------------------------
# (Lorens Solís)
#Fixture que se ejecuta para limpiar los archivos Excel generados por los módulos después de cada test.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_outputs():
    """
    Fixture que se ejecuta automáticamente para limpiar los archivos 
    Excel generados por los módulos de Lorena después de cada test.
    """
    yield 
    
    # Después del test, borramos archivos temporales si existen
    files_to_clean = [
        "predicciones_finales.csv",
        "predicciones_finales.xlsx",
        "analisis_clinico_falsos_negativos.csv",
        "analisis_clinico_falsos_negativos.xlsx",
        "curva_roc.png",
        "importancia_atributos.png"
    ]
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)