"""
conftest.py — Fixtures compartidas para las pruebas unitarias.

Autor: Andrea Blanco
"""

import pandas as pd
import pytest


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
