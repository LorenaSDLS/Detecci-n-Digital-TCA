"""data.py — Carga de datos para las variantes de Fase 3.

A diferencia de :class:`src.data_loader.DataLoader` (que normaliza a
``[user_id, text, label]`` y descarta el identificador del texto), las variantes
de Fase 3 necesitan PRESERVAR el ``text_id`` para emitir predicciones alineables
con el baseline de Fase 2B y con el conjunto de prueba oficial.

El ``text_id`` se elige para coincidir con el que usa el baseline en
``predicciones_finales.csv`` (Esquema B → ``tweet_id``; Esquema A → ``text_id``),
de modo que ``compare.py`` pueda unir por clave todas las predicciones contra el
mismo conjunto de verdad-terreno.
"""

from __future__ import annotations

import re
from pathlib import Path

import ftfy
import pandas as pd
from sklearn.model_selection import train_test_split

# Un hashtag es ``#`` seguido de caracteres de palabra (incluye vocales
# acentuadas con re.UNICODE). Se comparte entre el loader, el script de limpieza
# y la generación de la variante "sin hashtags" para tener una sola definición.
_HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

# Esquema A (protocolo): id propio + título + cuerpo + etiqueta textual.
_SCHEMA_A_REQUIRED: frozenset[str] = frozenset({"text_id", "title", "text", "label"})
# Esquema B (tweets reales): el identificador es ``tweet_id`` y la clase ``class``.
_SCHEMA_B_REQUIRED: frozenset[str] = frozenset({"user_id", "tweet_id", "tweet_text", "class"})

_LABEL_MAP: dict[str, int] = {"anorexia": 1, "control": 0}


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """Lee un .csv/.xlsx y devuelve ``[text_id, text, label]`` con el id preservado.

    Args:
        filepath: Ruta al archivo de datos (con etiquetas).

    Returns:
        DataFrame con columnas exactamente ``[text_id, text, label]`` (label int).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el esquema de columnas no es reconocido, hay etiquetas
            inválidas, o el DataFrame queda vacío tras filtrar filas sin texto.
    """
    raw = _read_table(filepath)
    cols = set(raw.columns)

    if _SCHEMA_B_REQUIRED.issubset(cols):
        df = pd.DataFrame({
            "text_id": raw["tweet_id"],
            "text": raw["tweet_text"],
            "label": _map_labels(raw["class"]),
        })
    elif _SCHEMA_A_REQUIRED.issubset(cols):
        title = raw["title"].fillna("").astype(str)
        body = raw["text"].fillna("").astype(str)
        df = pd.DataFrame({
            "text_id": raw["text_id"],
            "text": (title + " " + body).str.strip(),
            "label": _map_labels(raw["label"]),
        })
    else:
        raise ValueError(
            f"Esquema de columnas no reconocido. Columnas: {sorted(cols)}. "
            f"Se esperaba Esquema A {sorted(_SCHEMA_A_REQUIRED)} "
            f"o Esquema B {sorted(_SCHEMA_B_REQUIRED)}."
        )

    df["text"] = df["text"].map(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("El DataFrame está vacío tras eliminar filas con texto inválido.")

    return df


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """División estratificada por etiqueta (preserva la proporción de clases).

    Args:
        df: DataFrame con columna ``label``.
        val_size: Proporción del conjunto de validación.
        seed: Semilla para reproducibilidad.

    Returns:
        Tupla ``(train_df, val_df)`` con índices reseteados.
    """
    train, val = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def load_gold(filepath: str | Path) -> pd.DataFrame:
    """Verdad-terreno para ``compare.py``: ``[text_id, label]`` del set de prueba."""
    return load_dataset(filepath)[["text_id", "label"]]


def strip_hashtags_text(text: str) -> str:
    """Elimina los hashtags (``#palabra``) de un texto y normaliza espacios.

    Base de la variante "sin hashtags": al quitar las etiquetas de comunidad
    (``#thinspo``, ``#proana``, ``#fit``…) se evalúa al modelo sin las pistas
    léxicas que separan trivialmente las clases.
    """
    if not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub(" ", _HASHTAG_RE.sub(" ", text)).strip()


def _read_table(filepath: str | Path) -> pd.DataFrame:
    """Lee CSV o XLSX según la extensión."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Extensión no soportada: {suffix!r}. Use .csv o .xlsx.")


def _map_labels(col: pd.Series) -> pd.Series:
    """Convierte etiquetas textuales {anorexia, control} a enteros {1, 0}."""
    lowered = col.astype(str).str.lower().str.strip()
    unknown = set(lowered.unique()) - set(_LABEL_MAP)
    if unknown:
        raise ValueError(
            f"Valores de etiqueta no reconocidos: {unknown}. "
            f"Permitidos: {set(_LABEL_MAP)}."
        )
    return lowered.map(_LABEL_MAP).astype(int)
