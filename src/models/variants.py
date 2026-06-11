"""variants.py — Variantes de datos para la matriz comparativa (Fase 3+).

Dos variantes del mismo conjunto, para medir cuánto depende cada método de las
pistas léxicas de los hashtags:

  * ``con_hashtags`` — datos originales (``data_train.csv`` / ``data_test_fold1.csv``).
  * ``sin_hashtags`` — los mismos textos con los hashtags eliminados.

La variante sin-hashtags se GENERA bajo demanda a partir de la original (no se
versiona): si el archivo derivado falta o es más viejo que su fuente, se
regenera con :func:`src.models.data.strip_hashtags_text`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.data import strip_hashtags_text

# Columna de texto en el esquema B (tweets reales); es la única que se limpia.
_TEXT_COL = "tweet_text"


DEFAULT_TRAIN = "data/data_train.csv"
DEFAULT_TEST = "data/data_test_fold1.csv"

# Sufijo de los archivos derivados sin hashtags (p. ej. ``..._nohashtags.csv``).
_NOHASH_SUFFIX = "_nohashtags"


@dataclass(frozen=True)
class DataVariant:
    """Una variante de datos: par (train, test) con una clave y etiqueta legible."""

    key: str
    label: str
    train_file: str
    test_file: str


def _nohashtags_name(path: str) -> str:
    """``data/x.csv`` → ``data/x_nohashtags.csv`` (nombre del derivado sin hashtags)."""
    p = Path(path)
    return str(p.parent / f"{p.stem}{_NOHASH_SUFFIX}{p.suffix}")


def build_variants(
    test_file: str = DEFAULT_TEST,
    train_file: str = DEFAULT_TRAIN,
) -> list[DataVariant]:
    """Construye las variantes con/sin hashtags para un fold de prueba dado.

    Permite evaluar cualquier ``data_test_foldN.csv`` (no sólo el fold 1): la
    variante sin-hashtags usa las versiones derivadas de ``train_file`` y
    ``test_file``, que se generan bajo demanda en :func:`ensure_variant_files`.
    """
    con = DataVariant("con_hashtags", "Con hashtags", train_file, test_file)
    sin = DataVariant(
        "sin_hashtags", "Sin hashtags",
        _nohashtags_name(train_file), _nohashtags_name(test_file),
    )
    return [con, sin]


# Variantes por defecto (fold 1), para compatibilidad con el flujo existente.
VARIANTS: list[DataVariant] = build_variants()
CON_HASHTAGS, SIN_HASHTAGS = VARIANTS


def ensure_variant_files(variant: DataVariant) -> None:
    """Garantiza que existan los archivos de la variante, generándolos si hace falta.

    Un archivo se considera derivado si su nombre termina en ``_nohashtags``; su
    fuente es el mismo nombre sin ese sufijo. Los archivos originales se dejan
    intactos.
    """
    for derived in (variant.train_file, variant.test_file):
        dst = Path(derived)
        if not dst.stem.endswith(_NOHASH_SUFFIX):
            continue  # archivo original: nada que derivar.

        base = dst.stem[: -len(_NOHASH_SUFFIX)]
        src = dst.parent / f"{base}{dst.suffix}"
        if not src.exists():
            raise FileNotFoundError(f"Falta el archivo fuente para la variante: {src}")
        # Reusar el derivado sólo si está al día respecto a la fuente.
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue

        df = pd.read_csv(src)
        if _TEXT_COL in df.columns:
            df[_TEXT_COL] = df[_TEXT_COL].fillna("").astype(str).map(strip_hashtags_text)
        df.to_csv(dst, index=False)
        print(f"  [variante] generado {dst.name} desde {src.name}")
