"""strip_hashtags.py — Genera versiones SIN hashtags de los datos (Fase 3 honesta).

Si la prueba de hipótesis (``keyword_baseline.py`` / ``keyword_leakage_report.py``)
confirma que la AUC alta se apoya en hashtags diagnósticos, este script produce
un corpus "honesto" eliminándolos. Re-entrenar sobre estos archivos da una cifra
defendible: lo que el modelo logra SIN las pistas obvias.

Nota: la matriz comparativa (``python -m src.models matrix``) ya genera estos
archivos automáticamente vía ``src.models.variants.ensure_variant_files``. Este
script es para uso manual y para la opción ``--strip-keywords`` (más estricta).

Escribe copias con el mismo esquema de columnas (``user_id, tweet_id,
tweet_text, class``) pero con los hashtags retirados de ``tweet_text``.

Uso:
    .venv/bin/python -m scripts.strip_hashtags
    .venv/bin/python -m scripts.strip_hashtags --strip-keywords
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.models.data import strip_hashtags_text

_WS_RE = re.compile(r"\s+", re.UNICODE)

# Columna de texto en el esquema B (tweets reales). Es la única que se limpia.
_TEXT_COL = "tweet_text"

# Términos pro-ana en texto plano (sólo si se pasa --strip-keywords). Se borran
# respetando límites de palabra para no mutilar términos legítimos.
_KEYWORDS = (
    "anorexia", "anoréxica", "anoréxico", "bulimia", "ayuno", "ayunar",
    "thinspo", "thinspiration", "proana", "promia", "anamia",
)
_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _KEYWORDS) + r")\b",
    re.IGNORECASE | re.UNICODE,
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for src in args.files:
        _process_file(Path(src), strip_keywords=args.strip_keywords)
    return 0


def _process_file(path: Path, strip_keywords: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)
    if _TEXT_COL not in df.columns:
        raise ValueError(
            f"{path.name}: se esperaba la columna {_TEXT_COL!r} (esquema B). "
            f"Columnas presentes: {list(df.columns)}."
        )

    original = df[_TEXT_COL].fillna("").astype(str)
    n_hashtags = int(original.map(lambda t: len(re.findall(r"#\w+", t))).sum())

    cleaned = original.map(lambda t: _clean(t, strip_keywords))
    became_empty = int((cleaned.str.strip().str.len() == 0).sum())

    df[_TEXT_COL] = cleaned
    out = path.with_name(f"{path.stem}_nohashtags{path.suffix}")
    df.to_csv(out, index=False)

    kw = " + palabras clave" if strip_keywords else ""
    print(
        f"[ok] {path.name} → {out.name}\n"
        f"     hashtags eliminados: {n_hashtags}{kw}\n"
        f"     filas que quedaron vacías: {became_empty} "
        f"(load_dataset las descartará al cargar)"
    )


def _clean(text: str, strip_keywords: bool) -> str:
    """Quita hashtags (helper compartido) y, opcionalmente, términos pro-ana."""
    text = strip_hashtags_text(text)
    if strip_keywords:
        text = _WS_RE.sub(" ", _KEYWORD_RE.sub(" ", text)).strip()
    return text


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "files", nargs="*",
        default=["data/data_train.csv", "data/data_test_fold1.csv"],
        help="CSV(s) de esquema B a limpiar (por defecto: train + test).",
    )
    p.add_argument(
        "--strip-keywords", action="store_true",
        help="Además de los hashtags, borra términos pro-ana en texto plano.",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
