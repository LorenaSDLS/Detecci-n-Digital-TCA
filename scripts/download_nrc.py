"""
download_nrc.py — Descarga el NRC Emotion Lexicon (Spanish) para la Vista D.

Autor: Carlos Zamudio

Uso:
    uv run python -m scripts.download_nrc
    uv run python scripts/download_nrc.py

Idempotente: si el archivo destino ya existe, no re-descarga.

El NRC EmoLex tiene licencia de investigación. Atribuir a:
    Saif M. Mohammad and Peter D. Turney (2013). Crowdsourcing a Word-Emotion
    Association Lexicon. Computational Intelligence, 29(3), 436-465.
    https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# URL oficial del NRC EmoLex (zip que contiene todos los idiomas)
_NRC_ZIP_URL = (
    "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
)

# Nombres canónicos de las 10 dimensiones del NRC EmoLex
_NRC_EMOTIONS = (
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "negative", "positive",
)

# Ruta de salida (formato largo: word\temotion\tvalue)
_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "lexicons" / "NRC-Emotion-Lexicon-v0.92-Spanish.txt"
)


def main() -> int:
    """Descarga y normaliza el lexicón NRC EmoLex Spanish."""
    if _OUTPUT_PATH.exists():
        print(f"[ok] Lexicón ya presente en {_OUTPUT_PATH}")
        return 0

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[..] Descargando NRC EmoLex desde {_NRC_ZIP_URL}")
    # El servidor del NRC rechaza requests sin User-Agent realista (HTTP 406).
    req = urllib.request.Request(
        _NRC_ZIP_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as exc:
        print(
            f"[err] No se pudo descargar: {exc}\n"
            f"      Descarga manualmente el archivo desde:\n"
            f"      https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm\n"
            f"      y colócalo en: {_OUTPUT_PATH}\n"
            f"      Formato esperado: TSV con columnas word\\temotion\\tvalue.",
            file=sys.stderr,
        )
        return 1

    print(f"[..] Descargados {len(zip_bytes) / 1024:.1f} KB; extrayendo...")
    spanish_lines = _extract_spanish(zip_bytes)
    if spanish_lines is None:
        print(
            "[err] No se encontró archivo en español dentro del zip. "
            "Inspecciona el zip manualmente.",
            file=sys.stderr,
        )
        return 1

    _OUTPUT_PATH.write_text("\n".join(spanish_lines), encoding="utf-8")
    print(f"[ok] Lexicón normalizado escrito en {_OUTPUT_PATH} ({len(spanish_lines)} filas)")
    return 0


def _extract_spanish(zip_bytes: bytes) -> list[str] | None:
    """Busca el archivo español dentro del zip y lo normaliza a formato largo.

    El zip oficial del NRC distribuye un archivo wide-format con columnas
    ``Spanish-Word``, ``English-Word`` y 10 columnas de emociones. Esta función
    detecta el archivo, lo lee, y lo convierte a long-format TSV.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        candidates = [
            name for name in zf.namelist()
            if "spanish" in name.lower() and name.endswith((".txt", ".csv", ".tsv"))
        ]
        if not candidates:
            return None
        # El archivo más probable: contiene "Spanish" en el nombre
        target = candidates[0]
        print(f"[..] Procesando: {target}")
        with zf.open(target) as f:
            content = f.read().decode("utf-8", errors="replace")

    return _convert_wide_to_long(content)


def _convert_wide_to_long(content: str) -> list[str]:
    """Convierte el TSV wide-format del NRC a long-format word/emotion/value."""
    lines = content.splitlines()
    if not lines:
        return []

    # Detectar header
    header = lines[0].split("\t")
    header_lower = [h.lower().strip() for h in header]

    # Identificar índice de columnas de emoción
    emo_indices: dict[str, int] = {}
    for emotion in _NRC_EMOTIONS:
        for i, col in enumerate(header_lower):
            if emotion in col:
                emo_indices[emotion] = i
                break

    # Si no encontramos columnas wide, asumir formato long y devolver como está
    if len(emo_indices) < 5:
        # Quizás ya está en formato long; devolverlo tal cual
        return lines

    # Identificar la columna de palabra en español (primera columna usualmente)
    word_col = 0
    for i, col in enumerate(header_lower):
        if "spanish" in col or "español" in col or "es" == col.strip():
            word_col = i
            break

    output: list[str] = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) <= max(emo_indices.values(), default=0):
            continue
        word = cells[word_col].strip().lower()
        if not word or word == "no translation":
            continue
        for emotion, idx in emo_indices.items():
            try:
                val = int(cells[idx].strip() or "0")
            except ValueError:
                val = 0
            output.append(f"{word}\t{emotion}\t{val}")
    return output


if __name__ == "__main__":
    sys.exit(main())
