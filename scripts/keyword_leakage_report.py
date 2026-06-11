"""keyword_leakage_report.py — Cuantifica la fuga léxica por token (Fase 3).

ARCHIVO 1/2 de la prueba de hipótesis: ¿la AUC alta del fine-tuning de
RoBERTuito proviene de comprensión lingüística o de pistas léxicas obvias?

Este script NO entrena nada: sólo mide. Sobre el conjunto de ENTRENAMIENTO
calcula, token por token (palabras y hashtags), cuánto discrimina cada uno entre
``anorexia`` y ``control`` mediante el log-odds ponderado de su frecuencia de
documento por clase. Si los hashtags/términos diagnósticos encabezan el ranking,
la fuga léxica queda demostrada (y se cuantifica de forma reproducible).

La pareja de este script es ``keyword_baseline.py`` (la demostración: un modelo
trivial sólo-léxico que se acerca a la cifra del transformer).

Uso:
    .venv/bin/python -m scripts.keyword_leakage_report
    .venv/bin/python -m scripts.keyword_leakage_report --train-file data/data_train.csv
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from src.models.data import load_dataset

# Token = hashtag completo (``#palabra``) o palabra suelta. re.UNICODE incluye
# las vocales acentuadas del español.
_HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Marcadores pro-ana que la auditoría señaló como presuntas fugas. Los listamos
# explícitamente para verificar sus conteos REALES, no estimados.
_AUDIT_MARKERS = (
    "#thinspo", "#thinspiration", "#thin", "#hastaloshuesos",
    "anorexia", "hambre", "ayuno", "ana", "huesos", "delgada",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    df = load_dataset(args.train_file)

    n_ano = int((df["label"] == 1).sum())
    n_con = int((df["label"] == 0).sum())
    print("=" * 78)
    print("REPORTE DE FUGA LÉXICA — frecuencia de documento por clase")
    print("=" * 78)
    print(f"Entrenamiento: {len(df)} textos · anorexia={n_ano} · control={n_con}\n")

    # Frecuencia de DOCUMENTO por clase: en cuántos textos de cada clase aparece
    # el token al menos una vez (presencia, no recuento bruto).
    df_ano: Counter[str] = Counter()
    df_con: Counter[str] = Counter()
    for text, label in zip(df["text"], df["label"]):
        tokens = _tokenize(text)
        target = df_ano if label == 1 else df_con
        for tok in tokens:
            target[tok] += 1

    ranked = _rank_by_log_odds(df_ano, df_con, n_ano, n_con, min_docs=args.min_docs)

    _print_table("TOP tokens hacia ANOREXIA", ranked[: args.top], n_ano, n_con)
    _print_table("TOP tokens hacia CONTROL", list(reversed(ranked))[: args.top // 2],
                 n_ano, n_con)

    _print_hashtag_summary(df, n_ano, n_con)
    _print_audit_verification(df_ano, df_con)
    return 0


def _tokenize(text: str) -> set[str]:
    """Conjunto de tokens en minúscula: hashtags completos + palabras sueltas."""
    if not isinstance(text, str):
        return set()
    low = text.lower()
    hashtags = _HASHTAG_RE.findall(low)
    # Quitar los hashtags antes de extraer palabras para no contar ``ayuno`` dos
    # veces a partir de ``#ayuno``.
    words = _WORD_RE.findall(_HASHTAG_RE.sub(" ", low))
    return set(hashtags) | {w for w in words if len(w) >= 3}


def _rank_by_log_odds(
    df_ano: Counter[str],
    df_con: Counter[str],
    n_ano: int,
    n_con: int,
    min_docs: int,
) -> list[tuple[str, float, int, int]]:
    """Ordena tokens por log-odds ponderado (suavizado de Laplace)."""
    vocab = set(df_ano) | set(df_con)
    scored: list[tuple[str, float, int, int]] = []
    for tok in vocab:
        a, c = df_ano[tok], df_con[tok]
        if a + c < min_docs:
            continue
        # log-odds de presencia: anorexia vs. control, con +0.5 de suavizado.
        odds = math.log((a + 0.5) / (n_ano - a + 0.5)) - math.log(
            (c + 0.5) / (n_con - c + 0.5)
        )
        scored.append((tok, odds, a, c))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _print_table(
    title: str,
    rows: list[tuple[str, float, int, int]],
    n_ano: int,
    n_con: int,
) -> None:
    print("-" * 78)
    print(title)
    print("-" * 78)
    print(f"{'token':<24}{'log-odds':>10}{'#anorexia':>12}{'#control':>11}{'lift':>9}")
    for tok, odds, a, c in rows:
        # lift = (prevalencia en anorexia) / (prevalencia en control), acotado.
        p_ano = a / n_ano
        p_con = c / n_con if c else 0.0
        lift = (p_ano / p_con) if p_con else float("inf")
        lift_s = "∞" if lift == float("inf") else f"{lift:>8.1f}"
        print(f"{tok:<24}{odds:>10.2f}{a:>12}{c:>11}{lift_s:>9}")
    print()


def _print_hashtag_summary(df, n_ano: int, n_con: int) -> None:
    """Cuántos textos de cada clase contienen al menos un hashtag."""
    has_ht_ano = has_ht_con = 0
    for text, label in zip(df["text"], df["label"]):
        has = bool(_HASHTAG_RE.search(text)) if isinstance(text, str) else False
        if label == 1:
            has_ht_ano += has
        else:
            has_ht_con += has
    print("-" * 78)
    print("PRESENCIA DE HASHTAGS POR CLASE")
    print("-" * 78)
    print(f"  anorexia con ≥1 hashtag: {has_ht_ano:>4}/{n_ano}  ({has_ht_ano / n_ano:.1%})")
    print(f"  control  con ≥1 hashtag: {has_ht_con:>4}/{n_con}  ({has_ht_con / n_con:.1%})\n")


def _print_audit_verification(
    df_ano: Counter[str],
    df_con: Counter[str],
) -> None:
    """Verifica los conteos REALES de los marcadores citados por la auditoría."""
    print("-" * 78)
    print("VERIFICACIÓN DE MARCADORES CITADOS EN LA AUDITORÍA")
    print("-" * 78)
    print(f"{'marcador':<20}{'#anorexia':>12}{'#control':>11}{'exclusividad':>14}")
    for m in _AUDIT_MARKERS:
        a, c = df_ano.get(m, 0), df_con.get(m, 0)
        excl = (a / (a + c)) if (a + c) else 0.0
        print(f"{m:<20}{a:>12}{c:>11}{excl:>13.0%}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--train-file", default="data/data_train.csv",
                   help="CSV/XLSX de entrenamiento con etiquetas.")
    p.add_argument("--top", type=int, default=30,
                   help="Cuántos tokens mostrar por tabla.")
    p.add_argument("--min-docs", type=int, default=5,
                   help="Mínimo de documentos (ambas clases) para considerar un token.")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
