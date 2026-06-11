"""keyword_baseline.py — Baselines triviales sólo-léxico (Fase 3).

ARCHIVO 2/2 de la prueba de hipótesis. Mientras ``keyword_leakage_report.py``
*describe* qué tokens filtran la etiqueta, este script lo *demuestra*: entrena
clasificadores deliberadamente TONTOS —sin comprender el texto, sólo contando
palabras— sobre ``data_train.csv`` y los evalúa sobre ``data_test_fold1.csv``.

Si un modelo así se acerca a la AUC del fine-tuning de RoBERTuito, entonces el
transformer no está "entendiendo" anorexia: explota las mismas pistas léxicas
superficiales, y la cifra alta no es defendible.

Tres baselines, de más interpretable a más potente:
  1. presencia de marcadores diagnósticos curados  (conteo → score continuo)
  2. regresión logística sobre bolsa-de-palabras SÓLO de hashtags
  3. regresión logística sobre bolsa-de-palabras completa  (techo léxico)

Todo se ajusta SÓLO con el train (los baselines 2 y 3 nunca ven el test al
entrenar), así que la comparación es justa frente al transformer.

Uso:
    .venv/bin/python -m scripts.keyword_baseline
    .venv/bin/python -m scripts.keyword_baseline --train-file ... --test-file ...
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.models.data import load_dataset

# Referencia: AUC del fine-tuning de RoBERTuito reportada en comparativa_fase3.
_TRANSFORMER_AUC = 0.9817
_TRANSFORMER_F1 = 0.9281

# Token = hashtag completo o palabra; ``#?`` mantiene los hashtags como tokens
# propios (el patrón por defecto de CountVectorizer descarta el ``#``).
_TOKEN_PATTERN = r"(?u)#?[^\W\d_]+"
_HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)

# Marcadores pro-ana (hashtags y términos). Coinciden como subcadena dentro de
# hashtags y como palabra suelta en el texto plano.
_MARKERS = (
    "thinspo", "thinspiration", "thinspire", "hastaloshuesos", "huesos",
    "proana", "promia", "anamia", "thin", "skinny", "bonespo", "ayuno",
    "anorexia", "bulimia", "delgada", "gorda",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train = load_dataset(args.train_file)
    test = load_dataset(args.test_file)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    print("=" * 70)
    print("BASELINES SÓLO-LÉXICO vs. fine-tuning de RoBERTuito")
    print("=" * 70)
    print(f"train={len(train)}  test={len(test)}\n")

    results: list[tuple[str, float, float]] = []
    results.append(_baseline_markers(train["text"], test["text"], y_test))
    results.append(_baseline_logreg(
        "LogReg · SÓLO hashtags",
        _hashtags_only(train["text"]), y_train,
        _hashtags_only(test["text"]), y_test,
    ))
    results.append(_baseline_logreg(
        "LogReg · bolsa-de-palabras",
        train["text"].tolist(), y_train,
        test["text"].tolist(), y_test,
    ))

    _print_table(results)
    return 0


def _baseline_markers(train_text, test_text, y_test) -> tuple[str, float, float]:
    """Baseline 1: score = nº de marcadores diagnósticos presentes en el texto.

    No usa el train para nada (la lista es fija): es el detector más ingenuo
    imaginable. Si aún así separa las clases, la señal es puramente superficial.
    """
    del train_text  # lista fija, no se aprende nada
    scores = np.array([_marker_count(t) for t in test_text], dtype=float)
    auc = roc_auc_score(y_test, scores)
    # Etiqueta dura: hay al menos un marcador presente.
    f1 = f1_score(y_test, (scores > 0).astype(int))
    return ("presencia de marcadores", auc, f1)


def _baseline_logreg(
    name: str,
    train_text: list[str],
    y_train,
    test_text: list[str],
    y_test,
) -> tuple[str, float, float]:
    """Baselines 2 y 3: bolsa-de-palabras + regresión logística (ajuste sólo-train)."""
    vec = CountVectorizer(token_pattern=_TOKEN_PATTERN, min_df=2)
    x_train = vec.fit_transform(train_text)
    x_test = vec.transform(test_text)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, (proba >= 0.5).astype(int))
    return (f"{name}  (|vocab|={len(vec.vocabulary_)})", auc, f1)


def _marker_count(text: str) -> int:
    """Cuántos marcadores pro-ana aparecen (en hashtags o como palabra)."""
    if not isinstance(text, str):
        return 0
    low = text.lower()
    return sum(1 for m in _MARKERS if m in low)


def _hashtags_only(texts) -> list[str]:
    """Reduce cada texto a la cadena de sus hashtags (vacía si no tiene)."""
    return [" ".join(_HASHTAG_RE.findall(t.lower())) if isinstance(t, str) else ""
            for t in texts]


def _print_table(results: list[tuple[str, float, float]]) -> None:
    print("-" * 70)
    print(f"{'baseline':<40}{'AUC':>8}{'F1':>8}{'Δ AUC':>10}")
    print("-" * 70)
    for name, auc, f1 in results:
        delta = auc - _TRANSFORMER_AUC
        print(f"{name:<40}{auc:>8.4f}{f1:>8.4f}{delta:>+10.4f}")
    print("-" * 70)
    print(f"{'RoBERTuito fine-tune (referencia)':<40}"
          f"{_TRANSFORMER_AUC:>8.4f}{_TRANSFORMER_F1:>8.4f}{0.0:>+10.4f}")
    print("-" * 70)
    best = max(results, key=lambda r: r[1])
    gap = _TRANSFORMER_AUC - best[1]
    print(
        f"\nMejor baseline tonto: «{best[0].split('  ')[0]}» con AUC {best[1]:.4f}.\n"
        f"El transformer sólo lo supera por {gap:+.4f} de AUC.\n"
        + ("→ FUGA LÉXICA CONFIRMADA: un modelo sin comprensión casi iguala "
           "al transformer.\n" if gap < 0.10 else
           "→ El transformer aporta señal MÁS ALLÁ del léxico superficial.\n")
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--train-file", default="data/data_train.csv",
                   help="CSV/XLSX de entrenamiento con etiquetas.")
    p.add_argument("--test-file", default="data/data_test_fold1.csv",
                   help="CSV/XLSX de prueba con etiquetas (verdad-terreno).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
