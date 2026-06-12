"""length_baseline.py — Baseline ingenuo de longitud (Fase 3).

El "modelo" más tonto posible: ``score = -len(texto)`` (cuanto más corto, más
probable anorexia). No lee ni una palabra; sólo cuenta caracteres.

Sirve como piso de referencia para la tabla comparativa: si la pura longitud ya
separa las clases (en este corpus, AUC ≈ 0.8), esa parte del desempeño de
cualquier modelo "viene gratis" de la construcción del dataset (confesiones
cortas vs. recetas/promos largas), no de comprender el lenguaje del TCA.

Reporta, para los datos originales y para su versión sin hashtags (calculada en
memoria, sin archivos derivados):
  * estadísticas de longitud por clase,
  * AUC-ROC de la longitud como score,
  * una regla dura «anorexia si len < umbral», con el umbral elegido sobre el
    TRAIN (elegirlo sobre el test inflaría la cifra) y evaluado sobre el test.

Uso:
    .venv/bin/python -m scripts.length_baseline
    .venv/bin/python -m scripts.length_baseline --test-file data/data_test_fold1.csv
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.models.data import load_dataset, strip_hashtags_text


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train = load_dataset(args.train_file)
    test = load_dataset(args.test_file)

    print("=" * 70)
    print("BASELINE INGENUO — longitud del texto como único atributo")
    print("=" * 70)
    print(f"train={args.train_file} ({len(train)})  test={args.test_file} ({len(test)})")

    for label, transform in [
        ("CON hashtags", lambda t: t),
        ("SIN hashtags", strip_hashtags_text),
    ]:
        tr_lens = np.array([len(transform(t)) for t in train["text"]])
        te_lens = np.array([len(transform(t)) for t in test["text"]])
        _report(label, tr_lens, train["label"].to_numpy(),
                te_lens, test["label"].to_numpy())
    return 0


def _report(label: str, tr_lens, y_train, te_lens, y_test) -> None:
    print("\n" + "-" * 70)
    print(f"[{label}]")
    print("-" * 70)

    for cls, name in [(1, "anorexia"), (0, "control ")]:
        lens = te_lens[y_test == cls]
        print(f"  {name} (test): media {lens.mean():>6.0f} chars · "
              f"mediana {np.median(lens):>5.0f} · "
              f"p25-p75: {np.percentile(lens, 25):.0f}-{np.percentile(lens, 75):.0f}")

    # AUC: score continuo = -longitud (más corto → más probable anorexia).
    auc = roc_auc_score(y_test, -te_lens)
    print(f"\n  AUC-ROC (score = -longitud): {auc:.4f}")

    # Regla dura: umbral elegido por F1 sobre el TRAIN, evaluado sobre el test.
    candidates = np.unique(tr_lens)
    threshold = max(candidates, key=lambda u: f1_score(y_train, (tr_lens < u).astype(int)))
    pred = (te_lens < threshold).astype(int)
    acc = float((pred == y_test).mean())
    print(f"  Regla «anorexia si < {threshold} chars» (umbral del train): "
          f"accuracy {acc:.1%} · F1 {f1_score(y_test, pred):.4f}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--train-file", default="data/data_train.csv",
                   help="CSV/XLSX de entrenamiento (sólo para elegir el umbral).")
    p.add_argument("--test-file", default="data/data_test_fold2.csv",
                   help="CSV/XLSX de prueba con etiquetas.")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
