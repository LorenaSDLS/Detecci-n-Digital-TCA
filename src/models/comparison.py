"""comparison.py — Lógica de comparación de métodos de Fase 3.

Lee las predicciones del baseline de Fase 2B y de cada variante de Fase 3,
calcula el AUC-ROC (métrica primaria del protocolo) de cada método disponible,
imprime una tabla comparativa y genera el análisis de falsos positivos /
falsos negativos (IDs y conteos) exigido por la rúbrica de Fase 3.

El baseline de Fase 2B NUNCA se reentrena: se consume su archivo de
predicciones tal cual (permanece intacto). Las variantes aún no implementadas y
los métodos sin archivo de predicciones se omiten limpiamente.

Este módulo contiene sólo la lógica; la interfaz de línea de comandos vive en
``src/models/__main__.py`` (``python -m src.models``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.models.data import load_dataset, load_gold
from src.models.registry import REGISTRY, ModelEntry


@dataclass
class ModelEvaluation:
    """Resultado de evaluar un método sobre la verdad-terreno."""

    name: str
    description: str
    auc: float
    f1: float
    n: int
    tp: int
    tn: int
    fp: int
    fn: int
    fp_ids: list = field(default_factory=list)
    fn_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_predictions(gold: pd.DataFrame, preds: pd.DataFrame) -> ModelEvaluation | None:
    """Une predicciones y verdad-terreno por ``text_id`` y calcula las métricas.

    Args:
        gold: DataFrame ``[text_id, label]``.
        preds: DataFrame ``[text_id, predicted_label, probability_yes]``.

    Returns:
        ``ModelEvaluation`` (sin nombre/descripción, que rellena el llamador), o
        ``None`` si no hay ningún ``text_id`` en común.
    """
    merged = gold.merge(preds, on="text_id", how="inner")
    if merged.empty:
        return None

    y_true = merged["label"].to_numpy()
    y_prob = merged["probability_yes"].to_numpy()
    y_pred = merged["predicted_label"].to_numpy()

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    return ModelEvaluation(
        name="",
        description="",
        auc=float(roc_auc_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        n=len(merged),
        tp=int(((y_pred == 1) & (y_true == 1)).sum()),
        tn=int(((y_pred == 0) & (y_true == 0)).sum()),
        fp=int(fp_mask.sum()),
        fn=int(fn_mask.sum()),
        fp_ids=merged.loc[fp_mask, "text_id"].tolist(),
        fn_ids=merged.loc[fn_mask, "text_id"].tolist(),
    )


def collect_evaluations(
    gold: pd.DataFrame,
    output_dir: Path,
    only: set[str] | None = None,
) -> list[ModelEvaluation]:
    """Evalúa todos los métodos del registro que tengan predicciones disponibles."""
    evaluations: list[ModelEvaluation] = []
    for entry in REGISTRY:
        if only is not None and entry.name not in only:
            continue
        pred_path = output_dir / entry.predictions_file
        if not pred_path.exists():
            print(f"  [{entry.name}] sin predicciones en {pred_path.name} — omitido")
            continue
        result = evaluate_predictions(gold, pd.read_csv(pred_path))
        if result is None:
            print(f"  [{entry.name}] sin text_id en común con la verdad-terreno — omitido")
            continue
        result.name = entry.name
        result.description = entry.description
        evaluations.append(result)
    evaluations.sort(key=lambda r: r.auc, reverse=True)
    return evaluations


# ---------------------------------------------------------------------------
# Entrenamiento de variantes
# ---------------------------------------------------------------------------

def train_one(entry: ModelEntry, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> bool:
    """(Re)genera las predicciones de una variante entrenable.

    Returns:
        ``True`` si se generaron predicciones; ``False`` si la variante está
        pendiente (``NotImplementedError``).
    """
    model = entry.factory()
    try:
        model.run(train_df, test_df, output_dir=output_dir)
        return True
    except NotImplementedError as exc:
        print(f"  [{entry.name}] pendiente — omitido ({exc})")
        return False


def train_models(
    selector: str,
    train_file: str | Path,
    test_file: str | Path,
    output_dir: str | Path,
) -> None:
    """Entrena la variante ``selector`` (un nombre del registro) o todas (``"all"``)."""
    output_dir = Path(output_dir)
    train_df = load_dataset(train_file)
    test_df = load_dataset(test_file)

    targets = [e for e in REGISTRY if e.is_trainable]
    if selector != "all":
        targets = [e for e in targets if e.name == selector]
        if not targets:
            raise ValueError(f"Variante entrenable desconocida: {selector!r}")

    for entry in targets:
        print(f"  [{entry.name}] entrenando…")
        train_one(entry, train_df, test_df, output_dir)


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------

def print_table(evaluations: list[ModelEvaluation]) -> None:
    """Imprime la tabla comparativa ordenada por AUC descendente."""
    print("\n" + "=" * 78)
    print("TABLA COMPARATIVA — AUC-ROC (primaria) · F1 (secundaria)")
    print("=" * 78)
    print(f"{'modelo':<22}{'AUC':>8}{'F1':>8}{'N':>6}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}")
    print("-" * 78)
    for r in evaluations:
        print(f"{r.name:<22}{r.auc:>8.4f}{r.f1:>8.4f}{r.n:>6}{r.tp:>6}{r.tn:>6}{r.fp:>6}{r.fn:>6}")
    print("-" * 78)
    if evaluations:
        best = evaluations[0]
        print(f"Mejor AUC: {best.name} ({best.auc:.4f})")


def write_comparison(evaluations: list[ModelEvaluation], output_dir: Path) -> Path:
    """Vuelca la tabla comparativa a ``output/comparativa_fase3.csv``."""
    rows = [
        {
            "modelo": r.name,
            "descripcion": r.description,
            "auc_roc": r.auc,
            "f1": r.f1,
            "n": r.n,
            "tp": r.tp,
            "tn": r.tn,
            "fp": r.fp,
            "fn": r.fn,
        }
        for r in evaluations
    ]
    path = output_dir / "comparativa_fase3.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nComparativa escrita en: {path}")
    return path


def write_error_analysis(evaluations: list[ModelEvaluation], output_dir: Path) -> Path:
    """Análisis de FP/FN por modelo: IDs y conteos (rúbrica de Fase 3)."""
    rows = []
    print("\n" + "=" * 78)
    print("ANÁLISIS DE ERRORES — Falsos Positivos / Falsos Negativos")
    print("=" * 78)
    for r in evaluations:
        print(f"\n[{r.name}]  FP={r.fp}  FN={r.fn}")
        if r.fp_ids:
            print(f"  Falsos positivos (sano clasificado como anorexia): {r.fp_ids}")
        if r.fn_ids:
            print(f"  Falsos negativos (anorexia no detectada): {r.fn_ids}")
        rows.extend({"modelo": r.name, "tipo_error": "FP", "text_id": tid} for tid in r.fp_ids)
        rows.extend({"modelo": r.name, "tipo_error": "FN", "text_id": tid} for tid in r.fn_ids)

    path = output_dir / "analisis_errores_fase3.csv"
    pd.DataFrame(rows, columns=["modelo", "tipo_error", "text_id"]).to_csv(path, index=False)
    print(f"\nAnálisis de errores escrito en: {path}")
    return path


def run_comparison(
    test_file: str | Path,
    output_dir: str | Path,
    only: set[str] | None = None,
) -> list[ModelEvaluation]:
    """Carga la verdad-terreno, evalúa los métodos disponibles y escribe reportes."""
    output_dir = Path(output_dir)
    gold = load_gold(test_file)
    print(f"Verdad-terreno: {len(gold)} muestras de {test_file}")

    print("\n--- Evaluando métodos disponibles ---")
    evaluations = collect_evaluations(gold, output_dir, only=only)
    if not evaluations:
        print("\nNo hay ningún método con predicciones evaluables.")
        return []

    print_table(evaluations)
    write_comparison(evaluations, output_dir)
    write_error_analysis(evaluations, output_dir)
    return evaluations
