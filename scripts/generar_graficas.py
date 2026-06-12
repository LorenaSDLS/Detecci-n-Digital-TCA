"""generar_graficas.py — Gráficas y métricas del reporte de Fase 3.

Genera, a partir de las predicciones de la matriz método×variante (no entrena
ni llama a ninguna API), los artefactos visuales que pide el reporte:

  * ``roc_<variante>.png``        — curvas ROC de todos los métodos (sección 8.6).
  * ``auc_comparacion.png``       — barras de AUC por método, con vs sin hashtags (9.1).
  * ``delta_hashtags.png``        — caída de AUC al quitar hashtags por método (9.1).
  * ``metricas_fase2_vs_fase3.png`` — AUC/Accuracy/Precision/Recall/F1 del baseline
                                      de Fase 2 frente a los mejores de Fase 3 (8.4/8.5).
  * ``matrices_confusion.png``    — matrices de confusión Fase 2 vs Fase 3 (8.6).
  * ``metricas_completas.csv``    — tabla método×variante con todas las métricas,
                                    para llenar las tablas 8.4, 8.5 y 9.1.

Uso:
    .venv/bin/python -m scripts.generar_graficas
    .venv/bin/python -m scripts.generar_graficas --matrix-dir output/matriz/data_test_fold1 \\
        --test-file data/data_test_fold1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: sólo archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.data import load_dataset

# Nombre del método → etiqueta legible en las figuras (y orden de despliegue).
_DISPLAY = {
    "baseline_clasico": "Baseline Fase 2B",
    "robertuito_finetune": "RoBERTuito fine-tune",
    "robertuito_svm": "Embeddings + SVM",
    "robertuito_logreg": "Embeddings + LogReg",
    "robertuito_xgb": "Embeddings + XGBoost",
    "nli_zeroshot": "NLI zero-shot",
    "llm_groq": "LLM zero-shot",
    "llm_groq_fewshot": "LLM few-shot",
}

_VARIANT_LABEL = {"con_hashtags": "Con hashtags", "sin_hashtags": "Sin hashtags"}

# Métodos destacados para la comparación Fase 2 vs Fase 3 (8.4/8.5).
_HIGHLIGHT = ["baseline_clasico", "robertuito_finetune", "llm_groq_fewshot"]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_dataset(args.test_file)[["text_id", "label"]]
    rows = _collect(matrix_dir, gold)
    if not rows:
        print(f"No se encontraron predicciones en {matrix_dir}/*/predicciones_*.csv")
        return 1
    metrics = pd.DataFrame(rows)

    csv_path = out_dir / "metricas_completas.csv"
    metrics.drop(columns=["fpr", "tpr", "y_true", "y_pred"]).to_csv(csv_path, index=False)
    print(f"[ok] {csv_path}")

    for variant in metrics["variante"].unique():
        _plot_roc(metrics[metrics["variante"] == variant], variant, out_dir)
    _plot_auc_bars(metrics, out_dir)
    _plot_delta(metrics, out_dir)
    _plot_phase_comparison(metrics, out_dir)
    _plot_confusions(metrics, out_dir)

    print(f"\nListo: gráficas en {out_dir}/")
    return 0


def _collect(matrix_dir: Path, gold: pd.DataFrame) -> list[dict]:
    """Evalúa cada predicciones_*.csv de cada variante contra la verdad-terreno."""
    rows: list[dict] = []
    for variant_dir in sorted(matrix_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        for pred_path in sorted(variant_dir.glob("predicciones_*.csv")):
            method = pred_path.stem.removeprefix("predicciones_")
            merged = gold.merge(pd.read_csv(pred_path), on="text_id", how="inner")
            if merged.empty:
                continue
            y_true = merged["label"].to_numpy()
            y_prob = merged["probability_yes"].to_numpy()
            y_pred = merged["predicted_label"].to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            rows.append({
                "metodo": method,
                "variante": variant_dir.name,
                "auc": roc_auc_score(y_true, y_prob),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "n": len(merged),
                "fpr": fpr,
                "tpr": tpr,
                "y_true": y_true,
                "y_pred": y_pred,
            })
    return rows


def _order(df: pd.DataFrame) -> list[str]:
    """Métodos presentes, en el orden del catálogo de despliegue."""
    present = set(df["metodo"])
    return [m for m in _DISPLAY if m in present]


def _plot_roc(df: pd.DataFrame, variant: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for method in _order(df):
        row = df[df["metodo"] == method].iloc[0]
        ax.plot(row["fpr"], row["tpr"], lw=1.8,
                label=f"{_DISPLAY[method]} (AUC={row['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Azar (AUC=0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC — {_VARIANT_LABEL.get(variant, variant)}")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"roc_{variant}.png")


def _plot_auc_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    pivot = metrics.pivot(index="metodo", columns="variante", values="auc")
    pivot = pivot.loc[_order(metrics)].sort_values("con_hashtags", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(pivot))
    ax.barh(y + 0.2, pivot["con_hashtags"], height=0.4, label="Con hashtags")
    ax.barh(y - 0.2, pivot["sin_hashtags"], height=0.4, label="Sin hashtags")
    for i, (con, sin) in enumerate(zip(pivot["con_hashtags"], pivot["sin_hashtags"])):
        ax.text(con + 0.005, i + 0.2, f"{con:.3f}", va="center", fontsize=8)
        ax.text(sin + 0.005, i - 0.2, f"{sin:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, [_DISPLAY[m] for m in pivot.index])
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(0.5, 1.05)
    ax.set_title("AUC-ROC por método y variante de datos")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir / "auc_comparacion.png")


def _plot_delta(metrics: pd.DataFrame, out_dir: Path) -> None:
    pivot = metrics.pivot(index="metodo", columns="variante", values="auc")
    delta = (pivot["sin_hashtags"] - pivot["con_hashtags"]).loc[_order(metrics)]
    delta = delta.sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.arange(len(delta)), delta, color="#c0504d")
    for i, d in enumerate(delta):
        ax.text(d - 0.002, i, f"{d:+.3f}", va="center", ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(delta)), [_DISPLAY[m] for m in delta.index])
    ax.set_xlabel("Δ AUC (sin hashtags − con hashtags)")
    ax.set_title("Dependencia de los hashtags: caída de AUC al eliminarlos")
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir / "delta_hashtags.png")


def _plot_phase_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    """AUC/Accuracy/Precision/Recall/F1 del baseline (Fase 2) vs Fase 3 (con hashtags)."""
    df = metrics[(metrics["variante"] == "con_hashtags")
                 & (metrics["metodo"].isin(_HIGHLIGHT))]
    if df.empty:
        return
    labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    cols = ["auc", "accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.8 / len(_HIGHLIGHT)
    for i, method in enumerate(m for m in _HIGHLIGHT if m in set(df["metodo"])):
        vals = df[df["metodo"] == method][cols].iloc[0].to_numpy(dtype=float)
        offset = (i - (len(_HIGHLIGHT) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=_DISPLAY[method])
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Fase 2 (baseline) vs Fase 3 — conjunto de prueba, con hashtags")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / "metricas_fase2_vs_fase3.png")


def _plot_confusions(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Matrices de confusión del baseline (Fase 2) y del mejor método de Fase 3."""
    df = metrics[metrics["variante"] == "con_hashtags"]
    best_f3 = (df[df["metodo"] != "baseline_clasico"]
               .sort_values("auc", ascending=False).iloc[0]["metodo"])
    targets = [m for m in ("baseline_clasico", best_f3) if m in set(df["metodo"])]

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.6))
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, targets):
        row = df[df["metodo"] == method].iloc[0]
        cm = confusion_matrix(row["y_true"], row["y_pred"])
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        fontsize=14, color=color)
        ax.set_xticks([0, 1], ["control", "anorexia"])
        ax.set_yticks([0, 1], ["control", "anorexia"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_title(f"{_DISPLAY[method]}\n(AUC={row['auc']:.3f} · F1={row['f1']:.3f})")
    fig.suptitle("Matrices de confusión — con hashtags", y=1.0)
    _save(fig, out_dir / "matrices_confusion.png")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--matrix-dir", default="output/matriz/data_test_fold2",
                   help="Directorio de la matriz con subdirectorios por variante.")
    p.add_argument("--test-file", default="data/data_test_fold2.csv",
                   help="CSV de prueba con etiquetas (verdad-terreno).")
    p.add_argument("--output-dir", default="output/graficas",
                   help="Directorio donde escribir las gráficas.")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
