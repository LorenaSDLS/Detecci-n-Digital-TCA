"""__main__.py — Orquestador de Fase 3 (``python -m src.models``).

Punto de entrada único para entrenar las variantes de Fase 3 y compararlas
contra el baseline de Fase 2B por AUC-ROC. El baseline NUNCA se reentrena: sólo
se lee su archivo de predicciones (permanece intacto).

Subcomandos:

    # Comparar las predicciones ya existentes (baseline + variantes generadas):
    python -m src.models compare

    # Entrenar TODAS las variantes implementadas y comparar:
    python -m src.models train --model all

    # Probar UNA sola variante (la entrena, predice y compara):
    python -m src.models train --model robertuito_finetune

Sin subcomando equivale a ``compare``.
"""

from __future__ import annotations

import argparse

from src.models.comparison import run_comparison, train_models
from src.models.registry import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.models",
        description="Entrena y compara los métodos de Fase 3 por AUC-ROC.",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--test-file", default="data/data_test_fold1.csv",
                        help="CSV/XLSX de prueba con etiquetas (verdad-terreno).")
    common.add_argument("--output-dir", default="output",
                        help="Directorio con/para los archivos de predicciones.")

    p_compare = sub.add_parser("compare", parents=[common],
                               help="Evalúa las predicciones existentes.")
    p_compare.add_argument("--only", nargs="+", metavar="MODELO",
                           help="Limita la comparación a estos nombres del registro.")

    trainable = [e.name for e in REGISTRY if e.is_trainable]
    p_train = sub.add_parser("train", parents=[common],
                             help="(Re)genera predicciones de una variante o todas, y compara.")
    p_train.add_argument("--model", default="all", choices=["all", *trainable],
                         help="Variante a entrenar; 'all' entrena todas las implementadas.")
    p_train.add_argument("--train-file", default="data/data_train.csv",
                         help="CSV/XLSX de entrenamiento.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = args.command or "compare"

    if command == "train":
        print("--- Generando predicciones de variantes ---")
        train_models(args.model, args.train_file, args.test_file, args.output_dir)
        run_comparison(args.test_file, args.output_dir, only=None)
    else:
        only = set(args.only) if getattr(args, "only", None) else None
        run_comparison(args.test_file, args.output_dir, only=only)


if __name__ == "__main__":
    main()
