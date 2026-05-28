"""registry.py — Catálogo de todos los métodos comparables.

Enumera, en un solo lugar, el baseline de Fase 2B y las tres variantes de
Fase 3. ``compare.py`` itera este registro para evaluar cada método disponible.

Cada entrada declara:
  * ``name``: identificador mostrado en la tabla comparativa.
  * ``predictions_file``: archivo de predicciones (dentro de ``output/``) en el
    esquema ``[text_id, predicted_label, probability_yes]``.
  * ``factory``: constructor de la variante (``AnorexiaClassifier``) para poder
    (re)generar sus predicciones con ``compare.py --train``. Es ``None`` para el
    baseline, que NUNCA se reentrena: permanece intacto y sólo se lee su archivo.

Importar este módulo no requiere torch: las variantes difieren sus imports
pesados hasta ``fit``/``predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from src.models.base import AnorexiaClassifier
from src.models.embeddings_svm import RoBERTuitoSVM
from src.models.finetuning import RoBERTuitoFineTuner
from src.models.zeroshot_nli import ZeroShotNLI


@dataclass(frozen=True)
class ModelEntry:
    """Una entrada del catálogo de métodos."""

    name: str
    predictions_file: str
    factory: Optional[Callable[[], AnorexiaClassifier]]
    description: str

    @property
    def is_trainable(self) -> bool:
        """``True`` si ``compare.py --train`` puede (re)generar sus predicciones."""
        return self.factory is not None


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        name="baseline_2b",
        predictions_file="predicciones_finales.csv",
        factory=None,  # baseline intacto: sólo se lee su archivo de Fase 2B.
        description="Fase 2B — TF-IDF/BoW multivista + clasificador clásico (baseline)",
    ),
    ModelEntry(
        name="robertuito_finetune",
        predictions_file="predicciones_robertuito_finetune.csv",
        factory=RoBERTuitoFineTuner,
        description="Fase 3a — fine-tuning de RoBERTuito",
    ),
    ModelEntry(
        name="robertuito_svm",
        predictions_file="predicciones_robertuito_svm.csv",
        factory=RoBERTuitoSVM,
        description="Fase 3b — embeddings RoBERTuito + SVM (pendiente)",
    ),
    ModelEntry(
        name="nli_zeroshot",
        predictions_file="predicciones_nli_zeroshot.csv",
        factory=ZeroShotNLI,
        description="Fase 3c — zero-shot con modelo NLI (pendiente)",
    ),
]
