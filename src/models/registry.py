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
from src.models.baseline_classic import ClassicBaseline
from src.models.embeddings_svm import RoBERTuitoEmbeddings
from src.models.finetuning import RoBERTuitoFineTuner
from src.models.llm_groq import GroqLLMClassifier
from src.models.zeroshot_nli import ZeroShotNLI


@dataclass(frozen=True)
class ModelEntry:
    """Una entrada del catálogo de métodos."""

    name: str
    predictions_file: str
    factory: Optional[Callable[[], AnorexiaClassifier]]
    description: str
    in_matrix: bool = False  # ¿participa en la matriz método×variante (5 métodos)?

    @property
    def is_trainable(self) -> bool:
        """``True`` si ``compare.py --train`` puede (re)generar sus predicciones."""
        return self.factory is not None


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        name="baseline_2b",
        predictions_file="predicciones_finales.csv",
        factory=None,  # baseline intacto: sólo se lee su archivo de Fase 2B.
        description="Fase 2B — TF-IDF/BoW multivista + clasificador clásico (baseline congelado)",
    ),
    ModelEntry(
        name="baseline_clasico",
        predictions_file="predicciones_baseline_clasico.csv",
        factory=ClassicBaseline,
        description="Baseline — pipeline multivista de Fase 2B (reentrenable por variante)",
        in_matrix=True,
    ),
    ModelEntry(
        name="robertuito_finetune",
        predictions_file="predicciones_robertuito_finetune.csv",
        factory=RoBERTuitoFineTuner,
        description="Fase 3a — fine-tuning de RoBERTuito",
        in_matrix=True,
    ),
    ModelEntry(
        name="robertuito_svm",
        predictions_file="predicciones_robertuito_svm.csv",
        factory=RoBERTuitoEmbeddings,  # head="svm" por defecto
        description="Fase 3b — embeddings RoBERTuito + SVM calibrado",
        in_matrix=True,
    ),
    ModelEntry(
        name="robertuito_logreg",
        predictions_file="predicciones_robertuito_logreg.csv",
        factory=lambda: RoBERTuitoEmbeddings(head="logreg"),
        description="Fase 3b — embeddings RoBERTuito + regresión logística",
        in_matrix=True,
    ),
    ModelEntry(
        name="robertuito_xgb",
        predictions_file="predicciones_robertuito_xgb.csv",
        factory=lambda: RoBERTuitoEmbeddings(head="xgb"),
        description="Fase 3b — embeddings RoBERTuito + XGBoost",
        in_matrix=True,
    ),
    ModelEntry(
        name="nli_zeroshot",
        predictions_file="predicciones_nli_zeroshot.csv",
        factory=ZeroShotNLI,
        description="Fase 3c — zero-shot con modelo NLI",
        in_matrix=True,
    ),
    ModelEntry(
        name="llm_groq",
        predictions_file="predicciones_llm_groq.csv",
        factory=GroqLLMClassifier,
        description="Fase 3d — LLM zero-shot vía Groq (modelo configurable, GROQ_MODEL)",
        in_matrix=True,
    ),
    ModelEntry(
        name="llm_groq_fewshot",
        predictions_file="predicciones_llm_groq_fewshot.csv",
        factory=lambda: GroqLLMClassifier(examples_per_class=4),
        description="Fase 3d — LLM few-shot vía Groq (8 ejemplos del train, 4 por clase)",
        in_matrix=True,
    ),
]
