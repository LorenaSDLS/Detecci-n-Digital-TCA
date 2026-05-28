"""zeroshot_nli.py — Variante (c) de Fase 3: clasificación zero-shot con NLI.

PLACEHOLDER. Implementa la interfaz :class:`AnorexiaClassifier` pero aún no la
lógica; ``compare.py`` la detecta como "pendiente" y la omite limpiamente.

Plan de implementación (para completar después sin refactorizar nada):
  1. Cargar un pipeline ``zero-shot-classification`` de Hugging Face con un
     modelo NLI en español/multilingüe (p. ej.
     ``Recognai/bert-base-spanish-wwm-cased-xnli`` o un mNLI multilingüe).
  2. Definir las hipótesis candidatas en español, p. ej.
     ``hypothesis_template="Este texto trata sobre {}."`` con etiquetas
     ``["anorexia o un trastorno alimenticio", "bienestar normal"]``.
  3. Preprocesar el texto (``preprocess_tweet``) antes de inferir.
  4. ``predict_proba`` = score que el modelo asigna a la etiqueta de anorexia.

Al ser zero-shot, ``fit`` no entrena nada (no-op); sólo ``predict_proba`` queda
por implementar. ``run`` se hereda intacto.
"""

from __future__ import annotations

import numpy as np

from src.models.base import AnorexiaClassifier

_PENDING_MSG = "Variante (c) zero-shot NLI aún no implementada (Fase 3, pendiente)."


class ZeroShotNLI(AnorexiaClassifier):
    """Clasificación zero-shot mediante un modelo NLI (pendiente)."""

    name = "nli_zeroshot"

    def __init__(
        self,
        model_name: str = "Recognai/bert-base-spanish-wwm-cased-xnli",
        hypothesis_template: str = "Este texto trata sobre {}.",
        positive_label: str = "anorexia o un trastorno alimenticio",
        negative_label: str = "bienestar normal",
    ) -> None:
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template
        self.positive_label = positive_label
        self.negative_label = negative_label

    def fit(self, texts: list[str], labels: list[int]) -> "ZeroShotNLI":
        # Zero-shot: no requiere entrenamiento.
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(_PENDING_MSG)
