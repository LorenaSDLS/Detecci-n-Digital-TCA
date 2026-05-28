"""embeddings_svm.py — Variante (b) de Fase 3: embeddings RoBERTuito + SVM.

PLACEHOLDER. Implementa la interfaz :class:`AnorexiaClassifier` pero aún no la
lógica; ``compare.py`` la detecta como "pendiente" y la omite limpiamente.

Plan de implementación (para completar después sin refactorizar nada):
  1. Cargar ``pysentimiento/robertuito-base-uncased`` CONGELADO (sin gradientes).
  2. Preprocesar cada texto con ``preprocess_tweet`` y tokenizar (mismo límite
     de longitud que la variante de fine-tuning).
  3. Extraer un embedding por texto (mean-pooling de la última capa oculta, o
     el token ``[CLS]``) en lotes con ``torch.no_grad()``.
  4. Entrenar un SVM sobre esos embeddings:
     ``Pipeline([StandardScaler(), CalibratedClassifierCV(LinearSVC())])`` para
     obtener probabilidades calibradas compatibles con el AUC.
  5. ``predict_proba`` reutiliza el mismo extractor de embeddings + el SVM.

Sólo hay que rellenar ``fit`` y ``predict_proba``; ``run`` se hereda intacto.
"""

from __future__ import annotations

import numpy as np

from src.models.base import AnorexiaClassifier

_PENDING_MSG = (
    "Variante (b) embeddings RoBERTuito + SVM aún no implementada (Fase 3, pendiente)."
)


class RoBERTuitoSVM(AnorexiaClassifier):
    """Embeddings de RoBERTuito congelado + clasificador SVM (pendiente)."""

    name = "robertuito_svm"

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        max_length: int = 128,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.seed = seed

    def fit(self, texts: list[str], labels: list[int]) -> "RoBERTuitoSVM":
        raise NotImplementedError(_PENDING_MSG)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(_PENDING_MSG)
