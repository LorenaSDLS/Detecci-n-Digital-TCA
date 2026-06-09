"""zeroshot_nli.py — Variante (c) de Fase 3: clasificación zero-shot con NLI.

Clasifica cada texto SIN entrenamiento usando un modelo de Inferencia de Lenguaje
Natural (NLI) a través del pipeline ``zero-shot-classification`` de Hugging Face.
El modelo evalúa qué hipótesis (``"Este texto trata sobre {etiqueta}."``) entraña
mejor el texto y devuelve un score por etiqueta candidata.

Protocolo implementado (según el plan de la Fase 3):
  1. Pipeline ``zero-shot-classification`` con un modelo NLI en español
     (``Recognai/bert-base-spanish-wwm-cased-xnli`` por defecto).
  2. Hipótesis en español vía ``hypothesis_template`` con dos etiquetas
     candidatas mutuamente excluyentes (anorexia vs. bienestar normal).
  3. Preprocesamiento de tweets con ``pysentimiento.preprocessing.preprocess_tweet``
     antes de inferir (mismo preprocesado que las otras variantes de RoBERTuito).
  4. ``predict_proba`` = score (softmax sobre ambas etiquetas) que el modelo
     asigna a la etiqueta de anorexia.

Al ser zero-shot, ``fit`` no entrena nada (no-op); el pipeline se construye de
forma diferida en el primer ``predict_proba`` y se cachea en la instancia. Las
dependencias pesadas (torch, transformers, pysentimiento) se importan dentro de
los métodos para que importar el módulo no las requiera (mismo patrón que las
demás variantes). ``run`` se hereda intacto.
"""

from __future__ import annotations

import numpy as np

from src.models.base import AnorexiaClassifier


class ZeroShotNLI(AnorexiaClassifier):
    """Clasificación zero-shot mediante un modelo NLI.

    Attributes:
        model_name: Checkpoint NLI de Hugging Face usado por el pipeline.
        hypothesis_template: Plantilla de hipótesis; ``{}`` se sustituye por cada
            etiqueta candidata.
        positive_label: Etiqueta candidata que representa la clase positiva
            (anorexia); su score es la probabilidad devuelta.
        negative_label: Etiqueta candidata de la clase negativa (control).
        batch_size: Tamaño de lote para la inferencia del pipeline.
    """

    name = "nli_zeroshot"

    def __init__(
        self,
        model_name: str = "Recognai/bert-base-spanish-wwm-cased-xnli",
        hypothesis_template: str = "Este texto trata sobre {}.",
        positive_label: str = "anorexia o un trastorno alimenticio",
        negative_label: str = "bienestar normal",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.batch_size = batch_size

        # Pipeline construido de forma diferida en el primer predict_proba.
        self._classifier = None

    def fit(self, texts: list[str], labels: list[int]) -> "ZeroShotNLI":
        # Zero-shot: no requiere entrenamiento.
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) por texto como el score NLI de la etiqueta positiva."""
        classifier = self._get_classifier()
        proc_texts = self._preprocess(texts)

        results = classifier(
            proc_texts,
            candidate_labels=[self.positive_label, self.negative_label],
            hypothesis_template=self.hypothesis_template,
            multi_label=False,  # etiquetas mutuamente excluyentes → softmax entre ambas.
            batch_size=self.batch_size,
        )
        # Con una lista de entrada el pipeline devuelve una lista de dicts; con un
        # único texto devuelve un solo dict. Normalizamos a lista.
        if isinstance(results, dict):
            results = [results]

        probs = [self._positive_score(r) for r in results]
        return np.asarray(probs, dtype=float)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _get_classifier(self):
        """Construye (una sola vez) y cachea el pipeline zero-shot."""
        if self._classifier is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device,
            )
        return self._classifier

    @staticmethod
    def _preprocess(texts: list[str]) -> list[str]:
        """Aplica el preprocesamiento de tweets de RoBERTuito antes de inferir."""
        from pysentimiento.preprocessing import preprocess_tweet

        return [preprocess_tweet(t if isinstance(t, str) else "") for t in texts]

    def _positive_score(self, result: dict) -> float:
        """Extrae el score de la etiqueta positiva del resultado del pipeline.

        El pipeline devuelve ``labels`` ordenadas por score descendente y
        ``scores`` alineado; recuperamos el score correspondiente a
        ``positive_label`` por su posición en ``labels``.
        """
        labels = result["labels"]
        scores = result["scores"]
        idx = labels.index(self.positive_label)
        return float(scores[idx])
