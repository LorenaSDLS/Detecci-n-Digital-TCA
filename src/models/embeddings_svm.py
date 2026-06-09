"""embeddings_svm.py — Variante (b) de Fase 3: embeddings RoBERTuito + SVM.

Esta variante usa RoBERTuito como extractor de embeddings congelado y entrena
un SVM lineal sobre esos vectores. A diferencia del fine-tuning, los pesos de
RoBERTuito no se actualizan; sólo aprende el clasificador SVM.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.models.base import AnorexiaClassifier


class RoBERTuitoSVM(AnorexiaClassifier):
    """Embeddings de RoBERTuito congelado + clasificador SVM."""

    name = "robertuito_svm"

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed

        # Estos atributos se llenan durante fit().
        self._tokenizer = None
        self._model = None
        self._svm = None
        self._device = None
        self._effective_max_length: int | None = None

    # ------------------------------------------------------------------
    # API obligatoria de AnorexiaClassifier
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[int]) -> "RoBERTuitoSVM":
        """Extrae embeddings de los textos de entrenamiento y entrena el SVM."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        if len(texts) != len(labels):
            raise ValueError("texts y labels deben tener la misma longitud.")

        self._load_transformer()

        embeddings = self._extract_embeddings(texts)
        y = np.asarray(labels, dtype=int)

        # CalibratedClassifierCV necesita que cada clase tenga ejemplos
        # suficientes para hacer validación cruzada. Usamos hasta 5 folds,
        # pero bajamos el número si la clase minoritaria tiene menos ejemplos.
        class_counts = Counter(y.tolist())
        if len(class_counts) < 2:
            raise ValueError("El entrenamiento necesita ejemplos de ambas clases: 0 y 1.")

        cv = min(5, min(class_counts.values()))
        if cv < 2:
            raise ValueError("Cada clase necesita al menos 2 ejemplos para calibrar el SVM.")

        base_svm = LinearSVC(
            class_weight="balanced",
            random_state=self.seed,
            max_iter=10_000,
            dual="auto",
        )

        self._svm = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(estimator=base_svm, cv=cv)),
        ])

        self._svm.fit(embeddings, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) para cada texto usando RoBERTuito + SVM."""
        if self._svm is None:
            raise RuntimeError("El modelo no ha sido entrenado: llama a fit() primero.")

        embeddings = self._extract_embeddings(texts)
        probabilities = self._svm.predict_proba(embeddings)

        # Columna 1 = probabilidad de la clase positiva.
        # En este proyecto: 1 = anorexia / posible TCA.
        return probabilities[:, 1]

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _load_transformer(self) -> None:
        """Carga tokenizer y modelo base de RoBERTuito una sola vez."""
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer, set_seed

        set_seed(self.seed)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)

        # Congelar RoBERTuito: no calculamos gradientes ni actualizamos pesos.
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False

        self._effective_max_length = self._resolve_max_length(self._tokenizer)

    @staticmethod
    def _preprocess(texts: list[str]) -> list[str]:
        """Aplica el preprocesamiento recomendado para RoBERTuito."""
        from pysentimiento.preprocessing import preprocess_tweet

        return [preprocess_tweet(text if isinstance(text, str) else "") for text in texts]

    def _extract_embeddings(self, texts: list[str]) -> np.ndarray:
        """Convierte una lista de textos en una matriz de embeddings.

        Resultado:
            np.ndarray de forma (n_textos, hidden_size). Para RoBERTuito base,
            normalmente hidden_size = 768.
        """
        self._load_transformer()

        import torch

        processed_texts = self._preprocess(texts)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(processed_texts), self.batch_size):
            batch_texts = processed_texts[start:start + self.batch_size]

            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self._effective_max_length,
                return_tensors="pt",
            )

            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)
                batch_embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )

            all_embeddings.append(batch_embeddings.cpu().numpy())

        if not all_embeddings:
            hidden_size = int(getattr(self._model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        """Promedia los embeddings de tokens reales, ignorando el padding."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _resolve_max_length(self, tokenizer) -> int:
        """Usa el menor valor entre max_length y el límite real del tokenizer."""
        model_max = getattr(tokenizer, "model_max_length", None)

        if not model_max or model_max > 100_000:
            model_max = self.max_length

        return min(self.max_length, int(model_max))