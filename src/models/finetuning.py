"""finetuning.py — Variante (a) de Fase 3: fine-tuning de RoBERTuito.

Ajusta ``pysentimiento/robertuito-base-uncased`` para clasificación binaria
(anorexia vs. control) con la API ``Trainer`` de Hugging Face.

Puntos clave del protocolo:
  * RoBERTuito requiere preprocesamiento ESPECÍFICO antes del tokenizer:
    ``pysentimiento.preprocessing.preprocess_tweet`` (no es opcional).
  * ``max_length`` se acota al límite real del tokenizer de RoBERTuito.
  * Partición train/validación 80/20 ESTRATIFICADA (clases desbalanceadas).
  * Early stopping basado en el AUC-ROC de validación (métrica primaria).
  * Semilla fija para reproducibilidad.

Las dependencias pesadas (torch, transformers, datasets, pysentimiento) se
importan de forma diferida dentro de los métodos para que importar el módulo
no las requiera (mismo patrón que la carga diferida de spaCy en el baseline).

Se entrena/compara desde el orquestador ``python -m src.models train
--model robertuito_finetune`` (ver ``src/models/__main__.py``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from src.models.base import AnorexiaClassifier
from src.models.data import stratified_split

# Límite real de posiciones de RoBERTuito-base (128 tokens efectivos).
_ROBERTUITO_MAX_LENGTH = 128


class RoBERTuitoFineTuner(AnorexiaClassifier):
    """Fine-tuning supervisado de RoBERTuito para detección de anorexia.

    Attributes:
        model_name: Checkpoint base de Hugging Face.
        lr: Learning rate.
        batch_size: Tamaño de lote por dispositivo.
        epochs: Número máximo de épocas (puede recortarse por early stopping).
        max_length: Longitud máxima de tokens solicitada (se acota al límite
            real del tokenizer).
        val_size: Proporción de validación en la partición estratificada.
        seed: Semilla aleatoria.
        patience: Paciencia (en evaluaciones) del early stopping sobre AUC.
        work_dir: Directorio para checkpoints intermedios del ``Trainer``.
    """

    name = "robertuito_finetune"

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        lr: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 4,
        max_length: int = _ROBERTUITO_MAX_LENGTH,
        val_size: float = 0.20,
        seed: int = 42,
        patience: int = 2,
        work_dir: str | Path = "output/checkpoints/robertuito_finetune",
    ) -> None:
        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.val_size = val_size
        self.seed = seed
        self.patience = patience
        self.work_dir = Path(work_dir)

        # Atributos poblados en fit().
        self._tokenizer = None
        self._trainer = None
        self._effective_max_length: int | None = None

    # ------------------------------------------------------------------
    # API de AnorexiaClassifier
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[int]) -> "RoBERTuitoFineTuner":
        """Entrena el modelo con partición estratificada y early stopping por AUC."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            set_seed,
        )

        set_seed(self.seed)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._effective_max_length = self._resolve_max_length(self._tokenizer)

        # Partición 80/20 estratificada sobre el texto YA preprocesado.
        proc_texts = self._preprocess(texts)
        import pandas as pd
        df = pd.DataFrame({"text": proc_texts, "label": list(labels)})
        train_df, val_df = stratified_split(df, val_size=self.val_size, seed=self.seed)

        train_ds = self._to_dataset(train_df["text"].tolist(), train_df["label"].tolist())
        val_ds = self._to_dataset(val_df["text"].tolist(), val_df["label"].tolist())

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        args = _make_training_args(
            output_dir=str(self.work_dir),
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            save_total_limit=1,
            seed=self.seed,
            data_seed=self.seed,
            report_to=[],
            logging_strategy="epoch",
        )

        self._trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(self._tokenizer),
            compute_metrics=_compute_auc,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )
        self._trainer.train()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) por texto usando el mejor modelo cargado."""
        if self._trainer is None:
            raise RuntimeError("El modelo no ha sido entrenado: llama a fit() primero.")

        ds = self._to_dataset(self._preprocess(texts), labels=None)
        logits = self._trainer.predict(ds).predictions
        return _softmax(np.asarray(logits))[:, 1]

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(texts: list[str]) -> list[str]:
        """Aplica el preprocesamiento obligatorio de RoBERTuito antes del tokenizer."""
        from pysentimiento.preprocessing import preprocess_tweet

        return [preprocess_tweet(t if isinstance(t, str) else "") for t in texts]

    def _resolve_max_length(self, tokenizer) -> int:
        """Acota ``max_length`` al límite real del tokenizer de RoBERTuito.

        Hugging Face usa un centinela enorme (~1e30) cuando el tokenizer no
        declara límite; en ese caso se conserva el ``max_length`` solicitado.
        """
        model_max = getattr(tokenizer, "model_max_length", None)
        if not model_max or model_max > 100_000:
            model_max = self.max_length
        return min(self.max_length, int(model_max))

    def _to_dataset(self, texts: list[str], labels: list[int] | None):
        """Construye un ``datasets.Dataset`` tokenizado (con o sin etiquetas)."""
        from datasets import Dataset

        data = {"text": list(texts)}
        if labels is not None:
            data["labels"] = list(labels)
        ds = Dataset.from_dict(data)

        max_len = self._effective_max_length
        tokenizer = self._tokenizer

        def _tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_len)

        return ds.map(_tokenize, batched=True, remove_columns=["text"])


def _make_training_args(**kwargs):
    """Crea ``TrainingArguments`` tolerando el renombrado ``evaluation_strategy``.

    transformers renombró ``evaluation_strategy`` → ``eval_strategy`` en
    versiones recientes. Detectamos la firma para usar el nombre correcto y que
    el código funcione en ambas familias de versiones.
    """
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def _compute_auc(eval_pred):
    """``compute_metrics`` del Trainer: AUC-ROC sobre P(clase positiva)."""
    from sklearn.metrics import roc_auc_score

    logits, labels = eval_pred
    probs = _softmax(np.asarray(logits))[:, 1]
    return {"auc": roc_auc_score(labels, probs)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable sobre el último eje."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
