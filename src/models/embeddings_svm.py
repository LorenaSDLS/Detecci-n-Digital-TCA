"""embeddings_svm.py — Variante (b) de Fase 3: embeddings RoBERTuito + clasificador.

Usa RoBERTuito como extractor de embeddings congelado y entrena un clasificador
clásico sobre esos vectores. A diferencia del fine-tuning, los pesos de
RoBERTuito no se actualizan; sólo aprende la cabeza.

La cabeza es intercambiable (``head``) para comparar varios clasificadores sobre
la MISMA representación:

  * ``svm``    — LinearSVC calibrado (la variante original).
  * ``logreg`` — Regresión logística L2 (probabilidades nativas).
  * ``xgb``    — XGBoost (no lineal, basado en árboles).

Los embeddings se cachean en disco por (modelo, textos): extraerlos es lo caro
(~minutos en CPU), así que las tres cabezas y las repeticiones de la matriz
reutilizan la misma matriz de vectores.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from src.models.base import AnorexiaClassifier

_HEADS = ("svm", "logreg", "xgb")

# Versión del preprocesamiento+pooling: súbela si cambian, para invalidar el
# caché de embeddings en disco (su clave no ve el código, sólo modelo+textos).
_EMBEDDINGS_VERSION = "v1"


class RoBERTuitoEmbeddings(AnorexiaClassifier):
    """Embeddings de RoBERTuito congelado + cabeza clasificadora intercambiable.

    Attributes:
        head: Cabeza a entrenar sobre los embeddings (svm | logreg | xgb).
        model_name: Checkpoint de Hugging Face usado como extractor.
        max_length: Longitud máxima de tokens.
        batch_size: Tamaño de lote para la extracción.
        seed: Semilla aleatoria.
        embeddings_cache_dir: Directorio del caché de embeddings en disco.
    """

    def __init__(
        self,
        head: str = "svm",
        model_name: str = "pysentimiento/robertuito-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        seed: int = 42,
        embeddings_cache_dir: str = "output/cache_embeddings",
    ) -> None:
        if head not in _HEADS:
            raise ValueError(f"Cabeza desconocida: {head!r}. Opciones: {_HEADS}.")
        self.head = head
        self.name = f"robertuito_{head}"
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        self.embeddings_cache_dir = embeddings_cache_dir

        # Estos atributos se llenan durante fit().
        self._tokenizer = None
        self._model = None
        self._clf = None
        self._device = None
        self._effective_max_length: int | None = None

    # ------------------------------------------------------------------
    # API obligatoria de AnorexiaClassifier
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[int]) -> "RoBERTuitoEmbeddings":
        """Extrae embeddings de los textos de entrenamiento y entrena la cabeza."""
        if len(texts) != len(labels):
            raise ValueError("texts y labels deben tener la misma longitud.")

        embeddings = self._extract_embeddings(texts)
        y = np.asarray(labels, dtype=int)

        class_counts = Counter(y.tolist())
        if len(class_counts) < 2:
            raise ValueError("El entrenamiento necesita ejemplos de ambas clases: 0 y 1.")

        self._clf = self._build_head(class_counts)
        self._clf.fit(embeddings, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) para cada texto usando RoBERTuito + la cabeza."""
        if self._clf is None:
            raise RuntimeError("El modelo no ha sido entrenado: llama a fit() primero.")

        embeddings = self._extract_embeddings(texts)
        probabilities = self._clf.predict_proba(embeddings)

        # Columna 1 = probabilidad de la clase positiva.
        # En este proyecto: 1 = anorexia / posible TCA.
        return probabilities[:, 1]

    # ------------------------------------------------------------------
    # Cabezas clasificadoras
    # ------------------------------------------------------------------

    def _build_head(self, class_counts: Counter):
        """Construye la cabeza según ``self.head`` (imports diferidos)."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.head == "svm":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import LinearSVC

            # CalibratedClassifierCV necesita ejemplos suficientes por clase para
            # la validación cruzada interna: hasta 5 folds, menos si la clase
            # minoritaria es pequeña.
            cv = min(5, min(class_counts.values()))
            if cv < 2:
                raise ValueError("Cada clase necesita al menos 2 ejemplos para calibrar el SVM.")

            base_svm = LinearSVC(
                class_weight="balanced",
                random_state=self.seed,
                max_iter=10_000,
                dual="auto",
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(estimator=base_svm, cv=cv)),
            ])

        if self.head == "logreg":
            from sklearn.linear_model import LogisticRegression

            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=self.seed,
                )),
            ])

        # xgb — los árboles no necesitan escalado.
        from xgboost import XGBClassifier

        # n_jobs=1: en macOS, los hilos OpenMP de xgboost conviven mal con un
        # proceso que ya usó torch/MPS (segfault). Con 1500×768 el costo es bajo.
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=1,
        )

    # ------------------------------------------------------------------
    # Extracción de embeddings (con caché en disco)
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

        Reutiliza el caché en disco si estos mismos textos ya fueron extraídos
        con este mismo modelo (las distintas cabezas comparten la extracción).

        Resultado:
            np.ndarray de forma (n_textos, hidden_size). Para RoBERTuito base,
            normalmente hidden_size = 768.
        """
        cache_file = self._embeddings_cache_path(texts)
        if cache_file.exists():
            return np.load(cache_file)

        embeddings = self._compute_embeddings(texts)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
        return embeddings

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Extracción real (forward del transformer por lotes)."""
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

    def _embeddings_cache_path(self, texts: list[str]) -> Path:
        """Ruta del caché para estos textos: hash de (modelo, versión, textos)."""
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        digest.update(_EMBEDDINGS_VERSION.encode("utf-8"))
        digest.update(str(self.max_length).encode("utf-8"))
        for text in texts:
            digest.update((text if isinstance(text, str) else "").encode("utf-8", "ignore"))
            digest.update(b"\x00")
        return Path(self.embeddings_cache_dir) / f"{digest.hexdigest()}.npy"

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


# Alias de compatibilidad: la variante original (cabeza SVM por defecto).
RoBERTuitoSVM = RoBERTuitoEmbeddings
