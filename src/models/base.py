"""base.py — Interfaz común para las variantes de Fase 3.

Define el contrato que comparten todos los métodos (fine-tuning, embeddings+SVM,
zero-shot NLI) de modo que el orquestador ``compare.py`` los trate de forma
intercambiable y que nuevas variantes se integren sin refactorizar nada.

Una variante concreta sólo necesita implementar dos métodos:

    * :meth:`fit`           — entrena / prepara el modelo.
    * :meth:`predict_proba` — devuelve P(anorexia) en [0, 1] por texto.

El método concreto :meth:`run` (heredado, no se sobreescribe) orquesta
``fit → predict_proba → escritura del CSV`` en el esquema unificado
``[text_id, predicted_label, probability_yes]``, idéntico al que emite el
baseline de Fase 2B.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class AnorexiaClassifier(ABC):
    """Clase base abstracta para una variante de clasificación de anorexia.

    Attributes:
        name: Identificador corto del método. Se usa en la tabla comparativa
            y para nombrar el archivo de predicciones (``predicciones_<name>.csv``).
            Las subclases DEBEN sobreescribirlo.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, texts: list[str], labels: list[int]) -> "AnorexiaClassifier":
        """Entrena (o prepara) el modelo a partir de textos y etiquetas binarias.

        Args:
            texts: Textos crudos de entrenamiento.
            labels: Etiquetas binarias (1=anorexia, 0=control).

        Returns:
            ``self``, para permitir encadenamiento.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve la probabilidad de la clase positiva (anorexia) por texto.

        Args:
            texts: Textos crudos a clasificar.

        Returns:
            Array 1-D de probabilidades en [0, 1], alineado con ``texts``.
        """
        raise NotImplementedError

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str | Path,
        threshold: float = 0.5,
    ) -> Path:
        """Ejecuta el método de extremo a extremo y escribe sus predicciones.

        Entrena con ``train_df`` y predice sobre ``test_df``, volcando el
        resultado en ``output_dir/predicciones_<name>.csv`` con el esquema
        unificado ``[text_id, predicted_label, probability_yes]``.

        Args:
            train_df: DataFrame con columnas ``[text_id, text, label]``.
            test_df: DataFrame con columnas ``[text_id, text]`` (``label`` opcional).
            output_dir: Directorio donde escribir el archivo de predicciones.
            threshold: Umbral sobre P(anorexia) para la etiqueta dura.

        Returns:
            Ruta del archivo de predicciones escrito.
        """
        self.fit(train_df["text"].tolist(), train_df["label"].tolist())

        probs = np.asarray(self.predict_proba(test_df["text"].tolist()), dtype=float)
        preds = (probs >= threshold).astype(int)

        out = pd.DataFrame({
            "text_id": test_df["text_id"].to_numpy(),
            "predicted_label": preds,
            "probability_yes": probs,
        })

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"predicciones_{self.name}.csv"
        out.to_csv(path, index=False)
        return path
