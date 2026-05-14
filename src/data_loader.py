"""
data_loader.py — Módulo de ingesta de datos.

Autor: Andrea Blanco

Responsable de leer el archivo .xlsx de entrenamiento con pandas (motor openpyxl),
validar el esquema de columnas, normalizar a una representación interna unificada
[user_id, text, label], y proveer particiones de entrenamiento/validación tanto
en modalidad de división simple (80/20 estratificada) como en k-fold estratificado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import ftfy
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class DataLoader:
    """Carga y valida el conjunto de datos de tweets para detección de TCA.

    Soporta dos esquemas de columnas:

    Esquema A (hipotético, descrito en el protocolo):
        user_id, text_id, title, text, label
        combina ``title`` + ``text`` en un único campo.

    Esquema B (datos de entrenamiento reales):
        user_id, tweet_id, tweet_text, class

    En ambos casos la salida normalizada contiene únicamente:
        user_id (str), text (str), label (int: 1=anorexia, 0=control)
    """

    # Columnas mínimas requeridas por cada esquema
    _SCHEMA_A_REQUIRED: frozenset[str] = frozenset({"user_id", "text_id", "title", "text", "label"})
    _SCHEMA_B_REQUIRED: frozenset[str] = frozenset({"user_id", "tweet_id", "tweet_text", "class"})

    # Mapeo de etiquetas de clase a entero (insensible a mayúsculas)
    _LABEL_MAP: dict[str, int] = {"anorexia": 1, "control": 0}

    def __init__(self, filepath: str | Path, random_state: int = 42) -> None:
        """Inicializa el cargador.

        Args:
            filepath: Ruta al archivo .xlsx con los datos de entrenamiento.
            random_state: Semilla aleatoria para reproducibilidad de las particiones.
        """
        self.filepath = Path(filepath)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Lee el archivo, valida el esquema y devuelve el DataFrame normalizado.

        Soporta archivos ``.csv`` y ``.xlsx`` (detectado por extensión).

        Returns:
            DataFrame con columnas [user_id, text, label].

        Raises:
            FileNotFoundError: Si ``self.filepath`` no existe.
            ValueError: Si el esquema de columnas no es reconocido, si hay valores
                de etiqueta desconocidos, o si el archivo queda vacío.
        """
        raw = self._read_table()
        return self._validate_and_normalize(raw)

    def train_val_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """División estratificada del DataFrame según la proporción ``val_size``.

        Args:
            df: DataFrame normalizado devuelto por :meth:`load`.
            val_size: Proporción del conjunto de validación (0 < val_size < 1).

        Returns:
            Tupla ``(train_df, val_df)`` con índices reseteados.
        """
        train, val = train_test_split(
            df,
            test_size=val_size,
            stratify=df["label"],
            random_state=self.random_state,
        )
        return train.reset_index(drop=True), val.reset_index(drop=True)

    def kfold_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Validación cruzada estratificada de k iteraciones.

        Cada iteración garantiza que ambas particiones mantienen la proporción
        de clases del conjunto original.

        Args:
            df: DataFrame normalizado devuelto por :meth:`load`.
            n_splits: Número de particiones (folds).

        Yields:
            Tuplas ``(train_df, val_df)`` con índices reseteados para cada fold.
        """
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for train_idx, val_idx in skf.split(df, df["label"]):
            yield (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
            )

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _read_table(self) -> pd.DataFrame:
        """Lee el archivo de entrada, dispatcheando por extensión.

        Returns:
            DataFrame crudo con todas las columnas originales.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la extensión no es ``.csv``, ``.xlsx`` ni ``.xls``.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.filepath}")
        suffix = self.filepath.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.filepath)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.filepath, engine="openpyxl")
        raise ValueError(
            f"Extensión no soportada: {suffix!r}. Use .csv o .xlsx."
        )

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta el esquema, normaliza columnas, mapea etiquetas y filtra filas vacías.

        Args:
            df: DataFrame crudo leído desde el archivo .xlsx.

        Returns:
            DataFrame con columnas exactamente [user_id, text, label].

        Raises:
            ValueError: Si el esquema no es reconocido, hay etiquetas inválidas,
                o el DataFrame queda vacío tras el filtrado.
        """
        cols = set(df.columns)

        if self._SCHEMA_A_REQUIRED.issubset(cols):
            df = self._normalize_schema_a(df)
        elif self._SCHEMA_B_REQUIRED.issubset(cols):
            df = self._normalize_schema_b(df)
        else:
            raise ValueError(
                f"Esquema de columnas no reconocido. "
                f"Columnas encontradas: {sorted(cols)}. "
                f"Se esperaba Esquema A {sorted(self._SCHEMA_A_REQUIRED)} "
                f"o Esquema B {sorted(self._SCHEMA_B_REQUIRED)}."
            )

        # Normalizar encoding (corrige mojibake como "azÃºcar" → "azúcar").
        # ftfy.fix_text es idempotente. Se preservan los NaN para que el filtro
        # de filas vacías que sigue los elimine de forma uniforme.
        df = df.copy()
        df["text"] = df["text"].map(
            lambda x: ftfy.fix_text(x) if isinstance(x, str) else x
        )

        # Eliminar filas con texto nulo o vacío
        df = df[df["text"].notna()]
        df = df[df["text"].str.strip().str.len() > 0]
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "El DataFrame está vacío tras eliminar filas con texto inválido."
            )

        return df[["user_id", "text", "label"]]

    def _normalize_schema_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza Esquema A: combina title+text y mapea etiquetas."""
        df = df.copy()
        title = df["title"].fillna("").astype(str)
        body = df["text"].fillna("").astype(str)
        df["text"] = (title + " " + body).str.strip()
        df = self._map_labels(df, label_col="label")
        return df

    def _normalize_schema_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza Esquema B: renombra tweet_text y mapea la columna class."""
        df = df.copy()
        df = df.rename(columns={"tweet_text": "text"})
        df = self._map_labels(df, label_col="class")
        return df

    def _map_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Convierte la columna de etiquetas textuales a enteros {0, 1}.

        Args:
            df: DataFrame con la columna ``label_col``.
            label_col: Nombre de la columna que contiene las etiquetas.

        Returns:
            DataFrame con columna ``label`` (int) añadida.

        Raises:
            ValueError: Si se encuentran valores de etiqueta fuera de
                {'anorexia', 'control'}.
        """
        labels_lower = df[label_col].astype(str).str.lower().str.strip()
        unknown = set(labels_lower.unique()) - set(self._LABEL_MAP.keys())
        if unknown:
            raise ValueError(
                f"Valores de etiqueta no reconocidos en '{label_col}': {unknown}. "
                f"Valores permitidos: {set(self._LABEL_MAP.keys())}."
            )
        df = df.copy()
        df["label"] = labels_lower.map(self._LABEL_MAP).astype(int)
        if label_col != "label":
            df = df.drop(columns=[label_col])
        return df
