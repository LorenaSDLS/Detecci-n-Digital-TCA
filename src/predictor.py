"""
predictor.py — Módulo de predicción y serialización

Autor: Lorena Solís

El pipeline completo (preprocesador, extractor multi-vista, fusión y clasificador)
se serializa con joblib como un único artefacto .pkl.

El predictor carga el artefacto y procesa un archivo de prueba (.csv o .xlsx)
generando el dictamen por texto: ``[text_id, predicted_label, probability_yes]``.

Acepta dos esquemas de entrada:
    * Esquema A (protocolo oficial): ``text_id, title, text``
    * Esquema B (datos reales tipo tweet): ``user_id, tweet_id, tweet_text``
"""

from pathlib import Path

import ftfy
import joblib
import pandas as pd

_SCHEMA_A_REQUIRED = frozenset({"text_id", "title", "text"})
_SCHEMA_B_REQUIRED = frozenset({"tweet_id", "tweet_text"})


def save_pipeline(pipeline, filename: str | Path = "modelo_anorexia.pkl"):
    """Serializa el pipeline completo."""
    joblib.dump(pipeline, str(filename))
    print(f"Pipeline serializado exitosamente en: {filename}")


def run_predictor(
    test_file_path: str | Path,
    model_path: str | Path,
    output_path: str | Path = "predicciones_finales.csv",
):
    """Carga el modelo y genera el archivo de salida con el dictamen.

    Args:
        test_file_path: Ruta a un .csv o .xlsx con columnas de Esquema A
            (``text_id, title, text``) o Esquema B (``tweet_id, tweet_text``).
        model_path: Ruta al artefacto .pkl producido por ``save_pipeline``.
        output_path: Ruta del archivo de salida (.csv o .xlsx según extensión).

    Raises:
        ValueError: Si el archivo de entrada no coincide con ninguno de los
            esquemas soportados.
    """
    pipeline = joblib.load(str(model_path))
    df_test = _read_table(test_file_path)
    text_ids, X_input = _extract_inputs(df_test)

    predictions = pipeline.predict(X_input)
    probabilities = pipeline.predict_proba(X_input)[:, 1]

    output_df = pd.DataFrame({
        "text_id": text_ids,
        "predicted_label": predictions,
        "probability_yes": probabilities,
    })

    _write_table(output_df, output_path)
    print(f"Dictamen generado exitosamente en '{output_path}'.")


def _read_table(path: str | Path) -> pd.DataFrame:
    """Lee CSV o XLSX según extensión."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Extensión no soportada: {suffix!r}. Use .csv o .xlsx.")


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Escribe CSV o XLSX según extensión."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(p, index=False)
        return
    if suffix in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
        return
    raise ValueError(f"Extensión no soportada: {suffix!r}. Use .csv o .xlsx.")


def _extract_inputs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Detecta el esquema y devuelve ``(text_ids, textos)``.

    Para Esquema A combina ``title + text``; para Esquema B usa ``tweet_text``
    tal cual y mapea ``tweet_id`` → ``text_id`` en la salida.
    """
    cols = set(df.columns)
    if _SCHEMA_A_REQUIRED.issubset(cols):
        text_ids = df["text_id"]
        x = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
        return text_ids, x.str.strip().map(ftfy.fix_text)
    if _SCHEMA_B_REQUIRED.issubset(cols):
        text_ids = df["tweet_id"]
        x = df["tweet_text"].fillna("").astype(str).map(ftfy.fix_text)
        return text_ids, x
    raise ValueError(
        f"Esquema no reconocido. Columnas encontradas: {sorted(cols)}. "
        f"Se esperaba Esquema A {sorted(_SCHEMA_A_REQUIRED)} "
        f"o Esquema B {sorted(_SCHEMA_B_REQUIRED)}."
    )
