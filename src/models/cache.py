"""cache.py — Caché de predicciones por método y variante.

Evita rehacer todo desde cero al ver la tabla comparativa: si para un método
NO cambiaron ni los datos, ni su configuración, ni su código fuente, se reutiliza
su archivo de predicciones en vez de volver a entrenar/predecir (clave para el
fine-tuning, lento, y para el LLM, que cuesta llamadas de API).

Junto a cada ``predicciones_<name>.csv`` se escribe un sidecar
``predicciones_<name>.csv.meta.json`` con una huella (fingerprint). En la
siguiente corrida se recomputa la huella y, si coincide con la guardada, hay
acierto de caché.

La huella combina:
  * el nombre del método,
  * su configuración pública (``AnorexiaClassifier.config()`` → hiperparámetros),
  * el hash del contenido de los archivos de train y test,
  * el hash del código fuente de la clase del método.

Así, editar ``finetuning.py`` o cambiar ``data_train.csv`` invalida la caché
automáticamente, pero re-ejecutar sin cambios la respeta.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | Path) -> str:
    """Hash del contenido de un archivo (``"missing"`` si no existe)."""
    p = Path(path)
    return _sha256(p.read_bytes()) if p.exists() else "missing"


def code_digest(obj: Any) -> str:
    """Hash del código fuente de la CLASE de ``obj`` (invalida al editar el método)."""
    try:
        source = inspect.getsource(type(obj))
    except (OSError, TypeError):
        return "unknown"
    return _sha256(source.encode("utf-8"))


def fingerprint(
    *,
    name: str,
    config: dict,
    train_file: str | Path,
    test_file: str | Path,
    code: str,
) -> str:
    """Huella determinista de una corrida método×datos×config×código."""
    payload = json.dumps(
        {
            "name": name,
            "config": config,
            "train": file_digest(train_file),
            "test": file_digest(test_file),
            "code": code,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha256(payload.encode("utf-8"))


def meta_path(pred_path: Path) -> Path:
    """Ruta del sidecar de metadatos para un archivo de predicciones."""
    return pred_path.parent / (pred_path.name + ".meta.json")


def is_fresh(pred_path: Path, fp: str) -> bool:
    """``True`` si existen predicciones y su huella guardada coincide con ``fp``."""
    mp = meta_path(pred_path)
    if not pred_path.exists() or not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("fingerprint") == fp


def write_meta(pred_path: Path, fp: str, extra: dict | None = None) -> None:
    """Guarda el sidecar con la huella (y métricas opcionales) tras una corrida."""
    meta: dict[str, Any] = {"fingerprint": fp}
    if extra:
        meta.update(extra)
    meta_path(pred_path).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
