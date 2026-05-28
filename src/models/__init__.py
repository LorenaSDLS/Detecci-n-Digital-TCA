"""src.models — Catálogo de métodos de detección de anorexia.

Este paquete agrupa todas las *variantes* (métodos) del proyecto bajo una
interfaz común (:class:`~src.models.base.AnorexiaClassifier`):

  * Fase 2B  — baseline TF-IDF/BoW multivista + clasificador clásico
    (implementado en ``src/`` y consumido vía su archivo de predicciones; el
    :mod:`~src.models.registry` lo cataloga junto al resto).
  * Fase 3a  — fine-tuning de RoBERTuito        (``finetuning.py``, IMPLEMENTADO).
  * Fase 3b  — embeddings de RoBERTuito + SVM    (``embeddings_svm.py``, pendiente).
  * Fase 3c  — zero-shot con un modelo NLI       (``zeroshot_nli.py``, pendiente).

Todas las variantes producen predicciones en el mismo esquema que el baseline
(``[text_id, predicted_label, probability_yes]``) para que el orquestador
``python -m src.models`` las evalúe de forma homogénea por AUC-ROC.

Las dependencias pesadas (torch, transformers, pysentimiento, datasets) se
importan de forma diferida dentro de los métodos, por lo que importar este
paquete o el registro NO requiere tenerlas instaladas.
"""

from src.models.base import AnorexiaClassifier

__all__ = ["AnorexiaClassifier"]
