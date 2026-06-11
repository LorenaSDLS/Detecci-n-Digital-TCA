"""test_phase3_models.py — Pruebas de la infraestructura de métodos de Fase 3.

Cubre data.py (loader que preserva text_id), base.py (contrato run()),
registry.py (catálogo y manejo de variantes pendientes) y los helpers puros de
finetuning.py. No requiere torch/transformers (las variantes difieren esos
imports), por lo que estas pruebas corren en cualquier entorno.

Autor: Carlos Zamudio (Fase 3)
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base import AnorexiaClassifier
from src.models.data import load_dataset, load_gold, stratified_split
from src.models.embeddings_svm import RoBERTuitoSVM
from src.models.finetuning import RoBERTuitoFineTuner, _softmax
from src.models.registry import REGISTRY
from src.models.zeroshot_nli import ZeroShotNLI


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestDataLoader:
    def _write(self, tmp_path, df):
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        return path

    def test_schema_b_preserves_tweet_id_as_text_id(self, tmp_path, df_schema_b):
        path = self._write(tmp_path, df_schema_b)
        out = load_dataset(path)
        assert list(out.columns) == ["text_id", "text", "label"]
        assert out["text_id"].tolist() == ["t1", "t2", "t3", "t4"]
        assert out["label"].tolist() == [1, 0, 1, 0]

    def test_schema_a_combines_title_and_text(self, tmp_path, df_schema_a):
        path = self._write(tmp_path, df_schema_a)
        out = load_dataset(path)
        assert out["text_id"].tolist() == ["t1", "t2"]
        assert out.loc[0, "text"].startswith("ayuno extremo")
        assert out["label"].tolist() == [1, 0]

    def test_drops_empty_rows(self, tmp_path, df_schema_b_with_empty):
        path = self._write(tmp_path, df_schema_b_with_empty)
        out = load_dataset(path)
        assert len(out) == 2  # las dos filas vacías/espacios se descartan

    def test_unknown_label_raises(self, tmp_path, df_schema_b):
        df = df_schema_b.copy()
        df.loc[0, "class"] = "bulimia"
        path = self._write(tmp_path, df)
        with pytest.raises(ValueError, match="no reconocidos"):
            load_dataset(path)

    def test_unknown_schema_raises(self, tmp_path):
        path = self._write(tmp_path, pd.DataFrame({"foo": [1], "bar": [2]}))
        with pytest.raises(ValueError, match="no reconocido"):
            load_dataset(path)

    def test_load_gold_returns_id_and_label(self, tmp_path, df_schema_b):
        path = self._write(tmp_path, df_schema_b)
        gold = load_gold(path)
        assert list(gold.columns) == ["text_id", "label"]

    def test_stratified_split_preserves_class_ratio(self, df_balanced_100):
        df = pd.DataFrame({
            "text_id": df_balanced_100["tweet_id"],
            "text": df_balanced_100["tweet_text"],
            "label": [1] * 50 + [0] * 50,
        })
        train, val = stratified_split(df, val_size=0.2, seed=42)
        assert len(train) == 80 and len(val) == 20
        assert val["label"].sum() == 10  # 50% de cada clase preservado


# ---------------------------------------------------------------------------
# base.py — contrato run()
# ---------------------------------------------------------------------------

class _FakeClassifier(AnorexiaClassifier):
    """Variante de juguete: probabilidad fija por texto, sin dependencias."""

    name = "fake"

    def __init__(self, probs):
        self._probs = probs
        self.fitted = False

    def fit(self, texts, labels):
        self.fitted = True
        return self

    def predict_proba(self, texts):
        return np.array(self._probs[: len(texts)])


class TestRunContract:
    def test_run_writes_unified_schema(self, tmp_path):
        train = pd.DataFrame({"text_id": ["a"], "text": ["x"], "label": [1]})
        test = pd.DataFrame({"text_id": ["t1", "t2"], "text": ["a", "b"]})
        model = _FakeClassifier(probs=[0.9, 0.1])

        path = model.run(train, test, output_dir=tmp_path)

        assert path.name == "predicciones_fake.csv"
        assert model.fitted
        out = pd.read_csv(path)
        assert list(out.columns) == ["text_id", "predicted_label", "probability_yes"]
        assert out["predicted_label"].tolist() == [1, 0]  # umbral 0.5
        assert out["text_id"].tolist() == ["t1", "t2"]

    def test_run_threshold_is_applied(self, tmp_path):
        train = pd.DataFrame({"text_id": ["a"], "text": ["x"], "label": [1]})
        test = pd.DataFrame({"text_id": ["t1"], "text": ["a"]})
        path = _FakeClassifier(probs=[0.6]).run(train, test, tmp_path, threshold=0.7)
        assert pd.read_csv(path)["predicted_label"].tolist() == [0]


# ---------------------------------------------------------------------------
# registry.py y placeholders
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_catalog_lists_all_methods(self):
        names = [e.name for e in REGISTRY]
        assert names == [
            "baseline_2b",
            "baseline_clasico",
            "robertuito_finetune",
            "robertuito_svm",
            "nli_zeroshot",
            "llm_groq",
        ]

    def test_matrix_has_five_methods(self):
        # La matriz método×variante cubre exactamente 5 métodos (×2 variantes = 10).
        matrix = [e.name for e in REGISTRY if e.in_matrix]
        assert matrix == [
            "baseline_clasico",
            "robertuito_finetune",
            "robertuito_svm",
            "nli_zeroshot",
            "llm_groq",
        ]

    def test_baseline_is_not_trainable(self):
        baseline = next(e for e in REGISTRY if e.name == "baseline_2b")
        assert not baseline.is_trainable
        assert baseline.predictions_file == "predicciones_finales.csv"
        assert not baseline.in_matrix

    def test_variant_filenames_match_run_output(self):
        # base.run() escribe predicciones_<name>.csv → debe coincidir con el registro.
        for entry in REGISTRY:
            if entry.is_trainable:
                assert (
                    entry.predictions_file == f"predicciones_{entry.factory().name}.csv"
                )

    def test_finetuner_metadata_without_torch(self):
        m = RoBERTuitoFineTuner()
        assert m.name == "robertuito_finetune"
        assert m.model_name == "pysentimiento/robertuito-base-uncased"
        assert m.lr == 2e-5 and m.batch_size == 16 and m.epochs == 4

    def test_embeddings_svm_metadata_without_transformers(self):
        m = RoBERTuitoSVM()
        assert m.name == "robertuito_svm"
        assert m.model_name == "pysentimiento/robertuito-base-uncased"
        assert m.max_length == 128
        assert m.seed == 42

    def test_embeddings_svm_run_with_mocked_embeddings(self, tmp_path, monkeypatch):
        train = pd.DataFrame({
            "text_id": ["a", "b", "c", "d"],
            "text": [
                "riesgo tca uno",
                "riesgo tca dos",
                "control uno",
                "control dos",
        ],
            "label": [1, 1, 0, 0],
        })

        test = pd.DataFrame({
            "text_id": ["t1", "t2"],
            "text": ["riesgo tca prueba", "control prueba"],
        })

        def fake_extract_embeddings(self, texts):
            vectors = []
            for text in texts:
                if "riesgo" in text:
                    vectors.append([2.0, 2.0, 1.5])
                else:
                    vectors.append([-2.0, -2.0, -1.5])
            return np.asarray(vectors, dtype=np.float32)

        monkeypatch.setattr(RoBERTuitoSVM, "_load_transformer", lambda self: None)
        monkeypatch.setattr(RoBERTuitoSVM, "_extract_embeddings", fake_extract_embeddings)

        path = RoBERTuitoSVM().run(train, test, tmp_path)

        out = pd.read_csv(path)

        assert path.name == "predicciones_robertuito_svm.csv"
        assert list(out.columns) == ["text_id", "predicted_label", "probability_yes"]
        assert out["text_id"].tolist() == ["t1", "t2"]
        assert out["probability_yes"].between(0, 1).all()

    def test_zeroshot_metadata_and_noop_fit(self):
        z = ZeroShotNLI()
        assert z.name == "nli_zeroshot"
        assert z.model_name == "Recognai/bert-base-spanish-wwm-cased-xnli"
        assert z.hypothesis_template == "Este texto trata sobre {}."
        assert z.fit([], []) is z  # fit es no-op (zero-shot), encadenable.

    def test_zeroshot_positive_score_picks_positive_label(self):
        # _positive_score es puro (no requiere transformers): recupera el score
        # de positive_label por su posición en las labels ordenadas del pipeline.
        z = ZeroShotNLI()
        result = {
            "labels": [z.negative_label, z.positive_label],
            "scores": [0.7, 0.3],
        }
        assert z._positive_score(result) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# finetuning.py — helpers puros
# ---------------------------------------------------------------------------

class TestFineTuningHelpers:
    def test_softmax_rows_sum_to_one(self):
        out = _softmax(np.array([[2.0, 1.0], [0.0, 0.0]]))
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])
        assert out[0, 0] > out[0, 1]  # logit mayor → prob mayor

    @pytest.mark.parametrize("model_max,expected", [(128, 128), (64, 64), (int(1e30), 128)])
    def test_resolve_max_length(self, model_max, expected):
        tokenizer = type("Tok", (), {"model_max_length": model_max})()
        assert RoBERTuitoFineTuner(max_length=128)._resolve_max_length(tokenizer) == expected
