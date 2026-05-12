"""
test_data_loader.py — Pruebas unitarias para el módulo data_loader.

Autor: Andrea Blanco

Cubre:
  - Detección de esquemas A y B.
  - Combinación de campos title+text (Esquema A).
  - Mapeo de etiquetas insensible a mayúsculas.
  - Errores ante esquemas y etiquetas desconocidos.
  - Filtrado de filas con texto vacío o nulo.
  - Proporciones y estratificación en la división 80/20.
  - Número correcto de folds en k-fold.
  - No solapamiento entre conjuntos de validación en k-fold.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_loader import DataLoader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_with_mock(df: pd.DataFrame) -> pd.DataFrame:
    """Llama a DataLoader.load() usando ``df`` como dato del archivo."""
    loader = DataLoader("fake_path.xlsx")
    with patch.object(loader, "_read_excel", return_value=df):
        return loader.load()


# ---------------------------------------------------------------------------
# Clase 1: Detección y normalización de esquemas
# ---------------------------------------------------------------------------

class TestSchemaDetection:

    def test_schema_b_produces_correct_columns(self, df_schema_b):
        result = _load_with_mock(df_schema_b)
        assert list(result.columns) == ["user_id", "text", "label"]

    def test_schema_b_label_values_are_integers(self, df_schema_b):
        result = _load_with_mock(df_schema_b)
        assert set(result["label"].unique()).issubset({0, 1})
        assert result["label"].dtype.kind == "i"

    def test_schema_b_anorexia_maps_to_1(self, df_schema_b):
        result = _load_with_mock(df_schema_b)
        assert result.loc[result["user_id"] == "u1", "label"].values[0] == 1
        assert result.loc[result["user_id"] == "u3", "label"].values[0] == 1

    def test_schema_b_control_maps_to_0(self, df_schema_b):
        result = _load_with_mock(df_schema_b)
        assert result.loc[result["user_id"] == "u2", "label"].values[0] == 0

    def test_schema_a_combines_title_and_text(self, df_schema_a):
        result = _load_with_mock(df_schema_a)
        first_text = result.loc[0, "text"]
        assert "ayuno extremo" in first_text
        assert "tres días sin comer" in first_text

    def test_schema_a_produces_correct_columns(self, df_schema_a):
        result = _load_with_mock(df_schema_a)
        assert list(result.columns) == ["user_id", "text", "label"]

    def test_schema_a_label_mapping(self, df_schema_a):
        result = _load_with_mock(df_schema_a)
        assert result.loc[0, "label"] == 1   # anorexia
        assert result.loc[1, "label"] == 0   # control

    def test_unknown_schema_raises_value_error(self):
        bad_df = pd.DataFrame({"col_a": [1], "col_b": ["x"]})
        with pytest.raises(ValueError, match="[Ee]squema"):
            _load_with_mock(bad_df)

    def test_unknown_schema_error_lists_columns(self):
        bad_df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="foo|bar"):
            _load_with_mock(bad_df)


# ---------------------------------------------------------------------------
# Clase 2: Mapeo de etiquetas
# ---------------------------------------------------------------------------

class TestLabelMapping:

    def test_label_mapping_case_insensitive_upper(self, df_schema_b):
        df = df_schema_b.copy()
        df["class"] = df["class"].str.upper()
        result = _load_with_mock(df)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_label_mapping_mixed_case(self, df_schema_b):
        df = df_schema_b.copy()
        df["class"] = ["Anorexia", "Control", "ANOREXIA", "control"]
        result = _load_with_mock(df)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_unknown_label_raises_value_error(self, df_schema_b):
        df = df_schema_b.copy()
        df.loc[0, "class"] = "bulimia"
        with pytest.raises(ValueError, match="bulimia"):
            _load_with_mock(df)

    def test_unknown_label_error_message_contains_valid_values(self, df_schema_b):
        df = df_schema_b.copy()
        df.loc[0, "class"] = "unknown_disorder"
        with pytest.raises(ValueError, match="anorexia|control"):
            _load_with_mock(df)


# ---------------------------------------------------------------------------
# Clase 3: Filtrado de filas con texto inválido
# ---------------------------------------------------------------------------

class TestEmptyTextFiltering:

    def test_empty_string_rows_dropped(self, df_schema_b_with_empty):
        result = _load_with_mock(df_schema_b_with_empty)
        assert len(result) == 2

    def test_whitespace_only_rows_dropped(self, df_schema_b_with_empty):
        result = _load_with_mock(df_schema_b_with_empty)
        assert all(result["text"].str.strip().str.len() > 0)

    def test_nan_text_rows_dropped(self, df_schema_b_with_nan):
        result = _load_with_mock(df_schema_b_with_nan)
        assert len(result) == 3
        assert result["text"].notna().all()

    def test_all_empty_raises_value_error(self, df_schema_b):
        df = df_schema_b.copy()
        df["tweet_text"] = ""
        with pytest.raises(ValueError):
            _load_with_mock(df)


# ---------------------------------------------------------------------------
# Clase 4: División train/val estratificada
# ---------------------------------------------------------------------------

class TestTrainValSplit:

    def test_split_sizes_add_up(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        train, val = loader.train_val_split(result, val_size=0.20)
        assert len(train) + len(val) == len(result)

    def test_split_val_proportion(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        train, val = loader.train_val_split(result, val_size=0.20)
        assert len(val) == pytest.approx(0.20 * len(result), abs=2)

    def test_split_contains_both_classes_in_val(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        _, val = loader.train_val_split(result, val_size=0.20)
        assert 0 in val["label"].values
        assert 1 in val["label"].values

    def test_split_contains_both_classes_in_train(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        train, _ = loader.train_val_split(result, val_size=0.20)
        assert 0 in train["label"].values
        assert 1 in train["label"].values

    def test_split_reset_index(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        train, val = loader.train_val_split(result, val_size=0.20)
        assert list(train.index) == list(range(len(train)))
        assert list(val.index) == list(range(len(val)))

    def test_split_custom_val_size(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        train, val = loader.train_val_split(result, val_size=0.50)
        assert len(train) == pytest.approx(50, abs=2)
        assert len(val) == pytest.approx(50, abs=2)


# ---------------------------------------------------------------------------
# Clase 5: K-fold cross-validation estratificado
# ---------------------------------------------------------------------------

class TestKFoldSplit:

    def test_kfold_yields_correct_number_of_folds(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        folds = list(loader.kfold_split(result, n_splits=5))
        assert len(folds) == 5

    def test_kfold_all_indices_covered_exactly_once(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        val_indices = []
        for _, val in loader.kfold_split(result, n_splits=5):
            val_indices.extend(val.index.tolist())
        assert len(val_indices) == len(result)

    def test_kfold_val_sets_do_not_overlap(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        val_user_ids = []
        for _, val in loader.kfold_split(result, n_splits=5):
            val_user_ids.extend(val["user_id"].tolist())
        assert len(val_user_ids) == len(set(val_user_ids)), "Los conjuntos de val se solapan"

    def test_kfold_train_and_val_sizes_sum_to_total(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        for train, val in loader.kfold_split(result, n_splits=5):
            assert len(train) + len(val) == len(result)

    def test_kfold_3_folds(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        folds = list(loader.kfold_split(result, n_splits=3))
        assert len(folds) == 3

    def test_kfold_reset_index(self, df_balanced_100):
        result = _load_with_mock(df_balanced_100)
        loader = DataLoader("fake.xlsx")
        for train, val in loader.kfold_split(result, n_splits=5):
            assert list(train.index) == list(range(len(train)))
            assert list(val.index) == list(range(len(val)))
