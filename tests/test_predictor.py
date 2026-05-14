"""
test_predictor.py — Pruebas unitarias para el módulo predictor (Sección 3.7).

Autor: Lorena Solís

Cubre:
  - Serialización correcta del Pipeline completo con joblib (.pkl).
  - Integridad del artefacto cargado (persistencia de estados).
  - Procesamiento de archivos Excel de entrada siguiendo el esquema del protocolo.
  - Validación del formato de salida: columnas [text_id, predicted_label, probability_yes].
  - Consistencia en la combinación de campos title + text.
"""

import pytest
import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.predictor import save_pipeline, run_predictor
from sklearn.base import BaseEstimator  
import numpy as np 


def identity_step(X): 
    """Función de identidad global para que sea serializable."""
    return X



class DummyClf(BaseEstimator): 
    """Clase global compatible con las últimas versiones de sklearn."""
    def fit(self, X, y=None): 
        self.fitted_ = True 
        return self
    
    def predict(self, X): 
        # Devolvemos un array de numpy
        return np.array([1] * len(X))
    
    def predict_proba(self, X): 
        # Devolvemos un array de numpy con dos columnas
        return np.array([[0.1, 0.9]] * len(X))
    
    def __sklearn_is_fitted__(self):
        return True

class TestPredictor:

    @pytest.fixture
    def mock_pipeline(self):
        """Crea un Pipeline usando las funciones globales definidas arriba."""
        return Pipeline([
            ('prep', FunctionTransformer(identity_step)),
            ('clf', DummyClf())
        ])

    def test_save_and_load_consistency(self, mock_pipeline, tmp_path):
        """Verifica que el archivo .pkl se guarde y sea legible."""
        pkl_path = tmp_path / "model_test.pkl"
        save_pipeline(mock_pipeline, str(pkl_path))
        
        assert os.path.exists(pkl_path)
        loaded_model = joblib.load(str(pkl_path))
        assert isinstance(loaded_model, Pipeline)

    def test_protocol_output_schema_xlsx_input(self, mock_pipeline, tmp_path):
        """Esquema A en .xlsx → CSV de salida con las columnas exactas."""
        model_path = tmp_path / "final_model.pkl"
        input_path = tmp_path / "input_test.xlsx"

        save_pipeline(mock_pipeline, str(model_path))

        input_df = pd.DataFrame({
            'text_id': ['abc-123'],
            'title': ['Ayuda'],
            'text': ['No me siento bien']
        })
        input_df.to_excel(input_path, index=False)

        run_predictor(str(input_path), str(model_path))

        assert os.path.exists('predicciones_finales.csv')
        df_final = pd.read_csv('predicciones_finales.csv')

        expected = ['text_id', 'predicted_label', 'probability_yes']
        assert list(df_final.columns) == expected
        assert df_final.iloc[0]['text_id'] == 'abc-123'

    def test_schema_b_csv_input(self, mock_pipeline, tmp_path):
        """Esquema B en .csv (tweet_id/tweet_text) también es aceptado."""
        model_path = tmp_path / "final_model.pkl"
        input_path = tmp_path / "input_test.csv"

        save_pipeline(mock_pipeline, str(model_path))

        pd.DataFrame({
            'user_id': ['u1'],
            'tweet_id': ['xyz-789'],
            'tweet_text': ['hoy no he comido nada'],
        }).to_csv(input_path, index=False)

        run_predictor(str(input_path), str(model_path))

        df_final = pd.read_csv('predicciones_finales.csv')
        assert list(df_final.columns) == ['text_id', 'predicted_label', 'probability_yes']
        assert df_final.iloc[0]['text_id'] == 'xyz-789'