"""Comprueba la correspondencia entre el formulario y el modelo persistido."""
import unittest
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from src.processing.input_validation import prepare_patient_input


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "modelo_seleccionado.joblib"


class AppTests(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists(), "Ejecute el pipeline sintético antes de probar la app")
    def test_form_prediction_matches_pipeline(self):
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=90).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        row = {"edad": 55, "sexo": "F", "imc": 26.5, "tabaquismo": "nunca",
               "antecedente_familiar_cancer": 0, "antecedente_personal_benigno": 0,
               "presion_sistolica": 125, "glucosa_mg_dl": 98, "paridad": 1,
               "ca125_u_ml": 18.0, "resultado_imagen": "normal"}
        bundle = joblib.load(MODEL)
        x = prepare_patient_input(row)[bundle["feature_columns"]]
        expected = f"{float(bundle['pipeline'].predict_proba(x)[0, 1]):.1%}"
        displayed = [m.value for m in app.metric if m.label == "Probabilidad estimada"]
        self.assertEqual(displayed, [expected])


if __name__ == "__main__":
    unittest.main()
