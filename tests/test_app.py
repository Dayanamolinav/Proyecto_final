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

    @unittest.skipUnless(MODEL.exists(), "Ejecute el pipeline sintético antes de probar la app")
    def test_edited_submission_and_rerun_keep_expected_result(self):
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=90).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        previous = [m.value for m in app.metric if m.label == "Probabilidad estimada"]

        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric if m.label == "Probabilidad estimada"], previous)

        next(widget for widget in app.number_input if widget.label == "Edad (años)").set_value(75)
        next(widget for widget in app.number_input if widget.label == "IMC (kg/m²)").set_value(32.0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        row = {"edad": 75, "sexo": "F", "imc": 32.0, "tabaquismo": "nunca",
               "antecedente_familiar_cancer": 0, "antecedente_personal_benigno": 0,
               "presion_sistolica": 125, "glucosa_mg_dl": 98, "paridad": 1,
               "ca125_u_ml": 18.0, "resultado_imagen": "normal"}
        bundle = joblib.load(MODEL)
        x = prepare_patient_input(row)[bundle["feature_columns"]]
        expected = f"{float(bundle['pipeline'].predict_proba(x)[0, 1]):.1%}"
        self.assertEqual([m.value for m in app.metric if m.label == "Probabilidad estimada"], [expected])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric if m.label == "Probabilidad estimada"], [expected])


if __name__ == "__main__":
    unittest.main()
