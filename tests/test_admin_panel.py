"""El panel de administración debe usar el token PAT y no exigir Snowflake para el lote local."""
import os
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
SUMMARY = {"modelo_seleccionado": "logistic_regression", "evaluacion_prueba_reservada": {"modelo": {"auroc": 0.73}}}


def open_panel(source: str, secret: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.radio(key="admin_fuente").set_value(source).run()
    if secret is not None:
        app.text_input(key="admin_password").input(secret).run()
    return app


class AdminPanelTests(unittest.TestCase):
    def setUp(self):
        env = {"ADMIN_PANEL_CODE": "", "SNOWFLAKE_AUTHENTICATOR": "PROGRAMMATIC_ACCESS_TOKEN",
               "SNOWFLAKE_TOKEN": "", "SNOWFLAKE_PASSWORD": ""}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_token_pat_se_entrega_como_snowflake_token(self):
        with mock.patch("src.pipeline.run_full_pipeline", return_value=SUMMARY) as run:
            app = open_panel("Snowflake", secret="token-de-prueba")
            self.assertEqual(app.text_input(key="admin_password").label, "Token PAT de Snowflake")
            app.button(key="admin_entrenar").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Modelo actualizado:" in e.value for e in app.success))
            self.assertEqual(os.environ["SNOWFLAKE_TOKEN"], "token-de-prueba")
            self.assertEqual(os.environ["SNOWFLAKE_PASSWORD"], "")
            run.assert_called_once_with(backend="snowflake", use_snowflake_csv=True)

    def test_lote_local_no_usa_snowflake(self):
        with mock.patch("src.pipeline.run_full_pipeline", return_value=SUMMARY) as run:
            app = open_panel("Generar lote local pequeño")
            app.button(key="admin_entrenar").click().run()
            self.assertFalse(app.exception)
            run.assert_called_once_with(backend="local", use_snowflake_csv=False)

    def test_snowflake_sin_token_muestra_error(self):
        with mock.patch("src.pipeline.run_full_pipeline", return_value=SUMMARY) as run:
            app = open_panel("Snowflake")
            app.button(key="admin_entrenar").click().run()
            self.assertIn("Ingresa el token PAT de Snowflake.", [e.value for e in app.error])
            run.assert_not_called()

    def test_cargador_de_csv_disponible_en_ambas_fuentes(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 1)
        app.radio(key="admin_fuente").set_value("Generar lote local pequeño").run()
        self.assertEqual(len(app.get("file_uploader")), 1)
        app.radio(key="admin_fuente").set_value("Snowflake").run()
        self.assertEqual(len(app.get("file_uploader")), 1)


class AdminPanelClaveTests(unittest.TestCase):
    """Con ADMIN_PANEL_CODE configurado, el panel pide la clave antes de mostrar el reentrenamiento."""

    def setUp(self):
        env = {"ADMIN_PANEL_CODE": "clave-de-prueba", "SNOWFLAKE_AUTHENTICATOR": "PROGRAMMATIC_ACCESS_TOKEN"}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_sin_clave_no_muestra_el_panel(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_input(key="admin_code_input").value), 0)
        self.assertFalse([r for r in app.radio if r.key == "admin_fuente"])
        self.assertFalse([b for b in app.button if b.key == "admin_entrenar"])

    def test_clave_incorrecta_muestra_error(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        app.text_input(key="admin_code_input").input("otra").run()
        self.assertIn("Código incorrecto.", [e.value for e in app.error])
        self.assertFalse([b for b in app.button if b.key == "admin_entrenar"])

    def test_clave_correcta_desbloquea(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        app.text_input(key="admin_code_input").input("clave-de-prueba").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="admin_fuente").options, ["Snowflake", "Generar lote local pequeño"])
        self.assertTrue([b for b in app.button if b.key == "admin_entrenar"])


if __name__ == "__main__":
    unittest.main()
