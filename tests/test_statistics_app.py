"""Rutas, población, filtros y renderizado de las vistas del nuevo panel."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StatisticalAppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / 'app' / 'streamlit_app.py'), default_timeout=120)
        app.query_params['vista'] = 'estadisticas'
        return app.run()

    @staticmethod
    def selected(app):
        return next(c.value for c in app.caption if 'pacientes en la selección' in c.value).split()[0]

    def test_all_analysis_views_and_filter_reset(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(self.selected(app), '20.000')
        app.multiselect(key='stats_sexo').set_value(['F']).run()
        self.assertFalse(app.exception)
        self.assertEqual(self.selected(app), '15.583')
        app.multiselect(key='stats_sexo').set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No hay pacientes' in i.value for i in app.info))
        app.button(key='stats_reset').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.selected(app), '20.000')
        for view in ['Distribuciones', 'Factores asociados', 'Modelo', 'Calidad de datos', 'Cohorte']:
            app.segmented_control(key='stats_active_view').set_value(view).run()
            self.assertFalse(app.exception, view)
        app.segmented_control(key='stats_active_view').set_value('Factores asociados').run()
        app.multiselect(key='stats_riesgo_oncologico').set_value([1]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('requieren' in i.value for i in app.info))


if __name__ == '__main__':
    unittest.main()
