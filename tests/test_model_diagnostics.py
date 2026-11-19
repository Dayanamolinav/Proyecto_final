"""La nueva visualización debe reproducir la prueba persistida, no entrenar."""
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.analytics.model_diagnostics import diagnostic_results, recover_holdout

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'modelo_seleccionado.joblib'
ANALYTICAL = ROOT / 'data' / 'analytical' / 'clinical_analytical.parquet'


@unittest.skipUnless(MODEL.exists() and ANALYTICAL.exists(), "Ejecute el pipeline antes de probar la evaluación")
class ModelDiagnosticTests(unittest.TestCase):
    def test_reconstructs_original_holdout_and_known_metrics(self):
        bundle = joblib.load(MODEL)
        stored = bundle['evaluation']['modelo']
        holdout = recover_holdout(bundle, ANALYTICAL)
        expected_n = int(stored['n_verdaderos_positivos'] + stored['n_falsos_positivos']
                         + stored['n_verdaderos_negativos'] + stored['n_falsos_negativos'])
        self.assertEqual(len(holdout), expected_n)
        self.assertEqual(holdout.id_paciente.nunique(), expected_n)
        self.assertEqual(int(holdout.etiqueta.sum()), int(stored['n_verdaderos_positivos'] + stored['n_falsos_negativos']))
        threshold = bundle.get('threshold', .5)
        results = diagnostic_results(holdout, threshold=threshold, resamples=20)
        self.assertEqual(results['matrix'].Pacientes.tolist(),
                         [int(stored[k]) for k in ('n_verdaderos_negativos', 'n_falsos_positivos',
                                                   'n_falsos_negativos', 'n_verdaderos_positivos')])
        metrics = results['metrics'].set_index('Métrica')
        self.assertAlmostEqual(metrics.loc['ROC-AUC', 'Estimación'], stored['auroc'], places=4)
        accuracy = (stored['n_verdaderos_positivos'] + stored['n_verdaderos_negativos']) / expected_n
        self.assertAlmostEqual(metrics.loc['Accuracy (exactitud)', 'Estimación'], accuracy)
        self.assertTrue((metrics['IC 95% inferior'] <= metrics['IC 95% superior']).all())

    def test_mismatched_persisted_evaluation_is_rejected(self):
        bundle = joblib.load(MODEL)
        bundle['evaluation']['modelo']['auroc'] = .99
        with self.assertRaisesRegex(ValueError, 'no reproducen'):
            recover_holdout(bundle, ANALYTICAL)

    def test_persisted_threshold_was_chosen_on_development(self):
        bundle = joblib.load(MODEL)
        info = bundle['threshold_info']
        self.assertEqual(info['criterio'], 'max_f1_fuera_de_pliegue_desarrollo')
        self.assertEqual(bundle['evaluation']['modelo']['umbral_usado'], bundle['threshold'])
        self.assertTrue(0.05 <= bundle['threshold'] <= 0.95)
        self.assertTrue(np.isfinite(bundle['development']['modelo']['brier_score']))


if __name__ == '__main__':
    unittest.main()
