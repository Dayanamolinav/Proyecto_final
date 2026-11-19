"""La nueva visualización debe reproducir la prueba persistida, no entrenar."""
import unittest
from pathlib import Path
import joblib

from src.analytics.model_diagnostics import recover_holdout, diagnostic_results

ROOT = Path(__file__).resolve().parents[1]


class ModelDiagnosticTests(unittest.TestCase):
    def test_reconstructs_original_holdout_and_known_metrics(self):
        bundle = joblib.load(ROOT / 'models' / 'modelo_seleccionado.joblib')
        holdout = recover_holdout(bundle, ROOT / 'data' / 'analytical' / 'clinical_analytical.parquet')
        self.assertEqual(len(holdout), 240)
        self.assertEqual(holdout.id_paciente.nunique(), 240)
        self.assertEqual(int(holdout.etiqueta.sum()), 58)
        results = diagnostic_results(holdout, resamples=20)
        self.assertEqual(results['matrix'].Pacientes.tolist(), [119, 63, 32, 26])
        metrics = results['metrics'].set_index('Métrica')
        self.assertAlmostEqual(metrics.loc['ROC-AUC', 'Estimación'], bundle['evaluation']['modelo']['auroc'], places=4)
        self.assertAlmostEqual(metrics.loc['Accuracy (exactitud)', 'Estimación'], 145/240)
        self.assertTrue((metrics['IC 95% inferior'] <= metrics['IC 95% superior']).all())

    def test_mismatched_persisted_evaluation_is_rejected(self):
        bundle = joblib.load(ROOT / 'models' / 'modelo_seleccionado.joblib')
        bundle['evaluation']['modelo']['auroc'] = .99
        with self.assertRaisesRegex(ValueError, 'no reproducen'):
            recover_holdout(bundle, ROOT / 'data' / 'analytical' / 'clinical_analytical.parquet')


if __name__ == '__main__':
    unittest.main()
