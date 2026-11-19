"""Comprobaciones independientes de estimadores y estados límite."""
import unittest
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analytics import statistics as s
from src.analytics.data import load_patient_data


class StatisticalTests(unittest.TestCase):
    def test_wilson_known_cohort_and_boundaries(self):
        lo, hi = s.wilson_interval(4988, 20000)
        self.assertAlmostEqual(lo, .243452175894, places=11)
        self.assertAlmostEqual(hi, .255444072577, places=11)
        self.assertTrue(np.isnan(s.wilson_interval(0, 0)[0]))
        self.assertAlmostEqual(s.wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(s.wilson_interval(10, 10)[1], 1)

    def test_bh_matches_independent_scipy(self):
        p = np.array([.03, .001, .6, .009, .2])
        np.testing.assert_allclose(s.benjamini_hochberg(p), scipy_stats.false_discovery_control(p))
        self.assertTrue(np.isnan(s.benjamini_hochberg([np.nan])[0]))

    def test_numeric_summary_and_welch_match_scipy(self):
        d = pd.DataFrame({'edad': [20, 30, 40, 41, 52, 63, np.nan], 'riesgo_oncologico': [0, 0, 0, 1, 1, 1, 1]})
        result = s.numeric_summary(d).iloc[0]
        self.assertEqual(result.n, 6)
        self.assertEqual(result.ausentes, 1)
        self.assertAlmostEqual(result.desv_est, np.std(d.edad.dropna(), ddof=1))
        comparison = s.compare_numeric(d).iloc[0]
        independent = scipy_stats.ttest_ind([41, 52, 63], [20, 30, 40], equal_var=False)
        self.assertAlmostEqual(comparison.p_welch, independent.pvalue)
        self.assertGreater(comparison.diferencia_media, 0)

    def test_odds_reference_orientation_and_zero_correction(self):
        d = pd.DataFrame({'sexo': ['M'] * 4 + ['F'] * 8, 'riesgo_oncologico': [1, 1, 1, 0] + [1, 1, 0, 0, 0, 0, 0, 0]})
        row = s.odds_ratios(d).iloc[0]
        self.assertEqual(row.referencia, 'F')
        self.assertAlmostEqual(row['or'], 9)
        self.assertLess(row.ic_inf, row['or'])
        self.assertGreater(row.ic_sup, row['or'])
        d.loc[d.sexo.eq('M'), 'riesgo_oncologico'] = 1
        self.assertIn('0,5', s.odds_ratios(d).iloc[0].correccion)

    def test_normality_sample_size_and_constant(self):
        rng = np.random.default_rng(7)
        d = pd.DataFrame({'edad': rng.normal(50, 10, size=6000), 'imc': np.ones(6000)})
        result = s.normality_table(d).set_index('variable')
        self.assertEqual(result.loc['edad', 'n_prueba'], 5000)
        self.assertTrue(np.isnan(result.loc['imc', 'p']))
        self.assertTrue(s.density_curve([1, 1, 1]).empty)

    def test_time_windows_use_patient_weights_and_preserve_gaps(self):
        d = pd.DataFrame({'periodo': ['2023-S1'] + ['2023-S2'] * 3 + ['2024-S2'], 'edad': [20, 40, 40, 40, 80]})
        result = s.temporal_summary(d, 'edad').set_index('periodo')
        self.assertEqual(result.loc['2023-S2', 'media_movil'], 35)
        self.assertTrue(np.isnan(result.loc['2024-S1', 'media']))
        self.assertTrue(np.isnan(result.loc['2024-S2', 'media_movil']))
        self.assertNotEqual(result.loc['2023-S2', 'tramo'], result.loc['2024-S2', 'tramo'])

    def test_quality_reconciles_known_generator_issue(self):
        d, _ = load_patient_data()
        quality = s.quality_report(d)['summary']
        self.assertEqual(quality['rows'], 20000)
        self.assertEqual(quality['contradictions'], 3567)
        self.assertEqual(quality['missing_cells'], 0)
        self.assertEqual(quality['duplicate_ids'], 0)

    def test_empty_statistics_and_small_pca(self):
        empty = pd.DataFrame(columns=['edad', 'imc', 'riesgo_oncologico'])
        self.assertEqual(s.numeric_summary(empty).n.sum(), 0)
        self.assertTrue(s.odds_ratios(empty).empty)
        self.assertEqual(s.pca_analysis(empty)['status'], 'Se requieren al menos tres casos completos y dos variables no constantes.')


if __name__ == '__main__':
    unittest.main()
