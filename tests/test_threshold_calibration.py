"""Umbral elegido en desarrollo y modelos sin pesos de clase (decisiones_modelado.md)."""
import unittest

import numpy as np
import pandas as pd

from src.modeling.evaluate import development_summary, evaluate_against_null_model, select_threshold
from src.modeling.train import build_candidate_models, fit_and_cross_validate, make_train_test_split


class ThresholdTests(unittest.TestCase):
    def test_elige_el_umbral_que_maximiza_f1(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([.10, .20, .30, .40, .35, .90])
        result = select_threshold(y, p, grid=[.25, .35, .50])
        self.assertEqual(result['umbral'], .35)  # en .35: TP=2, FP=1 -> F1 0,8
        self.assertEqual(result['criterio'], 'max_f1_fuera_de_pliegue_desarrollo')

    def test_en_empate_conserva_el_umbral_mas_bajo(self):
        y = np.array([0, 1])
        p = np.array([.1, .9])
        self.assertEqual(select_threshold(y, p, grid=[.2, .5, .8])['umbral'], .2)

    def test_rechaza_predicciones_faltantes(self):
        with self.assertRaises(ValueError):
            select_threshold(np.array([0, 1]), np.array([np.nan, .5]))

    def test_el_umbral_se_aplica_al_modelo_y_al_nulo(self):
        y_train = pd.Series([0, 0, 0, 1])
        result = evaluate_against_null_model(y_train, np.array([0, 1, 0, 1]), np.array([.1, .3, .2, .9]), threshold=.3)
        self.assertEqual(result['modelo']['umbral_usado'], .3)
        self.assertEqual(result['modelo']['n_verdaderos_positivos'], 2)
        self.assertEqual(result['modelo_nulo']['umbral_usado'], .3)

    def test_resumen_de_desarrollo_compara_contra_nulo(self):
        y = pd.Series([0, 0, 1, 1])
        summary = development_summary(y, np.array([.1, .2, .8, .9]), .5)
        self.assertTrue(summary['brier_mejor_que_nulo'])


class CandidateModelTests(unittest.TestCase):
    def test_sin_pesos_de_clase_por_defecto(self):
        y = pd.Series([0] * 30 + [1] * 10)
        models = build_candidate_models(y)
        self.assertIsNone(models['logistic_regression'].class_weight)
        self.assertIsNone(models['random_forest'].class_weight)
        if 'xgboost' in models:
            self.assertEqual(models['xgboost'].get_params()['scale_pos_weight'], 1.0)

    def test_pesos_disponibles_si_se_activan(self):
        y = pd.Series([0] * 30 + [1] * 10)
        models = build_candidate_models(y, use_class_weights=True)
        self.assertEqual(models['logistic_regression'].class_weight, {0: 1.0, 1: 3.0})

    def test_validacion_cruzada_entrega_predicciones_fuera_de_pliegue(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            'id_paciente': [f'P{i}' for i in range(n)],
            'edad': rng.normal(55, 10, n), 'imc': rng.normal(27, 4, n),
            'sexo': rng.choice(['F', 'M'], n), 'tabaquismo': rng.choice(['nunca', 'activo'], n),
        })
        df['riesgo_oncologico'] = (df.edad + rng.normal(0, 8, n) > 58).astype(int)
        split = make_train_test_split(df)
        results = fit_and_cross_validate(split)
        for result in results.values():
            self.assertEqual(len(result['oof_proba']), len(split.y_train))
            self.assertFalse(np.isnan(result['oof_proba']).any())
            self.assertEqual(len(result['cv_scores']), 5)


if __name__ == '__main__':
    unittest.main()
