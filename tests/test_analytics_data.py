"""Contrato del CSV y conservación de la granularidad del panel."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.analytics.data import DEFAULT_SOURCE, ORIGINAL_COLUMNS, load_patient_data


class AnalyticsDataTests(unittest.TestCase):
    def test_distributed_source_and_derived_boundaries(self):
        frame, meta = load_patient_data()
        self.assertEqual(len(frame), 20000)
        self.assertEqual(frame.id_paciente.nunique(), 20000)
        self.assertEqual(int(frame.riesgo_oncologico.sum()), 4988)
        self.assertEqual(meta['hash'], 'bf2078859176c2292618f8688944f8d14b2ab4688754c3bdd5ac87e4b2bc652a')
        self.assertEqual(frame[ORIGINAL_COLUMNS].isna().sum().sum(), 0)
        self.assertEqual(set(frame.grupo_etario), {'<40', '40-54', '55-69', '70+'})
        self.assertTrue(meta['independent_ids'])

    def test_missing_source_and_schema_are_explicit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.csv'
            with self.assertRaises(FileNotFoundError):
                load_patient_data(path)
            pd.DataFrame({'edad': [20]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Faltan columnas'):
                load_patient_data(path)

    def test_invalid_values_remain_inspectable_not_imputed(self):
        row = pd.read_csv(DEFAULT_SOURCE, nrows=1)
        row['edad'] = 'not_a_number'
        row['imc'] = 101
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.csv'
            row.to_csv(path, index=False)
            frame, meta = load_patient_data(path)
        self.assertEqual(len(frame), 1)
        self.assertTrue(pd.isna(frame.edad.iloc[0]))
        self.assertTrue(pd.isna(frame.grupo_etario.iloc[0]))
        self.assertTrue(pd.isna(frame.imc_categoria.iloc[0]))
        self.assertEqual(meta['numeric_coercions']['edad'], 1)


if __name__ == '__main__':
    unittest.main()
