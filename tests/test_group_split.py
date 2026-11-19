import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.modeling.train import make_train_test_split

class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'id_paciente': np.repeat(np.arange(50), 2),
                                'edad': np.arange(100) + 18,
                                'riesgo_oncologico': np.repeat(np.arange(50) % 2, 2)})

    def test_pacientes_no_compartidos_en_prueba_ni_pliegues(self):
        split = make_train_test_split(self.df)
        train_ids = set(self.df.loc[split.X_train.index, 'id_paciente'])
        test_ids = set(self.df.loc[split.X_test.index, 'id_paciente'])
        self.assertFalse(train_ids & test_ids)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for train, valid in cv.split(split.X_train, split.y_train, split.groups_train):
            self.assertFalse(set(split.groups_train.iloc[train]) & set(split.groups_train.iloc[valid]))

    def test_rechaza_clases_inconsistentes(self):
        self.df.loc[1, 'riesgo_oncologico'] = 1
        with self.assertRaises(ValueError):
            make_train_test_split(self.df)

    def test_rechaza_id_faltante(self):
        self.df.loc[0, 'id_paciente'] = None
        with self.assertRaises(ValueError):
            make_train_test_split(self.df)