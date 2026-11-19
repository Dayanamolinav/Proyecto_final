"""Casos de aceptación del formulario sintético; ejecutar con unittest."""
import unittest

from src.processing.input_validation import prepare_patient_input


def valid_row():
    return {"edad": 55, "sexo": "F", "imc": 26.5, "tabaquismo": "nunca",
            "antecedente_familiar_cancer": 0, "antecedente_personal_benigno": 0,
            "presion_sistolica": 125, "glucosa_mg_dl": 98, "paridad": 1,
            "ca125_u_ml": 18.0, "resultado_imagen": "normal"}


class PatientInputTests(unittest.TestCase):
    def test_valid_input(self):
        result = prepare_patient_input(valid_row())
        self.assertEqual(result.loc[0, "grupo_etario"], "55-69")
        self.assertEqual(result.loc[0, "imc_categoria"], "sobrepeso")

    def test_missing(self):
        row = valid_row()
        del row["edad"]
        with self.assertRaisesRegex(ValueError, "edad"):
            prepare_patient_input(row)

    def test_unknown_category(self):
        row = valid_row()
        row["tabaquismo"] = "otro"
        with self.assertRaisesRegex(ValueError, "tabaquismo"):
            prepare_patient_input(row)

    def test_bad_format(self):
        row = valid_row()
        row["edad"] = "abc"
        with self.assertRaisesRegex(ValueError, "edad"):
            prepare_patient_input(row)

    def test_out_of_range(self):
        row = valid_row()
        row["edad"] = 150
        with self.assertRaisesRegex(ValueError, "edad"):
            prepare_patient_input(row)

    def test_range_boundaries(self):
        for age in (18, 100):
            row = valid_row()
            row["edad"] = age
            self.assertEqual(prepare_patient_input(row).loc[0, "edad"], age)

    def test_nonfinite_and_fractional(self):
        for value in (float("nan"), float("inf"), 19.5):
            row = valid_row()
            row["edad"] = value
            with self.assertRaises(ValueError):
                prepare_patient_input(row)

    def test_age_group_boundaries(self):
        for age, group in ((39, "<40"), (40, "40-54"), (54, "40-54"),
                           (55, "55-69"), (69, "55-69"), (70, "70+")):
            row = valid_row()
            row["edad"] = age
            self.assertEqual(prepare_patient_input(row).loc[0, "grupo_etario"], group)

    def test_bmi_boundaries(self):
        for bmi, group in ((18.4, "bajo_peso"), (18.5, "normal"), (24.9, "normal"),
                           (25, "sobrepeso"), (29.9, "sobrepeso"), (30, "obesidad")):
            row = valid_row()
            row["imc"] = bmi
            self.assertEqual(prepare_patient_input(row).loc[0, "imc_categoria"], group)


if __name__ == "__main__":
    unittest.main()
