"""CSV subido desde el panel de administración: validación de esquema y lectura sin tocar capas ni bitácora."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from config import LOCAL_SYNTHETIC_CSV
from src.data_layer.ingestion import REQUIRED_UPLOAD_COLUMNS, load_uploaded_csv, read_uploaded_csv, validate_uploaded_frame


def muestra(n: int = 300) -> pd.DataFrame:
    return read_uploaded_csv(LOCAL_SYNTHETIC_CSV).dropna(subset=["riesgo_oncologico"]).head(n).reset_index(drop=True)


class CsvSubidoTests(unittest.TestCase):
    def test_csv_con_el_esquema_correcto_no_tiene_problemas(self):
        self.assertEqual(validate_uploaded_frame(muestra()), [])

    def test_columnas_faltantes_se_nombran(self):
        problemas = validate_uploaded_frame(muestra().drop(columns=["edad", "imc"]))
        self.assertEqual(len(problemas), 1)
        self.assertIn("edad", problemas[0])
        self.assertIn("imc", problemas[0])

    def test_objetivo_solo_admite_cero_o_uno(self):
        df = muestra()
        df.loc[0, "riesgo_oncologico"] = 7
        self.assertIn("solo admite 0, 1 o vacío", validate_uploaded_frame(df)[0])

    def test_pocos_registros_o_una_sola_clase(self):
        self.assertTrue(validate_uploaded_frame(muestra(40)))
        una_clase = muestra()
        una_clase["riesgo_oncologico"] = 0
        self.assertTrue(any("= 1" in p for p in validate_uploaded_frame(una_clase)))

    def test_nombres_de_columna_se_normalizan(self):
        df = muestra().rename(columns={"edad": " Edad ", "sexo": "SEXO"})
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        leido = read_uploaded_csv(io.StringIO(buffer.getvalue()))
        self.assertTrue(set(REQUIRED_UPLOAD_COLUMNS) <= set(leido.columns))

    def test_carga_registra_origen_y_no_asume_que_es_sintetico(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "mis_datos.csv"
            muestra().to_csv(ruta, index=False)
            with mock.patch("src.data_layer.ingestion.log_event") as log:
                df, meta = load_uploaded_csv(ruta, "mis_datos.csv")
        self.assertEqual(len(df), 300)
        self.assertEqual(meta["origen"], "csv_cargado:mis_datos.csv")
        self.assertIsNone(meta["es_sintetico"])
        self.assertEqual(len(meta["sha256"]), 64)
        log.assert_called_once()

    def test_csv_invalido_se_rechaza_con_mensaje(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "malo.csv"
            muestra().drop(columns=["riesgo_oncologico"]).to_csv(ruta, index=False)
            with mock.patch("src.data_layer.ingestion.log_event"):
                with self.assertRaisesRegex(ValueError, "El CSV no se puede usar"):
                    load_uploaded_csv(ruta)


if __name__ == "__main__":
    unittest.main()
