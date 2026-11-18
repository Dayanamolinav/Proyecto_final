"""
Capa de ingesta (Sección VII-A.2 y VII-B.2 del informe).

En producción, esta etapa recibiría el archivo CSV entregado por el hospital
universitario a través de un Stage de Snowflake [18]. En este prototipo, al
no disponer todavía de ese archivo, `load_source_file()` admite dos modos:

  1. Si se indica `csv_path`, se ingiere ese archivo CSV real (así, cuando el
     hospital entregue los datos, basta con apuntar a ese archivo).
  2. Si no se indica ninguno, se genera el conjunto sintético de prueba
     (`generate_synthetic_clinical_data`) y se persiste como si fuera el
     archivo recién cargado, dejando constancia en la bitácora de que el
     origen es sintético (no se debe confundir con datos reales de pacientes).

Toda carga registra metadata de origen y momento (trazabilidad), como exige
el informe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RAW_DIR, SyntheticDataConfig
from src.data_layer.synthetic_data import generate_synthetic_clinical_data
from src.data_layer.traceability import TraceEvent, log_event


def load_source_file(csv_path: str | Path | None = None, synth_cfg: SyntheticDataConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Ingiere el archivo fuente y devuelve (DataFrame, metadata_de_carga)."""
    if csv_path is not None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        source_label = f"archivo_real:{csv_path.name}"
        is_synthetic = False
    else:
        df = generate_synthetic_clinical_data(synth_cfg)
        source_label = "generador_sintetico_pruebas"
        is_synthetic = True

    metadata = {
        "origen": source_label,
        "es_sintetico": is_synthetic,
        "n_registros_iniciales": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
    }

    log_event(
        TraceEvent(
            stage="INGESTA",
            action="carga_archivo_fuente",
            details=metadata,
            rows_affected=len(df),
        )
    )
    return df, metadata


def persist_raw(df: pd.DataFrame, filename: str = "clinical_raw.parquet") -> Path:
    """Persiste el conjunto tal como fue recibido en la capa RAW.

    RAW conserva los datos como fueron recibidos, UNA VEZ retirados o
    sustituidos los identificadores personales directos (ver Sección VII-B.3:
    "en RAW se conservarán los datos recibidos sin identificadores personales
    directos"). Aquí sustituimos esos identificadores por un hash
    seudonimizado antes de escribir en RAW.
    """
    from src.data_layer.storage import pseudonymize_direct_identifiers

    raw_df, pseudo_report = pseudonymize_direct_identifiers(df)
    out_path = RAW_DIR / filename
    raw_df.to_parquet(out_path, index=False)

    log_event(
        TraceEvent(
            stage="RAW",
            action="persistencia_raw_seudonimizada",
            details={"archivo": str(out_path), **pseudo_report},
            rows_affected=len(raw_df),
        )
    )
    return out_path
