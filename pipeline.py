"""
Pipeline de análisis de comportamiento de usuarios en bibliotecas públicas colombianas.
Fuente: Portal de Datos Abiertos del Gobierno Colombiano (datos.gov.co) — API SODA.

Datasets utilizados:
  - Red Nacional de Bibliotecas Públicas   : https://www.datos.gov.co/resource/a5h9-fqe9.json
  - Red BiblioRed Bogotá D.C.              : https://www.datos.gov.co/resource/i7i7-vszn.json
  - Red Departamental de Bolívar           : https://www.datos.gov.co/resource/hx8b-s96z.json
  - Bibliotecas Públicas de Casanare       : https://www.datos.gov.co/resource/fwr7-rti6.json
"""

import sys
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SODA_BASE = "https://www.datos.gov.co/resource/{dataset_id}.json"
PAGE_SIZE = 1000

DATASETS = {
    "red_nacional": {
        "id": "a5h9-fqe9",
        "url": "https://www.datos.gov.co/resource/a5h9-fqe9.json",
        "description": "Directorio de la Red Nacional de Bibliotecas Públicas",
    },
    "bibliored": {
        "id": "i7i7-vszn",
        "url": "https://www.datos.gov.co/resource/i7i7-vszn.json",
        "description": "Red Biblioteca Pública BiblioRed — Bogotá D.C.",
    },
    "bolivar": {
        "id": "hx8b-s96z",
        "url": "https://www.datos.gov.co/resource/hx8b-s96z.json",
        "description": "Red Departamental de Bibliotecas Públicas de Bolívar",
    },
    "casanare": {
        "id": "fwr7-rti6",
        "url": "https://www.datos.gov.co/resource/fwr7-rti6.json",
        "description": "Bibliotecas Públicas del Departamento de Casanare",
    },
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. INGESTA — extracción paginada desde la API SODA
# ---------------------------------------------------------------------------


def fetch_dataset(dataset_id: str, description: str) -> pd.DataFrame:
    """
    Descarga todas las filas de un dataset SODA usando paginación
    con $limit y $offset para no superar el límite por request.
    """
    url = SODA_BASE.format(dataset_id=dataset_id)
    rows = []
    offset = 0

    log.info(f"Descargando '{description}' (id={dataset_id}) ...")

    while True:
        try:
            response = requests.get(
                url,
                params={"$limit": PAGE_SIZE, "$offset": offset},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            log.error(f"  Error en request offset={offset}: {e}")
            break

        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        log.info(f"  offset={offset} → {len(batch)} filas recibidas (total acumulado: {len(rows)})")
        offset += PAGE_SIZE

        if len(batch) < PAGE_SIZE:
            break

    df = pd.DataFrame(rows)
    log.info(f"  ✓ {len(df)} filas totales | {len(df.columns)} columnas")
    return df


def ingest_all() -> dict[str, pd.DataFrame]:
    """Descarga los cuatro datasets y los retorna como diccionario."""
    raw = {}
    for name, meta in DATASETS.items():
        df = fetch_dataset(meta["id"], meta["description"])
        raw[name] = df
    return raw


# ---------------------------------------------------------------------------
# 2. LIMPIEZA — normalización por dataset
# ---------------------------------------------------------------------------


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte nombres de columnas a snake_case sin tildes ni caracteres especiales.
    Ejemplo: 'Municipio / Ciudad' → 'municipio_ciudad'
    """
    df.columns = (
        df.columns
        .str.strip()
        .map(lambda c: unidecode(c).lower().replace(" ", "_").replace("/", "_")
             .replace("-", "_").replace("(", "").replace(")", "").replace(".", "")
             .replace(",", "").replace("__", "_").strip("_"))
    )
    return df


def cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convierte columnas a numérico; los no-convertibles quedan como NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_departamento(valor: str) -> str:
    """Estandariza nombres de departamento: sin tildes, capitalizado."""
    if pd.isna(valor):
        return "Sin dato"
    return unidecode(str(valor)).strip().title()


def clean_red_nacional(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para el dataset de la Red Nacional."""
    df = normalize_column_names(df)

    rename_map = {
        "nombre_de_la_biblioteca": "nombre_biblioteca",
        "nombre_municipio": "municipio",
        "nombre_departamento": "departamento",
        "tipo_de_biblioteca": "tipo_biblioteca",
        "nombre_de_la_red": "red_pertenece",
        "direccion": "direccion",
        "telefono": "telefono",
        "correo_electronico": "correo",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["fuente"] = "red_nacional"
    df["departamento"] = df["departamento"].map(normalize_departamento) if "departamento" in df.columns else "Sin dato"
    df["municipio"] = df["municipio"].str.strip().str.title() if "municipio" in df.columns else "Sin dato"

    keep = ["nombre_biblioteca", "municipio", "departamento", "tipo_biblioteca",
            "red_pertenece", "direccion", "telefono", "correo", "fuente"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates()


def clean_bibliored(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para BiblioRed Bogotá."""
    df = normalize_column_names(df)

    rename_map = {
        "nombre": "nombre_biblioteca",
        "localidad": "municipio",
        "direccion": "direccion",
        "tipo": "tipo_biblioteca",
        "horario": "horario",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["fuente"] = "bibliored_bogota"
    df["departamento"] = "Cundinamarca"
    df["municipio"] = "Bogota"

    keep = ["nombre_biblioteca", "municipio", "departamento", "tipo_biblioteca",
            "direccion", "horario", "fuente"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates()


def clean_bolivar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para Bolívar."""
    df = normalize_column_names(df)

    rename_map = {
        "nombre_biblioteca": "nombre_biblioteca",
        "municipio": "municipio",
        "nombre_red": "red_pertenece",
        "estado": "estado_operacion",
        "conectividad": "tiene_internet",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["fuente"] = "bolivar"
    df["departamento"] = "Bolivar"
    df["municipio"] = df["municipio"].str.strip().str.title() if "municipio" in df.columns else "Sin dato"

    keep = ["nombre_biblioteca", "municipio", "departamento", "red_pertenece",
            "estado_operacion", "tiene_internet", "fuente"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates()


def clean_casanare(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para Casanare."""
    df = normalize_column_names(df)

    rename_map = {
        "nombre": "nombre_biblioteca",
        "municipio": "municipio",
        "horario_atencion": "horario",
        "numero_de_usuarios": "usuarios_registrados",
        "volumen_coleccion": "volumen_coleccion",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["fuente"] = "casanare"
    df["departamento"] = "Casanare"
    df["municipio"] = df["municipio"].str.strip().str.title() if "municipio" in df.columns else "Sin dato"

    numeric_cols = ["usuarios_registrados", "volumen_coleccion"]
    df = cast_numeric(df, numeric_cols)

    keep = ["nombre_biblioteca", "municipio", "departamento", "horario",
            "usuarios_registrados", "volumen_coleccion", "fuente"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates()


CLEANERS = {
    "red_nacional": clean_red_nacional,
    "bibliored":    clean_bibliored,
    "bolivar":      clean_bolivar,
    "casanare":     clean_casanare,
}


def clean_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aplica el limpiador correspondiente a cada dataset."""
    cleaned = {}
    for name, df in raw.items():
        log.info(f"Limpiando dataset '{name}' ...")
        cleaner = CLEANERS.get(name)
        if cleaner:
            cleaned[name] = cleaner(df.copy())
            log.info(f"  ✓ {len(cleaned[name])} registros limpios")
        else:
            log.warning(f"  No hay limpiador definido para '{name}', se omite.")
    return cleaned


# ---------------------------------------------------------------------------
# 3. INTEGRACIÓN — unión en un dataset maestro
# ---------------------------------------------------------------------------

MASTER_COLUMNS = [
    "nombre_biblioteca",
    "municipio",
    "departamento",
    "tipo_biblioteca",
    "red_pertenece",
    "direccion",
    "horario",
    "telefono",
    "correo",
    "estado_operacion",
    "tiene_internet",
    "usuarios_registrados",
    "volumen_coleccion",
    "fuente",
]


def integrate(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatena todos los datasets limpios en una tabla maestra.
    Columnas faltantes se rellenan con NaN para mantener estructura uniforme.
    """
    log.info("Integrando datasets en tabla maestra ...")
    frames = list(cleaned.values())
    master = pd.concat(frames, ignore_index=True, sort=False)

    for col in MASTER_COLUMNS:
        if col not in master.columns:
            master[col] = np.nan

    master = master[MASTER_COLUMNS]
    master = master.drop_duplicates(subset=["nombre_biblioteca", "municipio", "departamento"])
    master = master.reset_index(drop=True)

    log.info(f"  ✓ Tabla maestra: {len(master)} registros | {len(master.columns)} columnas")
    return master


# ---------------------------------------------------------------------------
# 4. ENRIQUECIMIENTO — métricas y columnas derivadas
# ---------------------------------------------------------------------------

REGION_MAP = {
    "Amazonas": "Amazonia", "Caqueta": "Amazonia", "Guainia": "Amazonia",
    "Guaviare": "Amazonia", "Putumayo": "Amazonia", "Vaupes": "Amazonia",
    "Vichada": "Amazonia",
    "Antioquia": "Andina", "Boyaca": "Andina", "Caldas": "Andina",
    "Cundinamarca": "Andina", "Huila": "Andina", "Narino": "Andina",
    "Norte De Santander": "Andina", "Quindio": "Andina", "Risaralda": "Andina",
    "Santander": "Andina", "Tolima": "Andina",
    "Bolivar": "Caribe", "Atlantico": "Caribe", "Cesar": "Caribe",
    "Cordoba": "Caribe", "La Guajira": "Caribe", "Magdalena": "Caribe",
    "San Andres Y Providencia": "Caribe", "Sucre": "Caribe",
    "Choco": "Pacifico", "Cauca": "Pacifico", "Valle Del Cauca": "Pacifico",
    "Arauca": "Orinoquia", "Casanare": "Orinoquia", "Meta": "Orinoquia",
}


def enrich(master: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas útiles para análisis."""
    log.info("Enriqueciendo tabla maestra ...")

    master["region_natural"] = master["departamento"].map(REGION_MAP).fillna("Sin clasificar")

    if "tiene_internet" in master.columns:
        master["tiene_internet_flag"] = (
            master["tiene_internet"]
            .astype(str).str.lower()
            .isin(["si", "sí", "true", "1", "yes"])
        )
    else:
        master["tiene_internet_flag"] = np.nan

    if master["volumen_coleccion"].notna().any():
        master["categoria_coleccion"] = pd.cut(
            master["volumen_coleccion"],
            bins=[0, 500, 2000, 10000, float("inf")],
            labels=["Pequeña (<500)", "Mediana (500-2k)", "Grande (2k-10k)", "Muy grande (>10k)"],
            right=True,
        )
    else:
        master["categoria_coleccion"] = np.nan

    log.info("  ✓ Enriquecimiento completado")
    return master


def export(master: pd.DataFrame) -> None:
    """Guarda el dataset final en CSV y Excel con metadatos de auditoría."""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / f"bibliotecas_colombia_final_{timestamp}.csv"
    xlsx_path = OUTPUT_DIR / f"bibliotecas_colombia_final_{timestamp}.xlsx"

    master.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"  CSV guardado: {csv_path}")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Bibliotecas", index=False)

        resumen = (
            master.groupby("departamento", dropna=False)
            .agg(
                total_bibliotecas=("nombre_biblioteca", "count"),
                con_internet=("tiene_internet_flag", lambda x: x.sum() if x.notna().any() else np.nan),
                usuarios_promedio=("usuarios_registrados", "mean"),
                coleccion_promedio=("volumen_coleccion", "mean"),
            )
            .reset_index()
        )
        resumen.to_excel(writer, sheet_name="Resumen_por_departamento", index=False)

        region = (
            master.groupby("region_natural", dropna=False)
            .agg(
                total_bibliotecas=("nombre_biblioteca", "count"),
                departamentos=("departamento", "nunique"),
            )
            .reset_index()
        )
        region.to_excel(writer, sheet_name="Resumen_por_region", index=False)

    log.info(f"  Excel guardado: {xlsx_path}")

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL DATASET FINAL")
    print("=" * 60)
    print(f"Total registros          : {len(master):,}")
    print(f"Total departamentos      : {master['departamento'].nunique()}")
    print(f"Total municipios         : {master['municipio'].nunique()}")
    print(f"Fuentes integradas       : {master['fuente'].nunique()}")
    print(f"Con datos de usuarios    : {master['usuarios_registrados'].notna().sum()}")
    print(f"Con datos de colección   : {master['volumen_coleccion'].notna().sum()}")
    print("\nRegistros por fuente:")
    print(master["fuente"].value_counts().to_string())
    print("\nRegistros por región natural:")
    print(master["region_natural"].value_counts().to_string())
    print("=" * 60)


def run_pipeline() -> pd.DataFrame:
    """Ejecuta el pipeline completo de extremo a extremo."""
    log.info("=" * 60)
    log.info("Iniciando pipeline — Bibliotecas Públicas Colombia")
    log.info("=" * 60)

    raw = ingest_all()
    cleaned = clean_all(raw)
    master = integrate(cleaned)
    master = enrich(master)
    export(master)

    log.info("Pipeline completado exitosamente.")
    return master


if __name__ == "__main__":
    df_final = run_pipeline()
