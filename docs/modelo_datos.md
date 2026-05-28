# Modelo de datos

## Entradas requeridas

`load_data.py` espera 6 tablas de entrada:

- `siniestros`
- `polizas`
- `asegurados`
- `beneficiarios_proveedores`
- `documentos`
- `vehiculos`

En pruebas locales se leen desde `data/raw/`. En backend/API se puede pasar un `file_map` con archivos recibidos desde el front o DataFrames ya construidos.

## Diccionario esperado por build_features

```python
tables = {
    "siniestros": pd.DataFrame,
    "polizas": pd.DataFrame,
    "asegurados": pd.DataFrame,
    "documentos": pd.DataFrame,
    "beneficiarios_proveedores": pd.DataFrame,
    "vehiculos": pd.DataFrame,
}
```

## Tabla final PostgreSQL

La salida final se guarda en:

```sql
fraud_ia.siniestros_scored_final
```

Llave primaria:

```sql
id_siniestro
```

La carga usa upsert por `id_siniestro`, evitando duplicados cuando se reprocesan datos.

## Campos principales de salida

- Identificacion: `id_siniestro`, `id_poliza`, `id_asegurado`, `id_proveedor`, `id_vehiculo`.
- Datos de negocio: ramo, cobertura, fechas, montos, estado, ciudades y descripcion.
- Features de riesgo: frecuencias, documentos, similitud textual y scores por senal.
- Modelo ML: `probabilidad_ml`, `prediccion_ml`.
- Resultado hibrido: `score_final`, `semaforo_final`.
- Explicabilidad: `reglas_criticas_activadas`, `alertas_score_activadas`, `explicabilidad`.
- NLP: `ids_siniestros_similares_top5`, `max_similitud_textual`.

Los campos tipo lista se guardan en PostgreSQL como `JSONB`.
