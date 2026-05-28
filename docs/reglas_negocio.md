# Reglas de negocio

Las reglas estan implementadas en `scripts/rules/fraud_rules.py` y se basan en las senales de riesgo y reglas criticas del reto.

## Score por senales

`build_features.py` calcula scores parciales para:

- Reclamo cercano al inicio o fin de vigencia.
- Demora en denuncia de robo.
- Alta frecuencia de reclamos por asegurado, vehiculo o conductor.
- Reclamos solo RC.
- Proveedor recurrente u observado.
- Documentos incompletos o inconsistentes.
- Dinamica sospechosa del accidente.
- Evento sin tercero identificado.
- Reporte tardio.
- Monto cercano a suma asegurada o atipico.
- Narrativas similares.

La suma se guarda en `score_total_reglas` y se limita a 100 puntos.

## Reglas criticas

`fraud_rules.py` agrega reglas RF:

- `RF_01_perdida_total_robo`
- `RF_02_adulteracion_doc`
- `RF_03_lista_restrictiva`
- `RF_04_dinamica_imposible`
- `RF_05_borde_vigencia_48h`
- `RF_06_demora_robo_4dias`
- `RF_07_narrativa_clonada`
- `RF_08_score_reglas_alto`
- `RF_09_score_alto_y_ml_riesgo`
- `RF_10_documental_multiple`
- `RF_11_proveedor_recurrente_monto_atipico`
- `RF_12_alta_frecuencia_y_borde_vigencia`

Algunas reglas se calculan dos veces en el pipeline: primero antes del modelo para crear features RF de entrada, y luego despues de la prediccion ML para activar reglas dependientes de `probabilidad_ml` o `prediccion_ml`.

## Score final

El score final combina reglas y ML:

```text
score_final = 70% score_total_reglas + 30% (probabilidad_ml * 100)
```

## Semaforo final

- `0 - 40`: Verde.
- `41 - 75`: Amarillo.
- `76 - 100`: Rojo.

Luego se ajusta por reglas criticas:

- Si una regla critica roja se activa, el semaforo final queda Rojo.
- Si una regla amarilla se activa, o el modelo predice riesgo con score final mayor o igual a 41, queda al menos Amarillo.
