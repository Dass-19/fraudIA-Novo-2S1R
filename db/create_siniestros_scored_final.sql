-- PostgreSQL DDL for the final processed claims table.
-- Stores processed, classified, scored and explained insurance claims.
-- The output is an alert for human review, not an automatic fraud accusation.

CREATE SCHEMA IF NOT EXISTS fraud_ia;

CREATE TABLE IF NOT EXISTS fraud_ia.siniestros_scored_final (
    id_siniestro BIGINT PRIMARY KEY,
    id_poliza BIGINT,
    id_asegurado BIGINT,
    id_proveedor BIGINT,
    id_vehiculo BIGINT,

    ramo VARCHAR(80),
    cobertura VARCHAR(120),
    fecha_ocurrencia DATE,
    fecha_reporte DATE,

    monto_reclamado NUMERIC(14, 2),
    monto_estimado NUMERIC(14, 2),
    monto_pagado NUMERIC(14, 2),
    suma_asegurada NUMERIC(14, 2),

    estado VARCHAR(80),
    sucursal VARCHAR(120),
    ciudad_poliza VARCHAR(120),
    ciudad_asegurado VARCHAR(120),
    ciudad_proveedor VARCHAR(120),
    descripcion TEXT,

    documentos_completos BOOLEAN,
    docs_faltantes BOOLEAN,
    docs_inconsistentes BOOLEAN,

    freq_asegurado_18m INTEGER DEFAULT 0,
    freq_vehiculo_18m INTEGER DEFAULT 0,
    freq_conductor_18m INTEGER DEFAULT 0,

    max_similitud_textual NUMERIC(8, 6),
    ids_siniestros_similares_top5 JSONB DEFAULT '[]'::jsonb,

    score_reclamo_vigencia INTEGER DEFAULT 0,
    score_demora_robo INTEGER DEFAULT 0,
    score_freq_asegurado INTEGER DEFAULT 0,
    score_freq_vehiculo INTEGER DEFAULT 0,
    score_freq_conductor INTEGER DEFAULT 0,
    score_rc_only INTEGER DEFAULT 0,
    score_proveedor INTEGER DEFAULT 0,
    score_docs_incompletos INTEGER DEFAULT 0,
    score_docs_inconsistentes INTEGER DEFAULT 0,
    score_dinamica_sospechosa INTEGER DEFAULT 0,
    score_sin_tercero INTEGER DEFAULT 0,
    score_reporte_tardio INTEGER DEFAULT 0,
    score_monto_suma_asegurada INTEGER DEFAULT 0,
    score_narrativas_similares INTEGER DEFAULT 0,
    score_total_reglas INTEGER DEFAULT 0,

    semaforo_score VARCHAR(20),
    probabilidad_ml NUMERIC(8, 6),
    prediccion_ml SMALLINT,
    score_final NUMERIC(8, 2),
    semaforo_score_final VARCHAR(20),
    semaforo_reglas_criticas VARCHAR(20),
    semaforo_final VARCHAR(20),

    rf_01_perdida_total_robo BOOLEAN DEFAULT FALSE,
    rf_02_adulteracion_doc BOOLEAN DEFAULT FALSE,
    rf_03_lista_restrictiva BOOLEAN DEFAULT FALSE,
    rf_04_dinamica_imposible BOOLEAN DEFAULT FALSE,
    rf_05_borde_vigencia_48h BOOLEAN DEFAULT FALSE,
    rf_06_demora_robo_4dias BOOLEAN DEFAULT FALSE,
    rf_07_narrativa_clonada BOOLEAN DEFAULT FALSE,
    rf_08_score_reglas_alto BOOLEAN DEFAULT FALSE,
    rf_09_score_alto_y_ml_riesgo BOOLEAN DEFAULT FALSE,
    rf_10_documental_multiple BOOLEAN DEFAULT FALSE,
    rf_11_proveedor_recurrente_monto_atipico BOOLEAN DEFAULT FALSE,
    rf_12_alta_frecuencia_y_borde_vigencia BOOLEAN DEFAULT FALSE,

    reglas_criticas_activadas JSONB DEFAULT '[]'::jsonb,
    alertas_score_activadas JSONB DEFAULT '[]'::jsonb,
    explicabilidad TEXT,

    etiqueta_fraude_simulada SMALLINT,

    CONSTRAINT chk_prediccion_ml CHECK (prediccion_ml IS NULL OR prediccion_ml IN (0, 1)),
    CONSTRAINT chk_etiqueta_fraude_simulada CHECK (etiqueta_fraude_simulada IS NULL OR etiqueta_fraude_simulada IN (0, 1)),
    CONSTRAINT chk_probabilidad_ml CHECK (probabilidad_ml IS NULL OR (probabilidad_ml >= 0 AND probabilidad_ml <= 1)),
    CONSTRAINT chk_score_final CHECK (score_final IS NULL OR (score_final >= 0 AND score_final <= 100)),
    CONSTRAINT chk_semaforo_score CHECK (semaforo_score IS NULL OR semaforo_score IN ('Verde', 'Amarillo', 'Rojo')),
    CONSTRAINT chk_semaforo_score_final CHECK (semaforo_score_final IS NULL OR semaforo_score_final IN ('Verde', 'Amarillo', 'Rojo')),
    CONSTRAINT chk_semaforo_reglas_criticas CHECK (semaforo_reglas_criticas IS NULL OR semaforo_reglas_criticas IN ('Verde', 'Amarillo', 'Rojo')),
    CONSTRAINT chk_semaforo_final CHECK (semaforo_final IS NULL OR semaforo_final IN ('Verde', 'Amarillo', 'Rojo')),
    CONSTRAINT chk_json_ids_similares_array CHECK (jsonb_typeof(ids_siniestros_similares_top5) = 'array'),
    CONSTRAINT chk_json_reglas_array CHECK (jsonb_typeof(reglas_criticas_activadas) = 'array'),
    CONSTRAINT chk_json_alertas_array CHECK (jsonb_typeof(alertas_score_activadas) = 'array')
);

COMMENT ON TABLE fraud_ia.siniestros_scored_final IS
'Final table for processed claims with rule score, ML prediction, final risk traffic light and explainability. Output is for human review only.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.score_total_reglas IS
'Sum of points activated by rule-based risk signals.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.probabilidad_ml IS
'ML estimated probability of risk, from 0 to 1.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.score_final IS
'Hybrid score: 70 percent normalized rule score plus 30 percent ML probability scaled to 0-100.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.semaforo_final IS
'Final risk classification: Verde, Amarillo or Rojo.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.ids_siniestros_similares_top5 IS
'JSON array with up to 5 claim IDs whose descriptions are most similar.';

COMMENT ON COLUMN fraud_ia.siniestros_scored_final.explicabilidad IS
'Human-readable explanation based on rule alerts, critical rules, ML probability and textual similarity.';

CREATE INDEX IF NOT EXISTS idx_siniestros_final_semaforo
    ON fraud_ia.siniestros_scored_final (semaforo_final);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_score_desc
    ON fraud_ia.siniestros_scored_final (score_final DESC);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_prob_ml_desc
    ON fraud_ia.siniestros_scored_final (probabilidad_ml DESC);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_fecha_ocurrencia
    ON fraud_ia.siniestros_scored_final (fecha_ocurrencia);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_asegurado
    ON fraud_ia.siniestros_scored_final (id_asegurado);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_proveedor
    ON fraud_ia.siniestros_scored_final (id_proveedor);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_reglas_gin
    ON fraud_ia.siniestros_scored_final USING GIN (reglas_criticas_activadas jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_siniestros_final_alertas_gin
    ON fraud_ia.siniestros_scored_final USING GIN (alertas_score_activadas jsonb_path_ops);
