import pandas as pd
import numpy as np


def apply_critical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa las reglas críticas de negocio (RF-01 a RF-07) sobre el dataset de siniestros.
    Retorna un DataFrame con banderas booleanas para cada regla y una columna de semáforo.
    """
    rules_df = pd.DataFrame(index=df.index)

    # Asegurarnos de que los campos de texto existan para evitar errores
    cobertura_str = df.get("cobertura", pd.Series("", index=df.index)).astype(str).str.lower()

    # ==========================================
    # CLASIFICACIÓN ROJO (Alta Severidad)
    # ==========================================

    # RF-01: Cobertura Pérdida Total por Robo (PTXRB)
    rules_df["RF_01_perdida_total_robo"] = cobertura_str.str.contains("pérdida total|ptxrb") & cobertura_str.str.contains("robo")

    # RF-02: Evidencia de Falsificación o Adulteración Documental Evidente
    # Asumimos que si hay docs inconsistentes reportados desde el feature engineering, se activa.
    rules_df["RF_02_adulteracion_doc"] = df.get("docs_inconsistentes", pd.Series(False, index=df.index)).astype(bool)

    # RF-03: Asegurado, Beneficiario o APS Coincidencia Exacta con "Lista Restrictiva"
    # El puntaje de proveedor de 10 indicaba explícitamente estar en lista restrictiva.
    rules_df["RF_03_lista_restrictiva"] = df.get("score_proveedor", 0) == 10

    # RF-04: Dinámica del Accidente Físicamente Imposible
    # Si el score de dinámica sospechosa alcanzó su máximo (6 pts por relato ilógico).
    rules_df["RF_04_dinamica_imposible"] = df.get("score_dinamica_sospechosa", 0) >= 6

    # ==========================================
    # CLASIFICACIÓN AMARILLO (Advertencia)
    # ==========================================

    # RF-05: Siniestro Extremo al Borde de Vigencia (< 48 hrs)
    # Calculamos la diferencia exacta en horas si tenemos datetime, o asumimos < 2 días.
    if "fecha_ocurrencia" in df.columns and "fecha_inicio" in df.columns and "fecha_fin" in df.columns:
        dias_inicio = (df["fecha_ocurrencia"] - df["fecha_inicio"]).dt.total_seconds() / 86400
        dias_fin = (df["fecha_fin"] - df["fecha_ocurrencia"]).dt.total_seconds() / 86400
        rules_df["RF_05_borde_vigencia"] = (dias_inicio <= 2) | (dias_fin <= 2)
    else:
        rules_df["RF_05_borde_vigencia"] = False

    # RF-06: Demora Atípica en Denuncia de Robo (> 4 días)
    is_robo = cobertura_str.str.contains("robo")
    demora_dias = pd.to_numeric(df.get("dias_entre_ocurrencia_reporte", 0), errors="coerce").fillna(0)
    rules_df["RF_06_demora_robo"] = is_robo & (demora_dias > 4)

    # RF-07: Narrativa Idéntica (Clonada)
    # Esta variable 'max_similitud_textual' vendrá de tu pipeline de NLP (TF-IDF + Cosine Similarity)
    rules_df["RF_07_narrativa_identica"] = df.get("max_similitud_textual", 0.0) > 0.85

    # ==========================================
    # SEMÁFORO DETERMINÍSTICO (Overrides)
    # ==========================================

    # Condición para ROJO: Al menos una regla roja (RF-01 a RF-04) es verdadera
    condicion_rojo = rules_df[["RF_01_perdida_total_robo", "RF_02_adulteracion_doc", 
                               "RF_03_lista_restrictiva", "RF_04_dinamica_imposible"]].any(axis=1)                  
    # Condición para AMARILLO: Al menos una regla amarilla (RF-05 a RF-07) es verdadera
    condicion_amarillo = rules_df[["RF_05_borde_vigencia", "RF_06_demora_robo", "RF_07_narrativa_identica"]].any(axis=1)

    rules_df["semaforo_reglas_criticas"] = np.where(
        condicion_rojo, "Rojo",
        np.where(condicion_amarillo, "Amarillo", "Verde")
    )

    return rules_df


def integrate_rules_with_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Función helper para unir las banderas booleanas al dataset principal.
    Esto permite que el modelo de ML utilice estas banderas como variables predictivas.
    """
    rules_df = apply_critical_rules(features_df)
    # Concatenamos las reglas al dataframe original (excepto la columna semáforo que es solo para la vista final)
    cols_to_add = [c for c in rules_df.columns if c.startswith("RF_")]

    return pd.concat([features_df, rules_df[cols_to_add]], axis=1)
