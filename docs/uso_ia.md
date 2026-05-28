# Uso de IA

La solucion usa IA en tres niveles.

## 1. Machine Learning supervisado

El modelo final se exporta en:

```text
artifact/final-model/model.pkl
```

`fraud_model.py` carga este archivo y expone:

```python
clasificar(features_df)
classify_claims(features_df)
```

El modelo se apoya en `artifact/model_input_metadata.json` para ordenar y completar columnas de entrada.

## 2. NLP para similitud de narrativas

`build_features.py` usa TF-IDF y similitud coseno para generar:

- `ids_siniestros_similares_top5`
- `max_similitud_textual`
- `score_narrativas_similares`

Esto permite detectar reclamos con descripciones parecidas o clonadas.

## 3. Agente SQL

`claims_agent.py` usa LangChain con `create_sql_agent`, PostgreSQL y un LLM de Hugging Face por API.

Variables requeridas:

```env
HUGGINGFACEHUB_API_TOKEN=hf_xxx
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
```

Modelos sugeridos para probar:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`

El agente consulta la tabla `fraud_ia.siniestros_scored_final` y puede responder preguntas como:

- Cuales son los 10 siniestros con mayor score final?
- Que proveedores concentran mas alertas rojas?
- Que ciudades presentan mas casos en Amarillo o Rojo?
- Que reglas criticas se activan con mayor frecuencia?

La respuesta del agente debe interpretarse como apoyo al analista humano.
