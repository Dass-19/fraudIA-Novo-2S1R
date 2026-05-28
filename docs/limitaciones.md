# Limitaciones

- El sistema genera alertas para revision humana; no acusa fraude ni decide pagos/rechazos.
- El modelo depende de datos sinteticos o publicos, por lo que sus metricas no representan desempeno real en produccion.
- Las reglas son trazables, pero pueden generar falsos positivos.
- La similitud textual puede marcar narrativas parecidas que no necesariamente impliquen irregularidad.
- El agente SQL puede generar consultas incorrectas si el LLM interpreta mal la pregunta.
- Para produccion se recomienda usar un usuario PostgreSQL solo lectura para el agente.
- No deben usarse datos personales reales ni credenciales en GitHub.
- Algunos modelos de Hugging Face requieren permisos, aceptacion de terminos o disponibilidad del endpoint.
