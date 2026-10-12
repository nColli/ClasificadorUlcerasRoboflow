# Clasificador Ulceras Por Presión Roboflow

Prueba de concepto de como hacer el deploy de un modelo entrenado en roboflow, utilizando la API provista.<br>

## Compatibilidad y configuración
Para correr el script, se necesita configurar las variables de entorno con la API_KEY, ID del modelo y la URL.<br>
Importante: instalar desde versión Python 3.12 y asegurarse que Inference-sdk sea version >= 1.5.0, sino se deniega el acceso.
## Límite de uso
- Los contadores están en memoria: se reinician con cada deploy o reinicio, y uvicorn debe correr con **un solo worker**.
- Si la app corre detrás de un proxy (Render, Railway, Fly, Cloudflare, nginx), definir `TRUSTED_PROXY_HOPS` con la cantidad de proxies (normalmente `1`). Si no, todos los usuarios comparten la IP del proxy y el límite.
