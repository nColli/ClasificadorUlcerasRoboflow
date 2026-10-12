# Clasificador Ulceras Por Presión Roboflow

Prueba de concepto de como hacer el deploy de un modelo entrenado en roboflow, utilizando la API provista.<br>

## Compatibilidad y configuración
Para correr el script, se necesita configurar las variables de entorno con la API_KEY, la URL y los ID de los modelos:<br>
- `MODEL_RESNET18`, `MODEL_RESNET50`, `MODEL_VIT`: MODEL_ID de Roboflow de cada modelo. Solo se ofrecen en la web los modelos con la variable definida; tiene que haber al menos uno.<br>
Importante: instalar desde versión Python 3.12 y asegurarse que Inference-sdk sea version >= 1.5.0, sino no se hace el deploy porque no esta Inference-sdk para Python 3.14, tiene una versión antigua que el server denega el acceso.
## Límite de uso
- Los contadores están en memoria: se reinician con cada deploy o reinicio, y uvicorn debe correr con **un solo worker**.
- Si la app corre detrás de un proxy (Render, Railway, Fly, Cloudflare, nginx), definir `TRUSTED_PROXY_HOPS` con la cantidad de proxies (normalmente `1`). Si no, todos los usuarios comparten la IP del proxy y el límite.
