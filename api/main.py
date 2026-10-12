"""API mínima para clasificar el estadio de una úlcera por presión con inference_sdk de Roboflow."""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image, ImageOps, UnidentifiedImageError
from dotenv import load_dotenv

load_dotenv('../.env')

ROBOFLOW_API_URL = os.getenv("API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.environ["API_KEY"]  # nunca se expone al navegador
MODEL_ID = os.getenv("API_MODEL")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY,
).configure(InferenceConfiguration(api_key_transport="header"))

app = FastAPI(title="PU Stage Classifier")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Endpoint sync (def): el SDK es bloqueante y FastAPI lo ejecuta en un threadpool.
@app.post("/api/classify")
def classify(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Formato no soportado. Use JPG, PNG o WEBP.")

    data = file.file.read()
    if not data:
        raise HTTPException(400, "El archivo está vacío.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"La imagen supera {MAX_BYTES // (1024 * 1024)} MB.")

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")  # corrige rotación de fotos de celular
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "El archivo no es una imagen válida.") from exc

    try:
        raw = client.infer(image, model_id=MODEL_ID)
    except Exception as exc:  # el SDK lanza varios tipos de error HTTP
        raise HTTPException(502, f"Error de Roboflow: {exc}") from exc

    return parse_result(raw)


def extract_block(raw) -> dict:
    """Devuelve el bloque de clasificación, sea respuesta de modelo o de workflow."""
    if isinstance(raw, list):  # run_workflow devuelve una lista de outputs
        raw = raw[0]
    if isinstance(raw, dict) and isinstance(raw.get("predictions"), dict) and "predictions" in raw["predictions"]:
        raw = raw["predictions"]  # output de workflow: {"predictions": {...bloque...}}
    return raw


def parse_result(raw) -> dict:
    try:
        block = extract_block(raw)
        preds = block["predictions"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, "Respuesta de Roboflow con formato inesperado.") from exc

    # Multi-label: {"stage1": {"confidence": ..}, ...}
    # Single-label: [{"class": "stage1", "confidence": ..}, ...]
    if isinstance(preds, dict):
        scores = [{"stage": k, "confidence": float(v.get("confidence", 0.0))} for k, v in preds.items()]
        predicted = block.get("predicted_classes") or []
    else:
        scores = [{"stage": p["class"], "confidence": float(p["confidence"])} for p in preds]
        predicted = [block["top"]] if block.get("top") else []

    scores.sort(key=lambda s: s["confidence"], reverse=True)
    conf_by_stage = {s["stage"]: s["confidence"] for s in scores}

    # predicted_classes vacío = ninguna clase supera el umbral del modelo.
    if predicted:
        stage = max(predicted, key=lambda c: conf_by_stage.get(c, 0.0))
        confident = True
    else:
        stage = scores[0]["stage"] if scores else None
        confident = False

    return {
        "stage": stage,
        "confident": confident,
        "scores": scores,
        "predicted_classes": predicted,
        "inference_id": block.get("inference_id"),
        "model_id": (block.get("resolved_model") or {}).get("model_id", MODEL_ID),
        "inference_time_s": block.get("time"),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")