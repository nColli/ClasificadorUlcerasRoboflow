"""API mínima para clasificar el estadio de una úlcera por presión con inference_sdk de Roboflow."""
import io
import math
import os
import threading
import time

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
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
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))  # inferencias por IP...
RATE_WINDOW_S = int(os.getenv("RATE_WINDOW_S", "3600"))  # ...por ventana; al agotarlas se bloquea este tiempo
# Cantidad de proxies propios delante de la app (1 en Render). Con 0 se usa la IP del socket.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))

client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY,
).configure(InferenceConfiguration(api_key_transport="header"))

app = FastAPI(title="PU Stage Classifier")


class RateLimiter:
    """Límite por IP en memoria: tras `limit` intentos dentro de `window` s, bloquea `window` s.

    El estado se pierde al reiniciar y es por proceso: correr uvicorn con un solo worker.
    """

    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.attempts: dict[str, list[float]] = {}
        self.blocked_until: dict[str, float] = {}
        self.lock = threading.Lock()  # los endpoints sync corren en un threadpool
        self.last_sweep = 0.0

    def _prune(self, ip: str, now: float) -> list[float]:
        recent = [t for t in self.attempts.get(ip, []) if now - t < self.window]
        self.attempts[ip] = recent
        if self.blocked_until.get(ip, 0) <= now:
            self.blocked_until.pop(ip, None)
        return recent

    def _sweep(self, now: float):
        """Descarta IPs inactivas para que los diccionarios no crezcan sin límite."""
        if now - self.last_sweep < 60:
            return
        self.last_sweep = now
        for ip in list(self.attempts):
            if not self._prune(ip, now):
                del self.attempts[ip]

    def status(self, ip: str) -> tuple[int, int]:
        """Devuelve (intentos restantes, segundos hasta desbloqueo)."""
        now = time.monotonic()
        with self.lock:
            recent = self._prune(ip, now)
            until = self.blocked_until.get(ip)
            if until:
                return 0, math.ceil(until - now)
            return self.limit - len(recent), 0

    def acquire(self, ip: str) -> int:
        """Registra un intento y devuelve los restantes. Lanza 429 si la IP está bloqueada."""
        now = time.monotonic()
        with self.lock:
            self._sweep(now)
            recent = self._prune(ip, now)
            until = self.blocked_until.get(ip)
            if until:
                retry = math.ceil(until - now)
                raise HTTPException(
                    429,
                    f"Alcanzó el límite de {self.limit} clasificaciones por hora. "
                    f"Podrá intentar de nuevo en {math.ceil(retry / 60)} min.",
                    headers={"Retry-After": str(retry), "X-RateLimit-Remaining": "0"},
                )
            recent.append(now)
            if len(recent) >= self.limit:
                self.blocked_until[ip] = now + self.window
            return self.limit - len(recent)

    def release(self, ip: str):
        """Devuelve el último intento (p. ej. si Roboflow falló y no se consumió crédito)."""
        with self.lock:
            if self.attempts.get(ip):
                self.attempts[ip].pop()
                self.blocked_until.pop(ip, None)


limiter = RateLimiter(RATE_LIMIT, RATE_WINDOW_S)


def client_ip(request: Request) -> str:
    if TRUSTED_PROXY_HOPS > 0:
        # Cada proxy agrega a la derecha la IP de quien le habló; las entradas de la izquierda las controla el cliente.
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if len(hops) >= TRUSTED_PROXY_HOPS:
            return hops[-TRUSTED_PROXY_HOPS]
    return request.client.host if request.client else "unknown"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/quota")
def quota(request: Request):
    remaining, retry_after = limiter.status(client_ip(request))
    return {"limit": RATE_LIMIT, "remaining": remaining, "retry_after_s": retry_after}


# Endpoint sync (def): el SDK es bloqueante y FastAPI lo ejecuta en un threadpool.
@app.post("/api/classify")
def classify(request: Request, response: Response, file: UploadFile = File(...)):
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

    # Solo cuenta como intento lo que llega a Roboflow (las validaciones de arriba no gastan crédito).
    ip = client_ip(request)
    remaining = limiter.acquire(ip)
    try:
        raw = client.infer(image, model_id=MODEL_ID)
    except Exception as exc:  # el SDK lanza varios tipos de error HTTP
        limiter.release(ip)
        raise HTTPException(502, f"Error de Roboflow: {exc}") from exc

    response.headers["X-RateLimit-Remaining"] = str(remaining)
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