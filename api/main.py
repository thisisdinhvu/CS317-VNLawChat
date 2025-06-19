from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
import logging
import os
import sys
import time
import uuid
from prometheus_client import Counter

from langchain_core.messages import HumanMessage, AIMessage
from src.agent import run_agent  # Đảm bảo import đúng

# ========================
# LOGGING CONFIG
# ========================
LOG_DIR = os.environ.get("LOG_DIR", "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Lawchat Logger
lawchat_logger = logging.getLogger("lawchat")
lawchat_logger.setLevel(logging.INFO)
lawchat_fh = logging.FileHandler(f"{LOG_DIR}/lawchat.log")
lawchat_sh = logging.StreamHandler(sys.stdout)
lawchat_formatter = logging.Formatter("%(asctime)s [LAWCHAT] %(levelname)s: %(message)s")
lawchat_fh.setFormatter(lawchat_formatter)
lawchat_sh.setFormatter(lawchat_formatter)
lawchat_logger.addHandler(lawchat_fh)
lawchat_logger.addHandler(lawchat_sh)

# Alerts Logger
alert_logger = logging.getLogger("alerts")
alert_logger.setLevel(logging.WARNING)
alert_fh = logging.FileHandler(f"{LOG_DIR}/alerts.log")
alert_sh = logging.StreamHandler(sys.stdout)
alert_formatter = logging.Formatter("%(asctime)s [ALERT] %(levelname)s: %(message)s")
alert_fh.setFormatter(alert_formatter)
alert_sh.setFormatter(alert_formatter)
alert_logger.addHandler(alert_fh)
alert_logger.addHandler(alert_sh)

# ========================
# PROMETHEUS METRICS
# ========================
REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds",
    "Latency of /chat endpoint",
    buckets=(0.1, 0.3, 0.5, 1, 2, 3, 5)
)

INFERENCE_TIME = Histogram(
    "chat_inference_duration_seconds",
    "Duration of LangGraph model inference",
    buckets=(0.1, 0.5, 1, 2, 3, 5)
)

ERROR_5XX_COUNTER = Counter(
    "http_5xx_errors_total",
    "Number of HTTP 5xx responses",
    ["path"]
)

# ========================
# FASTAPI INIT
# ========================
app = FastAPI(title="LangGraph Chat Agent API")
Instrumentator().instrument(app).expose(app)


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ========================
# REQUEST/RESPONSE SCHEMA
# ========================
class ChatRequest(BaseModel):
    user_input: str
    thread_id: str = str(uuid.uuid4())  # Generate a unique thread ID by default

# Define response model
class ChatResponse(BaseModel):
    response: str
    thread_id: str

# ========================
# /chat endpoint
# ========================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lawchat_logger.info("=" * 50)
    lawchat_logger.info(f"[START] Received chat: '{request.user_input}' | Thread ID: {request.thread_id}")
    start = time.time()

    try:
        with REQUEST_LATENCY.time():
            final_response = run_agent(request.user_input, request.thread_id)
            duration = time.time() - start
            INFERENCE_TIME.observe(duration)

        if not final_response:
            alert_logger.error("[ERROR] No valid response generated.")
            raise HTTPException(status_code=500, detail="No valid response generated.")

        lawchat_logger.info(
            f"[END] Response: '{final_response[:200]}...' | Inference Time: {duration:.3f}s"
        )
        return ChatResponse(response=final_response, thread_id=request.thread_id)

    except Exception as e:
        duration = time.time() - start
        alert_logger.error(f"[EXCEPTION] During /chat request: {str(e)} | Duration: {duration:.3f}s", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# ========================
# /alert: Nhận cảnh báo từ Alertmanager (Webhook)
# ========================
@app.post("/alert")
async def receive_alert(payload: dict):
    alert_logger.warning(f"[ALERTMANAGER] Alert received: {payload}")
    return {"status": "received"}

# ========================
# Homepage
# ========================
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
