from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.agent import run_agent
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
import logging
import os
import sys
import time

from fastapi.templating import Jinja2Templates
from prometheus_client import Histogram, Gauge

from fastapi import HTTPException

# ========================
# LOGGING CONFIG
# ========================
LOG_DIR = os.environ.get("LOG_DIR", "/fluentd/log")
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
    "request_latency_seconds",
    "Latency of /ask endpoint (full route)",
    buckets=(0.1, 0.3, 0.5, 1, 2, 3, 5)
)

INFERENCE_TIME = Histogram(
    "inference_duration_seconds",
    "Duration of model inference (agent)",
    buckets=(0.1, 0.5, 1, 2, 3, 5)
)

# ========================
# FASTAPI APP INIT
# ========================
app = FastAPI(title="Vietnamese Law Chatbot API")
Instrumentator().instrument(app).expose(app)

# Templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# ========================
# ROUTES
# ========================
@app.post("/ask", response_model=QueryResponse)
@REQUEST_LATENCY.time()  # Đo tổng thời gian route
def ask_question(data: QueryRequest):
    lawchat_logger.info(f"Received query: {data.query}")
    try:
        start = time.time()
        result = run_agent(data.query)
        inference_duration = time.time() - start
        INFERENCE_TIME.observe(inference_duration)

        lawchat_logger.info(
            f"Generated response: {result[:200]}... | Inference: {inference_duration:.3f}s"
        )
        return {"response": result}
    except Exception as e:
        alert_logger.error(f"Exception: {e}", exc_info=True)
        return {"response": "Lỗi hệ thống, vui lòng thử lại sau."}

@app.get("/test-alert")
def test_alert(trigger: str = "500"):
    """
    Endpoint dùng để test alert:
    - trigger=500: trả lỗi HTTP 500 để kích hoạt High5xxErrorRate
    - trigger=slow: sleep lâu để kích hoạt SlowInference
    """
    if trigger == "500":
        alert_logger.error("Simulated server error for testing High5xxErrorRate alert.")
        raise HTTPException(status_code=500, detail="Simulated 500 error")
    elif trigger == "slow":
        import time
        start = time.time()
        time.sleep(3)  # ngủ 3s để vượt ngưỡng SlowInference (> 2s)
        duration = time.time() - start
        INFERENCE_TIME.observe(duration)
        alert_logger.error(f"Simulated slow inference: {duration:.2f}s")
        return {"message": f"Simulated slow inference completed in {duration:.2f}s"}
    else:
        return {"message": "No test triggered"}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
