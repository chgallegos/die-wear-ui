# app.py
import base64
import os
import time
from typing import Dict, Any, Optional, Tuple

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# ====== ENV VARS (set these in your shell / Render) ======
AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT", "").strip()
AZURE_ML_KEY = os.getenv("AZURE_ML_KEY", "").strip()

# Optional: force routing to a specific deployment while testing
# e.g. die-wear-resnet18-ds3
AZURE_ML_DEPLOYMENT = os.getenv("AZURE_ML_DEPLOYMENT", "").strip()

# Upload limit
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def call_azure_ml(image_bytes: bytes) -> Tuple[Dict[str, Any], float]:
    """
    Sends base64 image JSON to Azure ML Online Endpoint.
    Returns (parsed JSON, latency_seconds).
    """
    if not AZURE_ML_ENDPOINT or not AZURE_ML_KEY:
        raise RuntimeError("Missing AZURE_ML_ENDPOINT or AZURE_ML_KEY environment variables.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image_base64": b64}

    headers = {
        "Content-Type": "application/json",
        # Key-auth header for Azure ML managed online endpoints
        "Authorization": f"Bearer {AZURE_ML_KEY}",
    }

    # Force a specific deployment (ds3) when AZURE_ML_DEPLOYMENT is set
    if AZURE_ML_DEPLOYMENT:
        headers["azureml-model-deployment"] = AZURE_ML_DEPLOYMENT

    t0 = time.time()
    resp = requests.post(AZURE_ML_ENDPOINT, json=payload, headers=headers, timeout=60)
    latency = time.time() - t0

    if not resp.ok:
        raise RuntimeError(f"Azure ML error {resp.status_code}: {resp.text}")

    return resp.json(), latency


def normalize_result(raw: Dict[str, Any], latency_s: float) -> Dict[str, Any]:
    """
    Normalizes the Azure response for consistent UI rendering.

    Handles either:
      A) classic:
        {"prediction": "REPLACE", "probabilities": {...}}

      B) gated ds3 style:
        {
          "prediction":"NOT_DIE",
          "is_die": false,
          "gate": {...},
          "probabilities": {...}
        }
    """
    prediction = (raw.get("prediction") or "UNKNOWN").upper()
    probs = raw.get("probabilities", {}) or {}

    # Make sure prob keys are upper-ish
    # (We will still read them safely in template)
    ok_p = _safe_float(probs.get("OK"), 0.0) or 0.0
    warn_p = _safe_float(probs.get("WARNING"), 0.0) or 0.0
    rep_p = _safe_float(probs.get("REPLACE"), 0.0) or 0.0

    # Badge class + recommendation
    badge_class = "badge-neutral"
    recommendation = "No recommendation available."

    if prediction == "OK":
        badge_class = "badge-ok"
        recommendation = "Die condition acceptable — continue use."
    elif prediction == "WARNING":
        badge_class = "badge-warning"
        recommendation = "Monitor wear — schedule a recheck soon."
    elif prediction == "REPLACE":
        badge_class = "badge-replace"
        recommendation = "Replace die as soon as possible."
    elif prediction in ("NOT_DIE", "NOT_A_DIE"):
        prediction = "NOT_DIE"
        badge_class = "badge-notdie"
        recommendation = "Image appears out of scope — retake a clear die photo."

    # Gate info (optional)
    gate = raw.get("gate") or {}
    gate_max_prob = _safe_float(gate.get("max_prob"))
    gate_margin = _safe_float(gate.get("margin"))
    gate_max_thr = _safe_float(gate.get("max_prob_threshold"))
    gate_margin_thr = _safe_float(gate.get("margin_threshold"))

    is_die = raw.get("is_die")
    gate_pass = None
    if isinstance(is_die, bool):
        gate_pass = is_die

    # For confidence bars, percentages:
    def pct(x: float) -> int:
        # Round to whole percent but keep within 0..100
        return max(0, min(100, int(round(x * 100))))

    result = {
        "prediction": prediction,
        "badge_class": badge_class,
        "recommendation": recommendation,
        "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
        "latency_ms": int(round(latency_s * 1000)),
        "probabilities": {
            "OK": ok_p,
            "WARNING": warn_p,
            "REPLACE": rep_p,
        },
        "prob_percent": {
            "OK": pct(ok_p),
            "WARNING": pct(warn_p),
            "REPLACE": pct(rep_p),
        },
        "gate": {
            "present": bool(gate),
            "pass": gate_pass,
            "max_prob": gate_max_prob,
            "margin": gate_margin,
            "max_prob_threshold": gate_max_thr,
            "margin_threshold": gate_margin_thr,
        },
        "raw": raw,  # for debugging if you want to render it later
    }
    return result


def _render(result: Optional[Dict[str, Any]] = None, image_data_url: Optional[str] = None, filename: str = ""):
    return render_template(
        "index.html",
        result=result,
        image_data_url=image_data_url,
        filename=filename,
        endpoint=AZURE_ML_ENDPOINT,
        deployment=AZURE_ML_DEPLOYMENT,
        endpoint_set=bool(AZURE_ML_ENDPOINT),
        key_set=bool(AZURE_ML_KEY),
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.route("/", methods=["GET"])
def index():
    return _render(result=None, image_data_url=None)


@app.route("/predict", methods=["POST"])
def predict():
    # IMPORTANT: field name is "image" (matches template)
    file = request.files.get("image")
    if not file or file.filename == "":
        return _render(
            result={
                "prediction": "NO_FILE",
                "badge_class": "badge-neutral",
                "recommendation": "Choose an image to analyze.",
                "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
                "latency_ms": None,
                "probabilities": {"OK": 0.0, "WARNING": 0.0, "REPLACE": 0.0},
                "prob_percent": {"OK": 0, "WARNING": 0, "REPLACE": 0},
                "gate": {"present": False},
                "error": None,
            },
            image_data_url=None,
        )

    image_bytes = file.read()
    mimetype = file.mimetype or "image/jpeg"

    # Preview for UI
    preview_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:{mimetype};base64,{preview_b64}"

    try:
        raw, latency_s = call_azure_ml(image_bytes)
        # Optional debug log (Render logs)
        print("RAW_AZURE_RESPONSE:", raw, flush=True)

        result = normalize_result(raw, latency_s)
    except Exception as e:
        result = {
            "prediction": "ERROR",
            "badge_class": "badge-replace",
            "recommendation": "Request failed — see error details.",
            "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
            "latency_ms": None,
            "probabilities": {"OK": 0.0, "WARNING": 0.0, "REPLACE": 0.0},
            "prob_percent": {"OK": 0, "WARNING": 0, "REPLACE": 0},
            "gate": {"present": False},
            "error": str(e),
        }

    return _render(result=result, image_data_url=image_data_url, filename=file.filename)


if __name__ == "__main__":
    # Works locally + Render (Render sets PORT)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
