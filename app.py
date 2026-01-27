import os
import base64
from typing import Any, Dict, Tuple, Optional

import requests
from flask import Flask, render_template, request


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT", "").strip()
AZURE_ML_KEY = os.getenv("AZURE_ML_KEY", "").strip()

# If the model is unsure (max probability below this), show NOT_A_DIE.
# Set lower (e.g., 0.10) to basically disable NOT_A_DIE for your die-only workflow.
NOT_DIE_THRESHOLD = float(os.getenv("NOT_DIE_THRESHOLD", "0.20"))

# Upload size limit (10 MB)
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)

# Server port (avoid macOS port 5000 conflicts)
PORT = int(os.getenv("PORT", "5001"))


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def status_to_css(status: str) -> str:
    s = (status or "").upper()
    if s == "OK":
        return "ok"
    if s == "WARNING":
        return "warning"
    if s == "REPLACE":
        return "replace"
    return "notdie"


def safe_round_probs(probs: Dict[str, Any]) -> Dict[str, str]:
    # Keep display clean and stable
    out = {}
    for k in ["OK", "WARNING", "REPLACE"]:
        v = probs.get(k)
        try:
            out[k] = f"{float(v):.3f}"
        except Exception:
            out[k] = "n/a"
    return out


def call_azure_endpoint(image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Sends base64 image to Azure ML endpoint and returns:
      prediction: "OK"/"WARNING"/"REPLACE" (or "NOT_A_DIE")
      probabilities: dict with keys OK/WARNING/REPLACE
    """
    if not AZURE_ML_ENDPOINT:
        raise RuntimeError("AZURE_ML_ENDPOINT is not set.")
    if not AZURE_ML_KEY:
        raise RuntimeError("AZURE_ML_KEY is not set.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image_base64": b64}

    headers = {
        "Content-Type": "application/json",
        # Azure ML online endpoints (Key auth) expects Authorization: Bearer <key>
        "Authorization": f"Bearer {AZURE_ML_KEY}",
    }

    resp = requests.post(AZURE_ML_ENDPOINT, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        # Surface useful details
        raise RuntimeError(f"Azure ML request failed ({resp.status_code}): {resp.text}")

    data = resp.json()  # expected: {"prediction": "...", "probabilities": {...}}
    prediction = (data.get("prediction") or "").upper()
    probabilities = data.get("probabilities") or {}

    # Optional NOT_A_DIE gate:
    # If model is very unsure, treat as out-of-scope.
    try:
        max_p = max(float(probabilities.get("OK", 0)), float(probabilities.get("WARNING", 0)), float(probabilities.get("REPLACE", 0)))
    except Exception:
        max_p = 0.0

    if max_p < NOT_DIE_THRESHOLD:
        return "NOT_A_DIE", probabilities

    # Otherwise stick to the core label
    if prediction not in {"OK", "WARNING", "REPLACE"}:
        # fallback: pick argmax
        try:
            scores = {
                "OK": float(probabilities.get("OK", 0)),
                "WARNING": float(probabilities.get("WARNING", 0)),
                "REPLACE": float(probabilities.get("REPLACE", 0)),
            }
            prediction = max(scores, key=scores.get)
        except Exception:
            prediction = "WARNING"

    return prediction, probabilities


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[str] = None
    probabilities_display: Optional[Dict[str, str]] = None
    css_class: Optional[str] = None
    message: Optional[str] = None
    image_data_url: Optional[str] = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            message = "Please choose an image file."
            return render_template(
                "index.html",
                result=result,
                probabilities=probabilities_display,
                css_class=css_class,
                message=message,
                image_data_url=image_data_url,
            )

        image_bytes = f.read()
        if not image_bytes:
            message = "That file looks empty. Try another image."
            return render_template(
                "index.html",
                result=result,
                probabilities=probabilities_display,
                css_class=css_class,
                message=message,
                image_data_url=image_data_url,
            )

        # Preview (base64 -> data URL)
        mimetype = f.mimetype or "image/jpeg"
        image_data_url = f"data:{mimetype};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        try:
            prediction, probs = call_azure_endpoint(image_bytes)
            result = prediction
            probabilities_display = safe_round_probs(probs)
            css_class = status_to_css(result)

            if result == "NOT_A_DIE":
                message = (
                    f"Low confidence (max prob < {NOT_DIE_THRESHOLD}). "
                    "If you never want this for your die workflow, set NOT_DIE_THRESHOLD lower (e.g., 0.10)."
                )
        except Exception as e:
            result = "ERROR"
            css_class = "notdie"
            message = str(e)

    return render_template(
        "index.html",
        result=result,
        probabilities=probabilities_display,
        css_class=css_class,
        message=message,
        image_data_url=image_data_url,
        endpoint_set=bool(AZURE_ML_ENDPOINT),
        key_set=bool(AZURE_ML_KEY),
        not_die_threshold=NOT_DIE_THRESHOLD,
        max_upload_mb=MAX_UPLOAD_MB,
    )


if __name__ == "__main__":
    # Local run: python3 app.py
    app.run(host="0.0.0.0", port=PORT)
