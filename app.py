# app.py
import base64
import os
from typing import Dict, Any, Optional

from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# ====== ENV VARS (set these in your shell / Render) ======
AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT", "").strip()
AZURE_ML_KEY = os.getenv("AZURE_ML_KEY", "").strip()

# If you want to force routing to a specific deployment while testing:
# export AZURE_ML_DEPLOYMENT="die-wear-resnet18-ds3"
AZURE_ML_DEPLOYMENT = os.getenv("AZURE_ML_DEPLOYMENT", "").strip()

# Optional: used only if your score.py returns NOT_A_DIE probability
NOT_DIE_THRESHOLD = float(os.getenv("NOT_DIE_THRESHOLD", "0.55"))

# Keep payload size sane (base64 can get big)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def call_azure_ml(image_bytes: bytes) -> Dict[str, Any]:
    """
    Sends base64 image JSON to Azure ML Online Endpoint.
    Returns parsed JSON.
    """
    if not AZURE_ML_ENDPOINT or not AZURE_ML_KEY:
        raise RuntimeError("Missing AZURE_ML_ENDPOINT or AZURE_ML_KEY environment variables.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image_base64": b64}

    headers = {
        "Content-Type": "application/json",
        # Azure ML online endpoints (Key auth)
        "Authorization": f"Bearer {AZURE_ML_KEY}",
    }

    # Force a specific deployment (ds3) when AZURE_ML_DEPLOYMENT is set
    if AZURE_ML_DEPLOYMENT:
        headers["azureml-model-deployment"] = AZURE_ML_DEPLOYMENT

    resp = requests.post(AZURE_ML_ENDPOINT, json=payload, headers=headers, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Azure ML error {resp.status_code}: {resp.text}")

    return resp.json()


def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes output so the template can render consistently.

    Expected Azure output:
      {
        "prediction": "REPLACE",
        "probabilities": {"OK": 0.27, "REPLACE": 0.43, "WARNING": 0.29}
      }

    Optional future:
      probabilities may include "NOT_A_DIE"
    """
    prediction = (result.get("prediction") or "UNKNOWN").upper()
    probs = result.get("probabilities", {}) or {}

    # Optional NOT_A_DIE gate (only if your model returns it)
    not_die_prob = probs.get("NOT_A_DIE", None)
    if not_die_prob is not None:
        try:
            if float(not_die_prob) >= NOT_DIE_THRESHOLD:
                prediction = "NOT_A_DIE"
        except Exception:
            pass

    badge_class = "badge-neutral"
    if prediction == "OK":
        badge_class = "badge-ok"
    elif prediction == "WARNING":
        badge_class = "badge-warning"
    elif prediction == "REPLACE":
        badge_class = "badge-replace"
    elif prediction == "NOT_A_DIE":
        badge_class = "badge-notdie"

    return {
        "prediction": prediction,
        "probabilities": probs,
        "badge_class": badge_class,
        "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
    }


def _render(result: Optional[Dict[str, Any]] = None, image_data_url: Optional[str] = None):
    return render_template(
        "index.html",
        result=result,
        image_data_url=image_data_url,
        endpoint=AZURE_ML_ENDPOINT,
        deployment=AZURE_ML_DEPLOYMENT,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Supports BOTH:
      - GET /  (page load)
      - POST / (if the form posts to "/")
    """
    if request.method == "GET":
        return _render(result=None, image_data_url=None)

    # If someone POSTs to "/", treat it like predict
    return _handle_prediction_post()


@app.post("/predict")
def predict():
    """
    Primary POST handler used by the UI (recommended form action="/predict")
    """
    return _handle_prediction_post()


def _handle_prediction_post():
    file = request.files.get("image")
    if not file or file.filename == "":
        return _render(
            result={
                "prediction": "NO_FILE",
                "badge_class": "badge-neutral",
                "probabilities": {},
                "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
            },
            image_data_url=None,
        )

    image_bytes = file.read()

    # show preview in UI
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mimetype = file.mimetype or "image/jpeg"
    image_data_url = f"data:{mimetype};base64,{image_b64}"

    try:
        raw = call_azure_ml(image_bytes)
        result = normalize_result(raw)
    except Exception as e:
        result = {
            "prediction": "ERROR",
            "badge_class": "badge-replace",
            "probabilities": {},
            "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default",
            "error": str(e),
        }

    return _render(result=result, image_data_url=image_data_url)


if __name__ == "__main__":
    # Run on 0.0.0.0 so it works in containers too
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
