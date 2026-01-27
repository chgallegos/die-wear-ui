# app.py
import base64
import os
from typing import Dict, Any

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
    if not AZURE_ML_ENDPOINT or not AZURE_ML_KEY:
        raise RuntimeError("Missing AZURE_ML_ENDPOINT or AZURE_ML_KEY environment variables.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image_base64": b64}

    headers = {
        "Content-Type": "application/json",
        # Azure ML online endpoints (Key auth) expects this:
        "Authorization": f"Bearer {AZURE_ML_KEY}",
    }

    # ✅ Force a specific deployment (ds3) when AZURE_ML_DEPLOYMENT is set
    if AZURE_ML_DEPLOYMENT:
        headers["azureml-model-deployment"] = AZURE_ML_DEPLOYMENT

    resp = requests.post(AZURE_ML_ENDPOINT, json=payload, headers=headers, timeout=60)
    # Helpful error details if auth/headers wrong
    if not resp.ok:
        raise RuntimeError(f"Azure ML error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data


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
    prediction = result.get("prediction", "UNKNOWN")
    probs = result.get("probabilities", {}) or {}

    # If you later add NOT_A_DIE to your model, you can gate it here:
    not_die_prob = probs.get("NOT_A_DIE", None)
    if not_die_prob is not None and float(not_die_prob) >= NOT_DIE_THRESHOLD:
        prediction = "NOT_A_DIE"

    # Determine badge color class for UI
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


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        image_data_url=None,
        endpoint=AZURE_ML_ENDPOINT,
        deployment=AZURE_ML_DEPLOYMENT,
    )


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template(
            "index.html",
            result={"prediction": "NO_FILE", "badge_class": "badge-neutral", "probabilities": {}, "deployment": AZURE_ML_DEPLOYMENT or "traffic-split/default"},
            image_data_url=None,
            endpoint=AZURE_ML_ENDPOINT,
            deployment=AZURE_ML_DEPLOYMENT,
        )

    image_bytes = file.read()

    # show preview in UI
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:{file.mimetype};base64,{image_b64}"

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

    return render_template(
        "index.html",
        result=result,
        image_data_url=image_data_url,
        endpoint=AZURE_ML_ENDPOINT,
        deployment=AZURE_ML_DEPLOYMENT,
    )


if __name__ == "__main__":
    # Run on 0.0.0.0 so it works in containers too
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
