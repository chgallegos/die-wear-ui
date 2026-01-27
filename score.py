import json
import base64
import io
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


model = None
classes = None
img_size = 224
device = "cpu"

# Gate thresholds (tunable via env vars)
# If model confidence is too low, treat as NOT_DIE
NOT_DIE_MAXPROB_THRESHOLD = float(os.environ.get("NOT_DIE_MAXPROB_THRESHOLD", "0.60"))
NOT_DIE_MARGIN_THRESHOLD = float(os.environ.get("NOT_DIE_MARGIN_THRESHOLD", "0.15"))


def init():
    """
    Azure ML calls this once when the container starts.
    Loads the model artifact from the registered model path.
    """
    global model, classes, img_size, device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(model_path):
        # Sometimes AML nests it; try to find the first .pt
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".pt"):
                    model_path = os.path.join(root, f)
                    break

    ckpt = torch.load(model_path, map_location=device)

    classes = ckpt.get("classes", ["OK", "REPLACE", "WARNING"])
    img_size = int(ckpt.get("img_size", 224))

    num_classes = len(classes)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device)
    m.eval()

    model = m
    print(f"Loaded model from: {model_path}")
    print(f"Classes: {classes} | img_size: {img_size} | device: {device}")
    print(
        f"Gate: NOT_DIE_MAXPROB_THRESHOLD={NOT_DIE_MAXPROB_THRESHOLD}, "
        f"NOT_DIE_MARGIN_THRESHOLD={NOT_DIE_MARGIN_THRESHOLD}"
    )


_preprocess = None


def _get_preprocess():
    global _preprocess, img_size
    if _preprocess is None:
        _preprocess = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
    return _preprocess


def _image_from_request(data):
    """
    Accepts either:
      - raw bytes (rare in studio test)
      - JSON with {"image_base64": "..."} OR {"image": "..."} OR {"data": "..."}
    """
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data)).convert("RGB")

    if isinstance(data, str):
        # Might be a JSON string
        try:
            data = json.loads(data)
        except Exception:
            # Might be base64 directly
            b = base64.b64decode(data)
            return Image.open(io.BytesIO(b)).convert("RGB")

    if isinstance(data, dict):
        b64 = data.get("image_base64") or data.get("image") or data.get("data")
        if not b64:
            raise ValueError("No image found. Send {'image_base64': '...'}")
        b = base64.b64decode(b64)
        return Image.open(io.BytesIO(b)).convert("RGB")

    raise ValueError("Unsupported input format. Provide JSON with image_base64.")


def run(data):
    """
    Azure ML calls this per request.
    Returns JSON-serializable output.
    """
    img = _image_from_request(data)
    x = _get_preprocess()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probs = probs_t.tolist()

    # Top-2 confidence for "not a die" gating
    top2 = torch.topk(probs_t, k=min(2, len(probs))).values.tolist()
    p1 = float(top2[0])
    p2 = float(top2[1]) if len(top2) > 1 else 0.0
    margin = p1 - p2

    pred_idx = int(torch.argmax(probs_t).item())
    pred_label = classes[pred_idx]

    is_die = (p1 >= NOT_DIE_MAXPROB_THRESHOLD) and (margin >= NOT_DIE_MARGIN_THRESHOLD)

    if not is_die:
        return {
            "prediction": "NOT_DIE",
            "is_die": False,
            "gate": {
                "max_prob": p1,
                "margin": margin,
                "max_prob_threshold": NOT_DIE_MAXPROB_THRESHOLD,
                "margin_threshold": NOT_DIE_MARGIN_THRESHOLD,
            },
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        }

    return {
        "prediction": pred_label,
        "is_die": True,
        "gate": {
            "max_prob": p1,
            "margin": margin,
            "max_prob_threshold": NOT_DIE_MAXPROB_THRESHOLD,
            "margin_threshold": NOT_DIE_MARGIN_THRESHOLD,
        },
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }
