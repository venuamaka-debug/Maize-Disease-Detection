import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import urllib.request

MODEL_PATH = "model/production_model.h5"
MODEL_URL = "https://huggingface.co/Venuamaka/maize-resnet50/resolve/main/production_model.h5"

def ensure_model_downloaded():
    """
    Render's free tier has an ephemeral filesystem — anything not in
    git is wiped on redeploy. The model file (93MB) is intentionally
    excluded from git (see .gitignore) since it exceeds sensible repo
    size limits. Instead it's hosted on Hugging Face's model hub and
    downloaded here on first run if not already present locally.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Model already present at {MODEL_PATH}, skipping download.")
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Model not found locally — downloading from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model download complete.")
IMG_SIZE = (224, 224)  # confirmed: image.target_size in config_snapshot.yaml

# Confirmed from src/config/settings.py:unified_classes — derived from
# class_mapping's first-insertion order, cross-checked against
# results.json per_class_metrics order, the recorded confusion matrices,
# and trainer.py's saved-order guard (line 318), which did not fire
# for this run. DO NOT reorder without re-deriving from settings.py.
CLASS_NAMES = [
    "Northern_Corn_Leaf_Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
]

# The saved .h5 contains a Lambda layer wrapping resnet50.preprocess_input
# (src/models/resnet50_model.py) — Keras's legacy H5 format only stores the
# function's NAME, not its code, so it must be supplied explicitly on load.
# Confirmed via commit dbc48ba (same fix already applied in trainer.py).
CUSTOM_OBJECTS = {
    "preprocess_input": tf.keras.applications.resnet50.preprocess_input
}

_model = None

def load_artifacts():
    global _model
    ensure_model_downloaded()
    _model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, safe_mode=False)
    return _model is not None

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Reproduces ONLY src/data/pipeline.py:374 (`/255.0`) — the input the
    model was trained to receive at its own input boundary.

    Do NOT apply resnet50.preprocess_input here. The saved model already
    contains, as its first two internal layers:
        Rescaling(255.0)  -> undoes the /255.0 above, back to [0,255]
        Lambda(preprocess_input) -> real Caffe-style/mean-subtracted prep
    Both run automatically inside model.predict(). Applying either again
    here would double-preprocess and silently corrupt every prediction.
    """
    img = pil_image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(pil_image: Image.Image) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_artifacts() first")
    x = preprocess_image(pil_image)
    probs = _model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)
        },
    }

# ---------------------------------------------------------------
# Two-gate rejection logic for out-of-scope input.
# Added after observing a live failure case: a screenshot of a
# Google image search for "dog" was classified as Common Rust at
# 85% confidence. As a closed-set softmax classifier, the model
# has no built-in way to say "none of these" — it will always
# force any input into one of its 4 trained classes. These gates
# are a coarse, heuristic mitigation, not a trained OOD detector.
# ---------------------------------------------------------------

LEAF_MIN_CENTER_GREEN_RATIO = 0.25
LEAF_MIN_TEXTURE_STD = 12
CONFIDENCE_THRESHOLD = 0.60

def leaf_likeness_score(pil_image, crop_frac=0.5):
    """
    Coarse heuristic, NOT a trained classifier.

    Uses the CENTER crop of the image, not the whole frame — an early
    version used whole-image green ratio and was fooled by a photo of
    a dog standing on grass (whole-image green_ratio=0.497, easily
    above threshold). The dataset's leaf photos are close-ups where
    the leaf fills the frame center; unrelated photos with a green
    background typically have their actual subject in the center
    instead. Verified on real data: center_green_ratio was 0.532 for
    a genuine leaf vs 0.013 for the dog photo.

    Returns:
      center_green_ratio: fraction of green-dominant pixels in the
        center crop_frac of the image
      texture_std: whole-image pixel-value variance (catches flat/
        blank images)

    Known false-negative risk: a heavily diseased leaf with little
    green remaining, an off-center leaf, or unusual lighting/cropping
    may also fail this check despite being a genuine maize leaf. This
    is a stated limitation, not a guarantee.
    """
    img = pil_image.convert("RGB").resize((100, 100))
    arr = np.array(img).astype(int)
    h, w, _ = arr.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    center = arr[y0:y0 + ch, x0:x0 + cw]
    r, g, b = center[:, :, 0], center[:, :, 1], center[:, :, 2]
    green_mask = (g > r) & (g > b - 10) & (g > 50)
    center_green_ratio = float(green_mask.sum()) / green_mask.size
    texture_std = float(arr.std())
    return center_green_ratio, texture_std

def is_leaf_like(pil_image):
    center_green_ratio, texture_std = leaf_likeness_score(pil_image)
    return center_green_ratio >= LEAF_MIN_CENTER_GREEN_RATIO and texture_std >= LEAF_MIN_TEXTURE_STD

def classify(pil_image):
    """
    Full classification with rejection gates. Always returns a dict
    containing a "status" key:
      "no_match"     - failed the leaf-likeness heuristic (Gate 1)
      "unrecognised" - leaf-like, but below the confidence threshold (Gate 2)
      "ok"           - confident prediction among the 4 trained classes

    predict() itself is untouched and remains independently verified
    (Section 8) — this function only wraps it with pre/post checks.
    """
    if not is_leaf_like(pil_image):
        return {"status": "no_match"}

    result = predict(pil_image)
    result["status"] = "ok" if result["confidence"] >= CONFIDENCE_THRESHOLD else "unrecognised"
    return result
