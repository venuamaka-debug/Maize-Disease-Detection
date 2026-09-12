import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model/production_model.h5"
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
