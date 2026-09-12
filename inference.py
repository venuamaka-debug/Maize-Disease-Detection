import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model/production_model.h5"
IMG_SIZE = (224, 224)  # confirmed: image.target_size in config_snapshot.yaml

# Confirmed from src/config/settings.py:unified_classes —
# derived from class_mapping's first-insertion order (dict.fromkeys),
# cross-checked against results.json per_class_metrics key order,
# the confusion matrices already recorded for this project, and
# trainer.py's own saved-order guard (line 318) which did not fire
# for this run.
# DO NOT reorder this list without re-deriving it from settings.py.
CLASS_NAMES = [
    "Northern_Corn_Leaf_Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
]

_model = None

def load_artifacts():
    global _model
    _model = load_model(MODEL_PATH)
    return _model is not None

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Confirmed against src/data/pipeline.py:374 —
        image = tf.cast(image, tf.float32) / 255.0
    Plain [0,1] rescale, NOT resnet50.preprocess_input.
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
