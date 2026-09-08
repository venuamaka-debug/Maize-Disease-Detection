import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
out_dir = PROJECT_ROOT / "data/processed/splits_cleaned"

# MUST match src.config.settings Settings().unified_classes order exactly
PIPELINE_CLASS_ORDER = [
    "Northern_Corn_Leaf_Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

train = pd.read_csv(out_dir / "train.csv")
y = train["class_name"].values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(PIPELINE_CLASS_ORDER),
    y=y
)

class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

print("Class weights (correct pipeline order):")
for i, c in enumerate(PIPELINE_CLASS_ORDER):
    print(f"  {i}: {c} -> {weights[i]:.4f}")

with open(out_dir / "class_weights.json", "w") as f:
    json.dump({
        "class_order": PIPELINE_CLASS_ORDER,
        "weights_by_idx": class_weight_dict
    }, f, indent=2)

print(f"\n✓ Corrected weights saved to {out_dir / 'class_weights.json'}")
