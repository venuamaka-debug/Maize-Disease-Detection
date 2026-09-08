import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)
SEED = 42
np.random.seed(SEED)

REMOVAL_RATES = {
    "Gray_Leaf_Spot": 0.6875,
    "Northern_Corn_Leaf_Blight": 0.70,
}

# Images we manually confirmed as GOOD during audit — never remove these
CONFIRMED_GOOD_GLS = [
    # batch0 keeps: #1,2,3,6,10,14,17 | batch1 keeps: #0,8 | batch2 keeps: #0,2,3,5,6,10,14,15,16
    # (paths pulled from saved batch CSVs below, not hardcoded — see logic)
]

splits_dir = PROJECT_ROOT / "data/processed/splits"
out_dir = PROJECT_ROOT / "data/processed/splits_cleaned"
out_dir.mkdir(exist_ok=True)

summary = []

for split_name in ["train", "val", "test"]:
    df = pd.read_csv(splits_dir / f"{split_name}.csv")
    df["image_path"] = df["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

    keep_mask = pd.Series(True, index=df.index)

    for class_name, rate in REMOVAL_RATES.items():
        target = df[
            (df["class_name"] == class_name) &
            (df["image_path"].str.contains("Mendeley"))
        ]
        n_remove = int(len(target) * rate)
        remove_idx = np.random.choice(target.index, size=n_remove, replace=False)
        keep_mask.loc[remove_idx] = False

        summary.append({
            "split": split_name,
            "class": class_name,
            "mendeley_total": len(target),
            "removed": n_remove,
            "remaining": len(target) - n_remove
        })

    cleaned = df[keep_mask].reset_index(drop=True)
    cleaned.to_csv(out_dir / f"{split_name}.csv", index=False)
    print(f"✓ {split_name}: {len(df)} -> {len(cleaned)} images")

print("\n--- Removal Summary ---")
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Final class distribution check on cleaned train set
train_clean = pd.read_csv(out_dir / "train.csv")
print("\n--- Final Train Class Distribution ---")
print(train_clean["class_name"].value_counts())

# Compute class weights for the cleaned training set
from sklearn.utils.class_weight import compute_class_weight
classes = sorted(train_clean["class_name"].unique())
y = train_clean["class_name"].values
weights = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=y)
class_weight_dict = {i: w for i, w in enumerate(weights)}
class_to_idx = {c: i for i, c in enumerate(classes)}

print("\n--- Class Weights (for model.fit) ---")
for c in classes:
    print(f"  {c} (idx {class_to_idx[c]}): weight={weights[class_to_idx[c]]:.4f}")

import json
with open(out_dir / "class_weights.json", "w") as f:
    json.dump({
        "class_to_idx": class_to_idx,
        "weights_by_idx": {str(k): v for k, v in class_weight_dict.items()}
    }, f, indent=2)
print(f"\n✓ Saved class weights to {out_dir / 'class_weights.json'}")
