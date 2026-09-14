import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import colorsys

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)

train = pd.read_csv(PROJECT_ROOT / "data/processed/splits/train.csv")
train["image_path"] = train["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

gls_mendeley = train[
    (train["class_name"] == "Gray_Leaf_Spot") &
    (train["image_path"].str.contains("Mendeley"))
].reset_index(drop=True)

print(f"Scoring {len(gls_mendeley)} Mendeley Gray_Leaf_Spot images...")

def lesion_pixel_fraction(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((100, 100))
        arr = np.array(img) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Lesion colors: tan/brown/gray — roughly where R and G are close
        # and both notably higher than a "pure green" leaf pixel, or all
        # channels are muted/grayish (low saturation, mid brightness)
        is_greenish = (g > r * 1.15) & (g > b * 1.15)
        is_lesion = ~is_greenish

        return is_lesion.mean()
    except Exception:
        return -1  # flag unreadable files separately

scores = []
for i, row in gls_mendeley.iterrows():
    scores.append(lesion_pixel_fraction(row["image_path"]))
    if i % 300 == 0:
        print(f"  {i}/{len(gls_mendeley)}...")

gls_mendeley["lesion_score"] = scores

# Save full scored list for inspection
out_path = PROJECT_ROOT / "gls_mendeley_scored.csv"
gls_mendeley.to_csv(out_path, index=False)

print(f"\n✓ Done. Saved: {out_path}")
print(f"\nScore distribution:")
print(gls_mendeley["lesion_score"].describe())
