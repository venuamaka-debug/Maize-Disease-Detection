import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sys

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)

TARGET_CLASS = sys.argv[1] if len(sys.argv) > 1 else "Gray_Leaf_Spot"
TARGET_SOURCE = sys.argv[2] if len(sys.argv) > 2 else "Mendeley"
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # which batch of 20 (0-indexed)
SEED = 42

train = pd.read_csv(PROJECT_ROOT / "data/processed/splits/train.csv")
train["image_path"] = train["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

subset = train[
    (train["class_name"] == TARGET_CLASS) &
    (train["image_path"].str.contains(TARGET_SOURCE))
].sample(frac=1, random_state=SEED).reset_index(drop=True)

start = BATCH * 20
end = start + 20
chunk = subset.iloc[start:end]

# Save the file list for this batch so removal-by-number maps correctly
chunk_list_path = PROJECT_ROOT / f"audit_batch_{TARGET_CLASS}_{TARGET_SOURCE}_{BATCH}.csv"
chunk[["image_path"]].to_csv(chunk_list_path, index=False)

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for i in range(20):
    ax = axes[i // 5, i % 5]
    if i < len(chunk):
        img = Image.open(chunk.iloc[i]["image_path"])
        ax.imshow(img)
        ax.set_title(f"#{i}", fontsize=14, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
out_png = f"audit_{TARGET_CLASS}_{TARGET_SOURCE}_batch{BATCH}.png"
plt.savefig(out_png, dpi=100)
print(f"✓ Saved {out_png}  (images {start}-{end-1} of {len(subset)} total)")
print(f"✓ Batch file list: {chunk_list_path}")
