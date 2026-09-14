import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)

train = pd.read_csv(PROJECT_ROOT / "data/processed/splits/train.csv")
train["image_path"] = train["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

first_path = Path(train.iloc[0]["image_path"])
print("Path exists:", first_path.exists(), "->", first_path)

blight = train[train["class_name"] == "Northern_Corn_Leaf_Blight"].sample(6, random_state=1)
grayleaf = train[train["class_name"] == "Gray_Leaf_Spot"].sample(6, random_state=1)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, (_, row) in enumerate(blight.iterrows()):
    img = Image.open(row["image_path"])
    axes[0, i].imshow(img)
    axes[0, i].set_title("Blight", fontsize=10)
    axes[0, i].axis("off")

for i, (_, row) in enumerate(grayleaf.iterrows()):
    img = Image.open(row["image_path"])
    axes[1, i].imshow(img)
    axes[1, i].set_title("Gray_Leaf_Spot", fontsize=10)
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("blight_vs_grayleaf_spotcheck.png", dpi=120)
print("Saved: blight_vs_grayleaf_spotcheck.png")
