import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)

train = pd.read_csv(PROJECT_ROOT / "data/processed/splits/train.csv")
train["image_path"] = train["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

grayleaf = train[train["class_name"] == "Gray_Leaf_Spot"].sample(20, random_state=7)

fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, (_, row) in enumerate(grayleaf.iterrows()):
    img = Image.open(row["image_path"])
    ax = axes[i // 5, i % 5]
    ax.imshow(img)
    source = "Mendeley" if "Mendeley" in row["image_path"] else "PlantVillage"
    ax.set_title(source, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("grayleaf_sample_20.png", dpi=100)
print("Saved: grayleaf_sample_20.png")
