import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")

def lesion_pixel_fraction(img_path):
    img = Image.open(img_path).convert("RGB").resize((100, 100))
    arr = np.array(img) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    is_greenish = (g > r * 1.15) & (g > b * 1.15)
    return (~is_greenish).mean()

# Manual labels from our 3 hand-reviewed batches (0=keep, 1=remove)
manual = {
    0: [1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1],
    1: [0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1],
    2: [0,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1],
}

for batch_num, labels in manual.items():
    chunk = pd.read_csv(PROJECT_ROOT / f"audit_batch_Gray_Leaf_Spot_Mendeley_{batch_num}.csv")
    scores = [lesion_pixel_fraction(p) for p in chunk["image_path"]]
    print(f"\n--- Batch {batch_num} ---")
    for i, (s, lbl) in enumerate(zip(scores, labels)):
        tag = "REMOVE" if lbl else "keep"
        print(f"#{i:2d}  score={s:.3f}  manual={tag}")
