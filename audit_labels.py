import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys

PROJECT_ROOT = Path("/home/victor-e/maize-disease-detection")
OLD_PREFIX = "/content/maize-disease-detection"
NEW_PREFIX = str(PROJECT_ROOT)

TARGET_CLASS = sys.argv[1] if len(sys.argv) > 1 else "Gray_Leaf_Spot"
TARGET_SOURCE = sys.argv[2] if len(sys.argv) > 2 else "Mendeley"
N_SAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 300

train = pd.read_csv(PROJECT_ROOT / "data/processed/splits/train.csv")
train["image_path"] = train["image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

subset = train[
    (train["class_name"] == TARGET_CLASS) &
    (train["image_path"].str.contains(TARGET_SOURCE))
].reset_index(drop=True)

if len(subset) > N_SAMPLE:
    subset = subset.sample(N_SAMPLE, random_state=42).reset_index(drop=True)

print(f"Auditing {len(subset)} images — class={TARGET_CLASS}, source={TARGET_SOURCE}")
print("Controls: LEFT ARROW = keep, RIGHT ARROW = remove, ESC = save & quit early")

removed = []
idx = [0]

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

def show_image(i):
    ax.clear()
    from PIL import Image
    img = Image.open(subset.iloc[i]["image_path"])
    ax.imshow(img)
    ax.set_title(f"{i+1}/{len(subset)}  —  LEFT=keep  RIGHT=remove  ESC=save&quit", fontsize=11)
    ax.axis("off")
    fig.canvas.draw()

def on_key(event):
    if event.key == "left":
        idx[0] += 1
    elif event.key == "right":
        removed.append(subset.iloc[idx[0]]["image_path"])
        idx[0] += 1
    elif event.key == "escape":
        finish()
        return

    if idx[0] >= len(subset):
        finish()
        return
    show_image(idx[0])

def finish():
    out_path = PROJECT_ROOT / f"removed_{TARGET_CLASS}_{TARGET_SOURCE}.txt"
    with open(out_path, "w") as f:
        f.write("\\n".join(removed))
    print(f"\\n✓ Reviewed {idx[0]} images, flagged {len(removed)} for removal")
    print(f"✓ Saved list to: {out_path}")
    plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)
show_image(0)
plt.show()
