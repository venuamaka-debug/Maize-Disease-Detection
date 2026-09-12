import sys
from PIL import Image
from inference import load_artifacts, predict

load_artifacts()
img = Image.open(sys.argv[1])
result = predict(img)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
for cls, p in result["all_probabilities"].items():
    print(f"  {cls}: {p:.4f}")
