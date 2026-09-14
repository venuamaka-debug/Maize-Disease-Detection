from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import io
from inference import load_artifacts, classify

app = Flask(__name__)
MODEL_READY = False
MAX_FILE_SIZE = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/health")
def health():
    return jsonify({"status": "ok" if MODEL_READY else "model_not_loaded"}), (200 if MODEL_READY else 503)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image before continuing."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Please select an image before continuing."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "This file type is not supported. Please upload a JPG, JPEG or PNG image."}), 400
    file.seek(0, io.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "The selected image is too large. Please choose a smaller image."}), 400
    try:
        img = Image.open(file.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "We couldn't read this image. Please try another image."}), 400
    try:
        result = classify(img)
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The analysis could not be completed. Please try again."}), 500
    return jsonify(result), 200

if __name__ == "__main__":
    MODEL_READY = load_artifacts()
    app.run(debug=True, host="0.0.0.0", port=5000)
