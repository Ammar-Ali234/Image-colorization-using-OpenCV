from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
import os

app = Flask(__name__)

# ── Model file paths (adjust if needed) ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROTO_FILE = os.path.join(BASE_DIR, "model", "colorization_deploy_v2.prototxt")
MODEL_FILE = os.path.join(BASE_DIR, "model", "colorization_release_v2.caffemodel")
PTS_FILE   = os.path.join(BASE_DIR, "model", "pts_in_hull.npy")

# ── Load model once at startup ────────────────────────────────────────────────
pts_in_hull = np.load(PTS_FILE)
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)

pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs   = [pts.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

print("✅ Colorization model loaded.")


def colorize(image_bytes: bytes) -> bytes:
    """
    Accept raw image bytes, run the colorization model,
    and return colorized image as JPEG bytes.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image. Make sure it is a valid image file.")

    # Convert to LAB
    frame_float = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(frame_float, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Resize L channel to 224×224 for model input
    L_resized = cv2.resize(L, (224, 224)) - 50

    # Forward pass
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab_decoded = net.forward()[0].transpose((1, 2, 0))

    # Resize ab channels back to original size and merge with L
    ab_resized = cv2.resize(ab_decoded, (frame.shape[1], frame.shape[0]))
    lab_out    = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)

    # Convert LAB → BGR
    colorized = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    output    = (colorized * 255).astype("uint8")

    # Encode result as JPEG bytes
    success, buf = cv2.imencode(".jpg", output)
    if not success:
        raise RuntimeError("Failed to encode colorized image.")
    return buf.tobytes()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "model": "colorization_release_v2"})


@app.route("/colorize", methods=["POST"])
def colorize_endpoint():
    """
    POST /colorize
    Accepts a multipart/form-data upload with field name 'image'.
    Returns the colorized image as image/jpeg.

    Example (curl):
        curl -X POST http://localhost:5000/colorize \
             -F "image=@photo.jpg" \
             --output colorized.jpg
    """
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes    = file.read()
        colorized_jpeg = colorize(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return send_file(
        io.BytesIO(colorized_jpeg),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="colorized.jpg",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)