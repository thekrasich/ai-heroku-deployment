from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import numpy as np

IMG_WIDTH = 150
IMG_HEIGHT = 150
LABELS = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street",
}

app = Flask(__name__)

model = load_model("my_model.keras")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image)
    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    predictions = predictions.tolist()
    maximum_element_index = np.argmax(predictions)
    preditction_label = LABELS[maximum_element_index]
    return jsonify({"predictions": predictions, "label": preditction_label})


if __name__ == "__main__":
    app.run()
