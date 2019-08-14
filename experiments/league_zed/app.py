from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

import tensorflow as tf
import numpy as np
import sys

from .code.model import Model, get_action_name


graph = tf.get_default_graph()
model = Model(summary=False)
sys.stdout.flush()


app = Flask(__name__)
@app.route("/", methods=["POST"])
def predict_output():
    image = Image.open(BytesIO(request.data)).convert("RGB")
    if image.size != (768, 432):
        return jsonify({"error": "invalid image dimensions, must be 768x432"})
    image_data = (np.array(image).astype(np.float32) / 255)
    image_data = image_data.reshape((1,) + image_data.shape)

    global graph
    with graph.as_default():
        prediction = model.predict_output(image_data, batch_size=1)[0]

    result = {
        "x": float(prediction[0]),
        "y": float(prediction[1]),
        "action": get_action_name(prediction),
    }

    return jsonify(result)
