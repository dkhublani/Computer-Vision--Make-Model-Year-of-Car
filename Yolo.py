from flask import Flask, Response, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from skimage.io import imread
from skimage.io._plugins.pil_plugin import pil_to_ndarray
from skimage.transform import resize
import math
import json
from keras.models import *
import tensorflow as tf
from json import dumps
from skimage.transform import resize
from skimage.io import imread
from darkflow.net.build import TFNet
import cv2


with open("../artifact/class_index_full.json") as f:
    car_dict = json.load(f)

with open("../artifact/model_architecture.json", "r") as f:
    json_model = json.load(f)

auto_model = model_from_json(json_model)

auto_model.load_weights("../artifact/model_weight.h5")

options = {"model": "../artifact/yolo-voc.cfg", "load": "../artifact/yolo-voc.weights", "threshold": 0.35}

tfnet = TFNet(options)

graph = tf.get_default_graph()
app = Flask(__name__)


@app.route('/find_objects', methods=["POST"])
def find_object():
    global graph
    with graph.as_default():
        data = request.get_json()
        b64img = Image.open(BytesIO(base64.b64decode(data['b64_image'])))
        img = pil_to_ndarray(b64img)

        objs = tfnet.return_predict(img)

        for obj in objs:
            obj['spec'] = None
            obj['confidence'] = str(obj['confidence'])
            if obj['label'] == 'car':
                car = extract_car(img, obj)
                car = resize(car, (299, 299))

                classe = np.argmax(pred[0])

                obj['spec'] = {"Make": car_dict[str(classe)]['Make'],
                               "Model": car_dict[str(classe)]['Model'],
                               "Prob": str(np.max(pred[0]))}

        resp = jsonify(objs)

        # resp = jsonify("")

        return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001, threaded=True)
