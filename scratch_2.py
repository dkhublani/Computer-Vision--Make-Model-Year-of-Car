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



def find_object():
    global graph
    with graph.as_default():
        data = request.get_json()
        #b64img = Image.open(BytesIO(base64.b64decode(data['b64_image'])))
        #img = pil_to_ndarray(b64img)
        #print(img.shape)
        img= cv2.imread(data['b64_image'])

        objs = tfnet.return_predict(img)