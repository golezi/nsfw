# -*-coding:utf-8-*-

import json
import requests

_IMAGE_SIZE = 64
SERVER_URL = 'http://localhost:8501/v1/models/nsfw:predict'
_LABEL_MAP = {0: 'drawings', 1: 'hentai', 2: 'neutral', 3: 'porn', 4: 'sexy'}

from PIL import Image
import numpy as np
import random
import os


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img.load()
    data = np.asarray(img, dtype="float32")
    data = standardize(data)
    data = data.astype(np.float16, copy=False)
    return data


def nsfw_predict(image_data):
    pay_load = json.dumps({"inputs": [image_data.tolist()]})
    response = requests.post(SERVER_URL, data=pay_load)
    data = response.json()
    outputs = data['outputs']
    predict_result = {"classes": _LABEL_MAP.get(outputs['classes'][0]),
                      'probabilities': {_LABEL_MAP.get(i): l for i, l in enumerate(outputs['probabilities'][0])}}
    return predict_result


def is_nsfw(image_url):
    print("is_nsfw", image_url)
    image_path = f"/tmp/{random.random()}.jpg"
    with open(image_path, "wb") as f:
        f.write(requests.get(image_url).content)
    image_data = load_image(image_path)
    predict = nsfw_predict(image_data)
    os.remove(image_path)
    print(predict)
    return predict


if __name__ == '__main__':
    print(is_nsfw("https://example.com/1.jpg"))
