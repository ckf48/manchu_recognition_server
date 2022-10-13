import os
import sys
import cv2
from flask import request, Flask, jsonify
import base64

from werkzeug.utils import secure_filename

from helper import preprocess
from modelatt并联备份2 import Model
from dataloader import MiniBatch
from segment import prepareImg, wordSegmentation
import time
import numpy as np

app = Flask(__name__)

model = None


def predict(model, word):
    ''' predict and return text recognized '''
    img = preprocess(word, Model.imgSize)
    minibatch = MiniBatch(None, [img])
    recognized = model.inferBatch(minibatch)
    return recognized[0]


def main(filepath):
    # with open('../data/test2000.txt', 'r', encoding='utf-8') as f:
    #     charList = f.read()
    # charList = list(charList)
    # model = Model(charList, restore=True)
    img = prepareImg(cv2.imread(filepath), 500)
    result = wordSegmentation(img, kernelSize=25, sigma=11, theta=4, minArea=500)

    recognized = str()
    draw = []
    for line in result:
        if len(line):
            for (_, w) in enumerate(line):
                (wordBox, wordImg) = w
                recognized += predict(model, wordImg) + ' '
                draw.append(wordBox)
            recognized += '\n'

    return recognized


@app.route('/')
def hello_world():  # put application's code here
    return 'Index Page!'


@app.route("/api/upload", methods=['POST', 'GET'])
def upload():
    img_data = base64.b64decode(request.form['image'].split(',')[1])
    bs = np.asarray(bytearray(img_data), dtype='uint8')
    filename = request.form['fileName']

    basepath = os.path.dirname(__file__)  # 当前文件所在路径

    upload_path = os.path.join(basepath, 'static/images', secure_filename(filename))
    image_data = cv2.imdecode(bs, cv2.IMREAD_COLOR)
    cv2.imwrite(upload_path, image_data)
    return main(upload_path)


if __name__ == '__main__':
    with open('../data/test2000.txt', 'r', encoding='utf-8') as f:
        charList = f.read()
    charList = list(charList)
    model = Model(charList, restore=True)
    app.run(host='0.0.0.0', port=9000, debug=True)
