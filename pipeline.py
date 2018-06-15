import os
import cv2

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from yolo3.utils import letterbox_image

def pipe_the_image(img_path, classifier, yolo):
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = { 0: 'keke', 1: 'ostr', 2: 'otsochodzi', 3: 'taco_hemingway' }
    image = Image.open(img_path)
    image_data = np.array(image, dtype='uint8')
    predictions = yolo.detect_image(image)
    for box in predictions:
        left, top, right, bottom = box
        print(left, top, right, bottom)
        print(image_data.shape)
        image_bbox = image_data[top:bottom, left:right]
        image_bbox = letterbox_image(Image.fromarray(image_bbox), tuple(reversed((224, 224))))
        image_bbox = np.expand_dims(np.array(image_bbox, dtype='uint8'), axis=0)
        which_rapper = classifier.predict(image_bbox)
        which_rapper = which_rapper.argmax(axis=-1)[0]
        cv2.rectangle(image_data, (left, top), (right, bottom), (255,0,0), 2)
        cv2.putText(image_data,labels[which_rapper],(left,top), font, 1, (200,255,155), 2, cv2.LINE_AA)
    plt.imshow(image_data)
    plt.show()
