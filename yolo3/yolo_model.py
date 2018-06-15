import os
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
from .model import yolo_eval, yolo_body, tiny_yolo_body
from .utils import letterbox_image


class YOLO:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.score = 0.3
        self.iou = 0.45
        self.class_names = ['face']
        self.anchors = np.array([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]).reshape(-1, 2)
        self.sess = K.get_session()
        self.model_image_size = (416, 416)
        self.boxes, self.scores, self.classes = self.generate()

    def generate(self):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        self.yolo_model = load_model(self.path_to_model, compile=False)
        assert self.yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        boxed_image = letterbox_image(image, tuple(reversed((416, 416))))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        predictions = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            predictions.append([int(left), int(top), int(right), int(bottom)]) # removed score
        return predictions
    
    def close_session(self):
        self.sess.close()