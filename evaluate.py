import os

import numpy as np
from tqdm import tqdm
from PIL import Image


def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih
    return intersection / ua  


def start_eval():
    with open('annots/validation.txt') as f:
        lines = f.readlines()
    from yolo3 import yolo_model
    yolo = yolo_model.YOLO()
    precision = []
    all_labels_count = 0
    for img in tqdm(lines):
        img = img.replace('\n', '').split(' ')
        filename, boxes = img[0], img[1:]
        image = Image.open(filename)
        image = yolo.detect_image(image)
        image.reverse()
        i = 0
        matched_preds = []
        for box in boxes:
            all_labels_count += 1
            box = box.split(',')
            box = [int(val) for val in box]
            box.pop() # removed score
            if i < len(boxes) and i < len(image):
                for img in image:
                    if img in matched_preds:
                        continue
                    curr_truth = np.array([box])
                    curr_pred = np.array([img])
                    curr_precision = compute_overlap(curr_truth, curr_pred)
                    if curr_precision > 0:
                        matched_preds.append(img)
                        precision.append(curr_precision[0][0])
                        i += 1
                        print("Success matching {} {}".format(curr_pred, curr_truth))
                        break
                    else:
                        print("Failed matching {} {}".format(curr_pred, curr_truth))
    print("Achieved score (found face vs. all faces) {} {}".format(len(precision), all_labels_count))
    print("Achieved precision for found faces (recall): {}".format(np.mean(precision)))

