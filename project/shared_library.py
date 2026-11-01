import datetime, base64, io, os, errno, math
import keras
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import product


from PIL import Image

from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

import tensorflow as tf

from time import perf_counter

from wurlitzer import pipes

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

SEED = 42

CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

# 5000 images training dataset
# path to the binary train file with image data
TRAIN_DATA_PATH = "./stl-10/train_X.bin"
# path to the binary train file with labels
TRAIN_LABEL_PATH = "./stl-10/train_y.bin"
# 8000 images training dataset
# path to the binary test file with image data
TEST_DATA_PATH = "./stl-10/test_X.bin"
# path to the binary test file with labels
TEST_LABEL_PATH = "./stl-10/test_y.bin"

RESULTS_PATH = "/results"


def read_labels(path_to_labels):
    """
    return an array of labels from path to the binary file containing labels from the STL-10 dataset
    """
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def load_all_images_raw(path_to_data, path_to_labels):
    """
    Load STL-10 as Keras-ready images: (N, 96, 96, 3) uint8, upright,
    with labels in 0..9.
    """
    labels = np.fromfile(path_to_labels, dtype=np.uint8)

    expected = len(labels) * 3 * 96 * 96
    actual = os.path.getsize(path_to_data)
    if actual != expected:
        raise ValueError(f"{path_to_data}: {actual} bytes, expected {expected}")

    raw = np.memmap(
        path_to_data, dtype=np.uint8, mode="r", shape=(len(labels), 3, 96, 96)
    )

    # File layout is (N, C, x, y), column-major. Reorder to (N, y, x, C) = (N, H, W, C).
    images = np.ascontiguousarray(raw.transpose(0, 3, 2, 1))

    labels = labels.astype(np.int64) - 1  # 1..10 -> 0..9

    print("Image Shape: ", images.shape, images.dtype, images.min(), images.max())
    print("Labels: ", labels.shape, labels.min(), labels.max())
    return images, labels


def load_all_images_flat(path_to_data, path_to_labels):
    """
    Load add images from path, and make it available as flatten images with features.
    """
    labels = np.fromfile(path_to_labels, dtype=np.uint8)
    raw = np.memmap(
        path_to_data, dtype=np.uint8, mode="r", shape=(len(labels), 3, 96, 96)
    )
    images = raw.reshape(len(labels), -1).astype("float32") / 255.0  # scale to 0-1
    print("Image Shape: ", images.shape, images.dtype, images.min(), images.max())
    return images, labels


def num_2_text(label):
    match label:
        case 1:
            return "airplane"
        case 2:
            return "bird"
        case 3:
            return "car"
        case 4:
            return "cat"
        case 5:
            return "deer"
        case 6:
            return "dog"
        case 7:
            return "horse"
        case 8:
            return "monkey"
        case 9:
            return "ship"
        case 10:
            return "truck"


def yyyymmddhhmm():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")
