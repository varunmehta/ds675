import datetime, base64, gc, io, os, errno, math
import keras
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import product


from PIL import Image

from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

import tensorflow as tf

from time import perf_counter

from wurlitzer import pipes

from IPython.display import display

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


RUN_ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_id.txt")


def get_run_id(override=None):
    """Same ID for every notebook in one run: override if given, else run_id.txt, else a new one.

    Delete run_id.txt to start a new run.
    """
    if override:
        with open(RUN_ID_FILE, "w") as f:
            f.write(str(override))
        return override

    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    try:
        # "x" fails if the file exists, so notebooks started together share one ID
        with open(RUN_ID_FILE, "x") as f:
            f.write(run_id)
        return run_id
    except FileExistsError:
        with open(RUN_ID_FILE) as f:
            return f.read().strip()


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def results_dir(run_id, notebook):
    """results/{run_id}/{notebook}, created if missing. Every output of a notebook goes here."""
    path = os.path.join(RESULTS_DIR, str(run_id), notebook)
    os.makedirs(path, exist_ok=True)
    return path


def load_stl10_binary(data_path, label_path=None):
    """Load STL-10 as Keras-ready images (N, 96, 96, 3) uint8, with labels 0..9."""
    # Read images (stored in column-major order)
    with open(data_path, "rb") as f:
        everything = np.fromfile(f, dtype=np.uint8)
        # Images are stored as: channels, width, height
        images = np.reshape(everything, (-1, DEPTH, WIDTH, HEIGHT))
        # Transpose to Keras format: (batch, height, width, channels)
        images = np.transpose(images, (0, 3, 2, 1))

    if label_path:
        with open(label_path, "rb") as f:
            labels = np.fromfile(f, dtype=np.uint8)
        # Convert 1-indexed (1-10) to 0-indexed (0-9) for Keras cross-entropy
        labels = labels - 1
        return images, labels

    return images


def cross_validate_keras(build_fn, name, X_train, y_train, epochs, batch_size, run_id, output_dir, n_splits=5):
    """Stratified cross-validation on the training images. Scores are saved after every fold."""
    labels = y_train.argmax(axis=1)  # StratifiedKFold needs integer labels
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    output = os.path.join(output_dir, f"{name}_cv_{run_id}.csv")
    rows = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, labels), 1):
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + fold)
        model = build_fn()

        start_time = perf_counter()
        model.fit(
            X_train[train_idx],
            y_train[train_idx],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            callbacks=[
                keras.callbacks.CSVLogger(os.path.join(output_dir, f"{name}_cv_{run_id}_fold{fold}.csv"))
            ],
            verbose=2,
        )
        minutes = (perf_counter() - start_time) / 60

        loss, accuracy = model.evaluate(
            X_train[val_idx], y_train[val_idx], batch_size=batch_size, verbose=0
        )
        rows.append(
            {
                "fold": fold,
                "val_accuracy": accuracy,
                "val_loss": loss,
                "training_minutes": minutes,
            }
        )
        pd.DataFrame(rows).to_csv(output, index=False)  # flush to disk immediately

        # free this fold's model before the next one is built, otherwise both stay in memory
        del model
        gc.collect()
        keras.backend.clear_session()

    df = pd.DataFrame(rows)
    display(df)
    print(f"{name}: {df['val_accuracy'].mean():.4f} +/- {df['val_accuracy'].std():.4f}")
    return df


def fit_test(model, name, X_test, y_test, training_minutes, batch_size, run_id, output_dir):
    """Score a final model on the 8,000 test images and save the result."""
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    pd.DataFrame(
        [
            {
                "test_accuracy": accuracy,
                "test_loss": loss,
                "training_minutes": training_minutes,
            }
        ]
    ).to_csv(os.path.join(output_dir, f"{name}_final_{run_id}.csv"), index=False)
    print(f"{name}: test accuracy {accuracy:.4f}, test loss {loss:.4f}")
