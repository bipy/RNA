import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import config
import model

'''
模型相关方法
'''

MATRIX_SIZE = config.MATRIX_SIZE

SOURCE_PATH = config.Method.SOURCE_PATH

acgu = {"A": 0, "C": 1, "G": 2, "U": 3}


def load_data():
    if os.path.exists(SOURCE_PATH):
        return np.load(SOURCE_PATH)
    raise Exception("DATA IS NOT EXIST")


def gen_model():
    keras.backend.clear_session()

    m = model.get_model((MATRIX_SIZE, MATRIX_SIZE, 1))
    m.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )
    return m


def train_model():
    data = load_data()
    m = gen_model()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{now}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    m.fit(
        x=data["x"],
        y=data["y"],
        epochs=config.EPOCH,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[tensorboard_callback]
    )
    m.save(f"model_{now}")


def model_summary():
    m = gen_model()
    m.summary()

