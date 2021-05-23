import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import config
import model

'''
模型处理相关方法
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
        loss='binary_crossentropy',
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
    # m.summary()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{now}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    m.fit(
        x=data["x"],
        y=data["y"],
        epochs=config.EPOCH,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        validation_batch_size=config.VALIDATION_BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )
    m.save(f"model_{now}")


def model_summary():
    m = gen_model()
    m.summary()

# def model_predict(model_path, input_file_path):
#     m = keras.models.load_model(model_path)
#     input_x_list = []
#     input_csv = np.loadtxt(input_file_path, dtype=np.str, delimiter=',')
#     test_y = gen_matrix_y(input_csv)
#     input_x_list.append(gen_matrix_x(input_csv))
#     input_np = np.array(input_x_list)
#     rt = m.predict(input_np)
#     test_x = rt[0, :, :, 0]
#     print(rt)

# if __name__ == "__main__"
#     train()
