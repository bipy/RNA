import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import model
import model2
import model3
import data_output_matrix

'''
模型处理相关方法
'''

MATRIX_SIZE = 128


def train_model(train_npz, test_npz):
    data_train = np.load(train_npz)
    # data_test = np.load(test_npz)

    keras.backend.clear_session()

    m = model2.get_model((MATRIX_SIZE, MATRIX_SIZE, 1))
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
    # m.summary()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{now}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    m.fit(
        x=data_train["x"],
        y=data_train["y"],
        epochs=20,
        batch_size=8,
        # validation_data=(data_test["x"], data_test["y"]),
        validation_split=0.2,
        callbacks=[tensorboard_callback]
    )
    m.save(f"model_{now}")


def model_summary():
    m = model.get_model((MATRIX_SIZE, MATRIX_SIZE, 1,))
    m.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()


def model_predict(model_path, input_file_path):
    m = keras.models.load_model(model_path)
    input_x_list = []
    input_csv = np.loadtxt(input_file_path, dtype=np.str, delimiter=',')
    test_y = data_output_matrix.gen_matrix_y(input_csv)
    input_x_list.append(data_output_matrix.gen_matrix_x(input_csv))
    input_np = np.array(input_x_list)
    rt = m.predict(input_np)
    test_x = rt[0, :, :, 0]
    print(rt)

# if __name__ == "__main__":
#     train()
