import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import model
import model2
import model3

'''
模型处理相关方法
'''

MATRIX_SIZE = 128
x_list, y_list = [], []
x_test_list, y_test_list = [], []


def set_matrix_size(size):
    MATRIX_SIZE = size


def csv2matrix_x(source):
    seq = []
    for t in source:
        seq.append(t[1])
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float)
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            if (seq[i] == "C" and seq[j] == "G") or (seq[i] == "G" and seq[j] == "C"):
                matrix[i][j] = 3
            elif (seq[i] == "A" and seq[j] == "U") or (seq[i] == "U" and seq[j] == "A"):
                matrix[i][j] = 2
            elif (seq[i] == "G" and seq[j] == "U") or (seq[i] == "U" and seq[j] == "G"):
                matrix[i][j] = 1
    return matrix


def csv2matrix_y(source):
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float)
    for t in source:
        if t[2] != '0':
            first = min(int(t[0]), int(t[2]))
            second = max(int(t[0]), int(t[2]))
            matrix[first - 1][second - 1] = 1
    return matrix


def generate_data_list(SOURCE_PATH):
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                continue
            x_list.append(csv2matrix_x(source))
            y_list.append(csv2matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")


def generate_test_list(SOURCE_PATH):
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                continue
            x_test_list.append(csv2matrix_x(source))
            y_test_list.append(csv2matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")


def clean(SOURCE_PATH):
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                os.remove("{}/{}".format(SOURCE_PATH, i))
                continue
            x_list.append(csv2matrix_x(source))
            y_list.append(csv2matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")
            os.remove("{}/{}".format(SOURCE_PATH, i))
    x_list.clear()
    y_list.clear()


def train_model():
    x_train = np.array(x_list)
    y_train = np.array(y_list)
    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)

    keras.backend.clear_session()

    m = model3.get_model((MATRIX_SIZE, MATRIX_SIZE, 1))
    m.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # m.summary()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{now}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    m.fit(
        x=x_train,
        y=y_train,
        epochs=2,
        batch_size=8,
        validation_data=(x_test, y_test),
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
    test_y = csv2matrix_y(input_csv)
    input_x_list.append(csv2matrix_x(input_csv))
    rt = m.predict(np.array(input_x_list))
    test_x = rt[0].swapaxes(0, 2)[0]
    print(rt)

# if __name__ == "__main__":
#     train()
