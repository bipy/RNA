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

MATRIX_SIZE = 256

x_list, y_list = [], []


def gen_matrix_x_16(source):
    seq = []
    for t in source:
        seq.append(t[1])
    vector_col = np.zeros((MATRIX_SIZE, 4), dtype=np.float)

    for i in range(len(seq)):
        if seq[i] == "A":
            vector_col[i, 0] = 1
        elif seq[i] == "U":
            vector_col[i, 1] = 1
        elif seq[i] == "C":
            vector_col[i, 2] = 1
        elif seq[i] == "G":
            vector_col[i, 3] = 1

    vector_row = vector_col.T
    matrix_list = []
    for i in range(4):
        for j in range(4):
            matrix_list.append(vector_col[:, [i]] * vector_row[[j], :])
    return np.array(matrix_list).swapaxes(0, 1).swapaxes(1, 2)


def gen_matrix_y_16(source):
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float)
    for t in source:
        if t[2] != '0':
            matrix[int(t[0]) - 1][int(t[2]) - 1] = 1
            matrix[int(t[2]) - 1][int(t[0]) - 1] = 1
    return matrix


def generate_data_list_16(SOURCE_PATH):
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                continue
            x_list.append(gen_matrix_x_16(source))
            y_list.append(gen_matrix_y_16(source))
        except ValueError:
            print(f"Value Error: skip {i}")


def gen_matrix_x(source):
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
                matrix[i][j] = 0.8
    return matrix


def gen_matrix_y(source):
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
            x_list.append(gen_matrix_x(source))
            y_list.append(gen_matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")


def clean(SOURCE_PATH):
    valid = {"A", "C", "G", "U"}
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                os.remove("{}/{}".format(SOURCE_PATH, i))
                continue
            for t in source:
                if t[1] not in valid:
                    raise ValueError
            x_list.append(gen_matrix_x(source))
            y_list.append(gen_matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")
            os.remove("{}/{}".format(SOURCE_PATH, i))
    x_list.clear()
    y_list.clear()


def train_model():
    np_x = np.array(x_list)
    np_y = np.array(y_list)

    keras.backend.clear_session()

    m = model.get_model((MATRIX_SIZE, MATRIX_SIZE, 16))
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
        x=np_x,
        y=np_y,
        epochs=200,
        batch_size=16,
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
    test_y = gen_matrix_y(input_csv)
    input_x_list.append(gen_matrix_x(input_csv))
    input_np = np.array(input_x_list)
    rt = m.predict(input_np)
    test_x = rt[0, :, :, 0]
    print(rt)

# if __name__ == "__main__":
#     train()
