import os
import random

import numpy as np
import time
import tensorflow as tf
import shutil

import data_output_matrix
import model
import pydot

'''
临时测试工具
'''


class Node:
    cnt: int
    name: str

    def __init__(self, cnt, name):
        self.cnt = cnt
        self.name = name


def test1():
    SOURCE_PATH = "split_ct_output"
    maxl = -1
    length_list = []
    for i in os.listdir(SOURCE_PATH):
        # try:
        #     x_train.append(csv2matrix_x(i))
        #     y_train.append(csv2matrix_y(i))i
        # except ValueError:
        #     print(i)
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            length_list.append(len(source))
        except Exception as e:
            print(i)
            print(e)

    length_list = sorted(length_list, reverse=True)
    _512 = 0
    _256 = 0
    _128 = 0
    for i in length_list:
        if i < 512:
            _512 += 1
            if i < 256:
                _256 += 1
                if i < 128:
                    _128 += 1

    print(f"total = {len(length_list)}")
    print(f"_512 = {_512}")
    print(f"_256 = {_256}")
    print(f"_128 = {_128}")


def test2():
    print(f"{int(time.time())}")


def test3():
    source = np.loadtxt("split_ct_output/CRW_00746_7.csv", dtype=np.str, delimiter=',')
    seq = []
    output = []
    for i in source:
        seq.append(i[1])
        if int(i[2]) == 0:
            output.append('.')
        elif int(i[2]) > int(i[0]):
            output.append('(')
        elif int(i[2]) < int(i[0]):
            output.append(')')
    print("".join(seq))
    print("".join(output))


def test4():
    SOURCE_PATH = "split_ct_output"
    print(len(os.listdir(SOURCE_PATH)))


def test5():
    a = np.zeros((128, 128, 1), dtype=np.float)
    a[64, 64, 0] = 1
    b = tf.keras.activations.sigmoid(tf.constant(a, dtype=np.float, shape=(128, 128, 1)))
    c = tf.keras.activations.hard_sigmoid(tf.constant(a, dtype=np.float, shape=(128, 128, 1)))
    d = tf.keras.activations.softmax(tf.constant(a, dtype=np.float, shape=(128, 128, 1)))

    b = b.numpy()[:, :, 0]
    c = c.numpy()[:, :, 0]
    d = d.numpy()[:, :, 0]
    a = a[:, :, 0]
    print(a)


def test6():
    data_output_matrix.generate_data_list_16("csv_split_16", "data_train_16.npz")
    data_output_matrix.generate_data_list_16("csv_test_16", "data_test_16.npz")
    # data_output_matrix.generate_data_list("csv_split", "data_train.npz")
    # data_output_matrix.generate_data_list("csv_test", "data_test.npz")


def test7():
    if not os.path.exists("csv_split_16"):
        os.makedirs("csv_split_16")
    for file in os.listdir("csv_split"):
        if random.random() > 0.9:
            shutil.copyfile(f"csv_split/{file}", f"csv_split_16/{file}")
    if not os.path.exists("csv_test_16"):
        os.makedirs("csv_test_16")
    for file in os.listdir("csv_test"):
        if random.random() > 0.8:
            shutil.copyfile(f"csv_test/{file}", f"csv_test_16/{file}")


def test8():
    # vector_col = np.zeros((128, 4), dtype=np.float)
    vector_col = np.random.random((4, 4))
    vector_row = vector_col.T
    matrix_list = []
    for i in range(4):
        for j in range(4):
            matrix_list.append(vector_col[:, [i]] * vector_row[[j], :])
    matrix = np.array(matrix_list).swapaxes(2, 0)
    matrix2 = np.array(matrix_list)
    list_x = []
    list_x.append(matrix)
    list_x.append(matrix2)
    list_np = np.array(list_x)

    print("")


def test9():
    m = model.get_model((128, 128, 16))
    tf.keras.utils.plot_model(m, "model.png", show_shapes=True)


def test10():
    TP = 12517.0000
    FP = 2505.0000
    FN = 6947.0000
    R = TP / (TP + FN)
    P = TP / (TP + FP)
    F1 = (2 * P * R) / (R + P)
    print(f"R={R} - P={P} - F1={F1}")


if __name__ == "__main__":
    test10()
