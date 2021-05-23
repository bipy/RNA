import os
import numpy as np

import config

acgu = {"A": 0, "C": 1, "G": 2, "U": 3}

SOURCE_PATH = config.Matrix.SOURCE_PATH

WEIGHT_PATH = config.Matrix.WEIGHT_SOURCE_PATH

OUTPUT_PATH = config.Matrix.OUTPUT_PATH

MATRIX_SIZE = config.MATRIX_SIZE


def load_weight():
    if os.path.exists(WEIGHT_PATH):
        return np.load(WEIGHT_PATH)["data"]
    raise Exception("WEIGHT IS NOT EXIST")


def gen_matrix_x(source, weight):
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float)
    for i in range(len(source)):
        for j in range(i, len(source)):
            matrix[i][j] = weight[acgu[source[i][1]]][acgu[source[j][1]]]
    return matrix


def gen_matrix_y(source):
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float)
    for t in source:
        if t[2] != '0':
            first = min(int(t[0]), int(t[2]))
            second = max(int(t[0]), int(t[2]))
            matrix[first - 1][second - 1] = 1
    return matrix


def save(x_list, y_list):
    np_x = np.array(x_list)
    np_y = np.array(y_list)
    np.savez(OUTPUT_PATH, x=np_x, y=np_y)


def travel():
    x_list, y_list = [], []
    weight = load_weight()
    files = os.listdir(SOURCE_PATH)
    for i, file in enumerate(files):
        source = np.loadtxt("{}/{}".format(SOURCE_PATH, file), dtype=np.str, delimiter=',')
        x_list.append(gen_matrix_x(source, weight))
        y_list.append(gen_matrix_y(source))
        print("\rPREPROCESS 5/5: MATRIX - {}%".format(round((i + 1) * 100 / len(files))), end='')
    print("\nSaving Data...")
    save(x_list, y_list)


if __name__ == '__main__':
    travel()
