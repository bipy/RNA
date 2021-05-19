import numpy as np
import os

MATRIX_SIZE = 128


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


def generate_data_list_16(SOURCE_PATH, save_name):
    x_list, y_list = [], []
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
    np_x = np.array(x_list)
    np_y = np.array(y_list)
    np.savez(save_name, x=np_x, y=np_y)


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


def generate_data_list(SOURCE_PATH, save_name):
    x_list, y_list = [], []
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
    np_x = np.array(x_list)
    np_y = np.array(y_list)
    np.savez(save_name, x=np_x, y=np_y)


def clean(SOURCE_PATH):
    x_list, y_list = [], []
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            if len(source) > MATRIX_SIZE:
                print("Sequence Too Long: skip " + i)
                os.remove("{}/{}".format(SOURCE_PATH, i))
                continue
            x_list.append(gen_matrix_x(source))
            y_list.append(gen_matrix_y(source))
        except ValueError:
            print(f"Value Error: skip {i}")
            os.remove("{}/{}".format(SOURCE_PATH, i))
    x_list.clear()
    y_list.clear()
