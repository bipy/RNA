import os
import numpy as np
import config

SOURCE_PATH = config.Weight.SOURCE_PATH
OUTPUT_PATH = config.Weight.OUTPUT_PATH

acgu = {"A": 0, "C": 1, "G": 2, "U": 3}


def gen_weight():
    if not os.path.exists(SOURCE_PATH):
        raise Exception("SOURCE PATH IS NOT EXIST")
    weight = np.zeros(shape=(4, 4), dtype=np.float)
    files = os.listdir(SOURCE_PATH)
    for i, file in enumerate(files):
        source = np.loadtxt("{}/{}".format(SOURCE_PATH, file), dtype=np.str, delimiter=',')
        for t in source:
            if t[1] in acgu and t[2] != '0':
                pair = source[int(t[2]) - 1][1]
                if pair in acgu:
                    weight[acgu[pair], acgu[t[1]]] += 1
        print("\rPREPROCESS 3/5: WEIGHT - {}%".format(round((i + 1) * 100 / len(files))), end='')
    weight = (np.e - 1) * (weight / np.max(weight))
    weight = np.tanh(weight)
    np.savez(OUTPUT_PATH, data=weight)
    print('')


if __name__ == '__main__':
    gen_weight()
