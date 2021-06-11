import os
import shutil
import numpy as np

import config

MATRIX_SIZE = config.MATRIX_SIZE

SOURCE_PATH = config.Clean.SOURCE_PATH
OUTPUT_PATH = config.Clean.OUTPUT_PATH


def clean():
    if not os.path.exists(SOURCE_PATH):
        raise Exception("SOURCE PATH IS NOT EXIST")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    acgu = {"A", "C", "G", "U"}
    files = os.listdir(SOURCE_PATH)
    for i, file in enumerate(files):
        source = np.loadtxt("{}/{}".format(SOURCE_PATH, file), dtype=np.str, delimiter=',')
        if len(source) <= MATRIX_SIZE:
            valid = True
            for t in source:
                if t[1] not in acgu:
                    valid = False
                    break
            if valid:
                shutil.copyfile(f"{SOURCE_PATH}/{file}", f"{OUTPUT_PATH}/{file}")
        print("\rPREPROCESS 3/5: CLEAN - {}%".format(round((i + 1) * 100 / len(files))), end='')
    print('')


if __name__ == '__main__':
    clean()
