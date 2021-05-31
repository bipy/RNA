import numpy as np
import os

'''
RNA 序列长度计算
'''
SOURCE_PATH = "csv_split"


def count_length():
    length_list = []
    for i in os.listdir(SOURCE_PATH):
        try:
            source = np.loadtxt("{}/{}".format(SOURCE_PATH, i), dtype=np.str, delimiter=',')
            length_list.append(len(source))
        except Exception as e:
            print(f"{i}: {e}")
    return sorted(length_list)


if __name__ == '__main__':
    length_list = count_length()
    with open(f"{SOURCE_PATH}_count.txt", "w+") as fout:
        for i in length_list:
            print(i, file=fout)
