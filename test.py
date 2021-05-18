import os
import numpy as np
import time

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
    a = np.random.random((1, 128, 128, 1))
    b = a[0]
    b = b.swapaxes(0, 2)[0]
    print(a)


if __name__ == "__main__":
    test5()
