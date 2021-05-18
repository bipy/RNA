import os

import method

'''
RNA 测试集
'''

SOURCE_PATH = "../all_ct_files"
OUTPUT_PATH = "../csv_test"


class Item:
    idx: int
    val: str
    pair: int

    def __init__(self, idx, val, pair):
        self.idx = idx
        self.val = val
        self.pair = pair


def output(cur_filename, rna):
    with open("{}/{}.csv".format(OUTPUT_PATH, cur_filename), "w+") as fout:
        for i in rna:
            print(f"{i.idx},{i.val},{(i.pair)}", file=fout)
    print(f"{cur_filename} finished!")


def ct2list(cur_file):
    rna = []
    with open("{}/{}".format(SOURCE_PATH, cur_file), "r") as source:
        label = True
        lines = source.readlines()
        for line in lines:
            if line[0] == "#":
                continue
            elif label:
                label = False
                continue
            cur_line_list = list(filter(None, line.replace(' ', '\t').split('\t')))
            rna.append(Item(int(cur_line_list[0]), cur_line_list[1].upper(), int(cur_line_list[4])))
    return rna


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for i in os.listdir(SOURCE_PATH):
        cur_filename = os.path.splitext(i)[0]
        output(cur_filename, ct2list(i))
    method.clean(OUTPUT_PATH)
