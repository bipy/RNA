import os
import numpy as np

import config

'''
RNA 分割
'''

SOURCE_PATH = config.CsvSplit.SOURCE_PATH
OUTPUT_PATH = config.CsvSplit.OUTPUT_PATH


class Item:
    idx: int
    val: str
    pair: int

    def __init__(self, idx, val, pair):
        self.idx = idx
        self.val = val
        self.pair = pair


def split_output(cur_filename, rna, begin, end, count):
    with open("{}/{}_{}.csv".format(OUTPUT_PATH, cur_filename, str(count)), "w+") as fout:
        for i in range(begin - 1, end):
            print(
                f"{rna[i].idx - begin + 1},{rna[i].val if rna[i].val.isalpha() else ''},{(rna[i].pair - begin + 1) if rna[i].pair != 0 else 0}",
                file=fout
            )


def split_segment(cur_filename, rna):
    begin, end = 0, 0
    stack = []
    count = 0
    for item in rna:
        if item.pair != 0:
            if stack:
                if stack[-1].pair == item.idx:
                    begin = stack[-1].idx
                    end = item.idx
                    stack.pop()
                elif item.pair > item.idx:
                    if end - begin != 0:
                        split_output(cur_filename, rna, begin, end, count)
                        count += 1
                        begin, end = 0, 0
                    stack.append(item)
                else:
                    return
            else:
                stack.append(item)
    if end - begin != 0:
        split_output(cur_filename, rna, begin, end, count)


def gen_list(source):
    rna = []
    for t in source:
        rna.append(Item(int(t[0]), t[1], int(t[2])))
    return rna


def travel():
    if not os.path.exists(SOURCE_PATH):
        raise Exception(f"SOURCE PATH {SOURCE_PATH} IS NOT EXIST")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    files = os.listdir(SOURCE_PATH)
    for i, csv in enumerate(files):
        cur_filename = os.path.splitext(csv)[0]
        source = np.loadtxt("{}/{}".format(SOURCE_PATH, csv), dtype=np.str, delimiter=',')
        split_segment(cur_filename, gen_list(source))
        print("\rPREPROCESS 2/5: SPLIT - {}%".format(round((i + 1) * 100 / len(files))), end='')
    print('')


if __name__ == '__main__':
    travel()
