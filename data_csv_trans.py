import os
import config

'''
RNA CT 文件转 CSV
'''

SOURCE_PATH = config.CsvTrans.SOURCE_PATH
OUTPUT_PATH = config.CsvTrans.OUTPUT_PATH


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
            print(f"{i.idx},{i.val if i.val.isalpha() else ''},{i.pair}", file=fout)


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


def travel():
    if not os.path.exists(SOURCE_PATH):
        raise Exception(f"SOURCE PATH {SOURCE_PATH} IS NOT EXIST")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    files = os.listdir(SOURCE_PATH)
    for i, file in enumerate(files):
        cur_filename = os.path.splitext(file)[0]
        output(cur_filename, ct2list(file))
        print("\rPREPROCESS 1/5: TRANS - {}%".format(round((i + 1) * 100 / len(files))), end='')
    print('')


if __name__ == '__main__':
    travel()
