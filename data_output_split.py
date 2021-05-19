import os
import data_output_matrix

'''
RNA CT 文件分割 & 转换为 CSV
'''

SOURCE_PATH = "all_ct_files"
OUTPUT_PATH = "csv_split"


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
            print(f"{rna[i].idx - begin + 1},{rna[i].val},{(rna[i].pair - begin + 1) if rna[i].pair != 0 else 0}",
                  file=fout)
    print(f"{cur_filename}_{count} finished!")


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
        split_segment(cur_filename, ct2list(i))
    # data_output_matrix.clean(OUTPUT_PATH)
    data_output_matrix.generate_data_list(OUTPUT_PATH, "data_train.npz")

