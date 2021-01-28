import os
import numpy as np

SOURCE_PATH = "all_ct_files"


def pre_process_x(cur_file):
    ans = []
    sep = "\t"
    blank_sep_list = ["ASE", "TMR", "SRP"]
    if cur_file.split("_")[0] in blank_sep_list:
        sep = " "
    with open("{}/{}".format(SOURCE_PATH, cur_file), "r") as source:
        label = True
        lines = source.readlines()
        for line in lines:
            if line[0] == "#":
                continue
            elif label:
                label = False
                continue
            ans.append(list(filter(None, line.split(sep)))[1].upper())

    cur_length = len(ans)
    matrix = np.zeros((cur_length, cur_length), dtype=np.int)
    for i in range(cur_length):
        for j in range(cur_length):
            if (ans[i] == "C" and ans[j] == "G") or (ans[i] == "G" and ans[i] == "C"):
                matrix[i][j] = matrix[j][i] = 3
            elif (ans[i] == "A" and ans[j] == "U") or (ans[i] == "U" and ans[i] == "A"):
                matrix[i][j] = matrix[j][i] = 2
            elif (ans[i] == "G" and ans[j] == "U") or (ans[i] == "U" and ans[i] == "G"):
                matrix[i][j] = matrix[j][i] = 1

    with open("output.x", "w+") as fout:
        for i in matrix:
            fout.write()


pre_process_x("ASE_00001.ct")
