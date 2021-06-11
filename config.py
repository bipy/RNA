"""
Configuration
"""

'''General Config'''

MATRIX_SIZE = 256

EPOCH = 200

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

'''File Path'''


class Path:
    ALL_CT_FILES_PATH = "all_ct_files"

    CSV_ALL_PATH = "csv_all"

    CSV_SPLIT_PATH = "csv_split"

    CSV_FINAL_PATH = "csv_final"

    WEIGHT_FILE_PATH = "weight.npz"

    DATA_FILE_PATH = "data.npz"


'''Scripts IO Config'''


# data_csv_trans.py
class CsvTrans:
    SOURCE_PATH = Path.ALL_CT_FILES_PATH

    OUTPUT_PATH = Path.CSV_ALL_PATH


# data_csv_split.py
class CsvSplit:
    SOURCE_PATH = Path.CSV_ALL_PATH

    OUTPUT_PATH = Path.CSV_SPLIT_PATH


# data_weight.py
class Weight:
    SOURCE_PATH = Path.CSV_ALL_PATH

    OUTPUT_PATH = Path.WEIGHT_FILE_PATH


# data_clean.py
class Clean:
    SOURCE_PATH = Path.CSV_SPLIT_PATH

    OUTPUT_PATH = Path.CSV_FINAL_PATH


# data_matrix.py
class Matrix:
    SOURCE_PATH = Path.CSV_FINAL_PATH

    OUTPUT_PATH = Path.DATA_FILE_PATH

    WEIGHT_SOURCE_PATH = Path.WEIGHT_FILE_PATH


# method.py
class Method:
    SOURCE_PATH = Path.DATA_FILE_PATH
