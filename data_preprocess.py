import data_csv_trans
import data_csv_split
import data_weight
import data_clean
import data_matrix


def initialize():
    try:
        data_csv_trans.travel()

        data_csv_split.travel()

        data_weight.gen_weight()

        data_clean.clean()

        data_matrix.travel()

        print('\n' + '=' * 13 + ' PREPROCESS FINISHED ' + '=' * 13)
    except Exception as e:
        print(f"PREPROCESS ERROR: {e}")
    

if __name__ == '__main__':
    initialize()
