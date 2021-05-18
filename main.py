import method

'''
主程序 TensorFlow 2.*
'''

SOURCE_PATH = "csv_split"
TEST_SOURCE_PATH = "csv_test"

if __name__ == "__main__":
    method.generate_data_list(SOURCE_PATH)
    method.generate_test_list(TEST_SOURCE_PATH)
    method.train_model()

    # method.model_summary()

    # method.model_predict("model_20210517-221531", "csv_split/CRW_01549_1.csv")
