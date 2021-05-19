import method

'''
主程序 TensorFlow 2.*
'''

SOURCE_PATH = "csv_split"

if __name__ == "__main__":
    method.generate_data_list(SOURCE_PATH)
    # method.generate_data_list_16(SOURCE_PATH)
    method.train_model()

    # method.model_summary()

    # method.model_predict("model_20210519-183540", "csv_test/CRW_01500.csv")
