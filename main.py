import method

'''
主程序 TensorFlow 2.*
'''

if __name__ == "__main__":
    method.train_model("data_train.npz", "data_test.npz")

    # method.model_summary()

    # method.model_predict("model_20210519-183540", "csv_test/CRW_01500.csv")
