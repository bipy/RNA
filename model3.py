import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
U-Net 模型
'''


def get_model(shape):
    # ================= 输入层 =================
    # 将数据整形为特定格式 L × L × 1 进行处理
    inputs = keras.Input(shape=shape)

    outputs = layers.Conv2D(kernel_size=1, strides=1, filters=1)(inputs)

    # layers.Dense(units=10, activation='softmax')
    model = keras.Model(inputs, outputs, name='u-net')
    return model
