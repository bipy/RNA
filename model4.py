import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
U-Net 模型
'''


def conv_block(inputs, filters):
    x = layers.Conv2D(kernel_size=3, strides=1, filters=filters, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(kernel_size=3, strides=1, filters=filters, padding='same')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    return outputs


def up_conv_block(inputs, filters):
    x = layers.UpSampling2D(size=(2, 2))(inputs)
    x = layers.Conv2D(kernel_size=3, strides=1, filters=filters, padding='same')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    return outputs


def get_model(shape):
    # ================= 输入层 =================
    # 将数据整形为特定格式 L × L × 1 进行处理
    inputs = keras.Input(shape=shape)

    # ================= 第一模块 =================
    # 输入矩阵大小： L * L
    # 输出矩阵大小： L/2 * L/2 * 32

    # 卷积模块
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 32
    # 即将输入矩阵卷积成对应过滤器个数的深度（每个过滤器中深度叠加）
    x = conv_block(inputs=inputs, filters=32)

    # 连接点
    p1 = x

    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第二模块 =================
    # 输入矩阵大小： L/2 * L/2 * 32
    # 输出矩阵大小： L/4 * L/4 * 64

    # 卷积模块
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 64
    x = conv_block(inputs=x, filters=64)

    # 连接点
    p2 = x

    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第三模块 =================
    # 输入矩阵大小： L/4 * L/4 * 64
    # 输出矩阵大小： L/8 * L/8 * 128

    # 卷积模块
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 128
    x = conv_block(inputs=x, filters=128)

    # 连接点
    p3 = x

    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第四模块 =================
    # 输入矩阵大小： L/8 * L/8 * 128
    # 输出矩阵大小： L/16 * L/16 * 256

    # 卷积模块
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 256
    x = conv_block(inputs=x, filters=256)

    # x = layers.Dropout(0.5)(x)

    # 连接点
    p4 = x

    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第五模块 =================
    # 输入矩阵大小： L/16 * L/16 * 256
    # 输出矩阵大小： L/16 * L/16 * 512
    # 无池化层

    # 卷积模块
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 512
    x = conv_block(inputs=x, filters=512)

    # x = layers.Dropout(0.5)(x)

    # ================= 第六模块 =================
    # 输入矩阵大小： L/16 * L/16 * 512
    # 输出矩阵大小： L/8 * L/8 * 256

    # 上卷积模块
    # L/16 * L/16 * 512 -> L/8 * L/8 * 512
    # L/8 * L/8 * 512 -> L/8 * L/8 * 256
    x = up_conv_block(inputs=x, filters=256)

    # merge
    x = layers.Concatenate(axis=3)([p4, x])

    # 卷积模块
    x = conv_block(inputs=x, filters=256)

    # ================= 第七模块 =================
    # 输入矩阵大小： L/8 * L/8 * 256
    # 输出矩阵大小： L/4 * L/4 * 128

    # 上卷积模块
    # L/8 * L/8 * 256 -> L/4 * L/4 * 256
    # L/4 * L/4 * 256 -> L/4 * L/4 * 128
    x = up_conv_block(inputs=x, filters=128)

    # merge
    x = layers.Concatenate(axis=3)([p3, x])

    # 卷积模块
    x = conv_block(inputs=x, filters=128)

    # ================= 第八模块 =================
    # 输入矩阵大小： L/4 * L/4 * 128
    # 输出矩阵大小： L/2 * L/2 * 64

    # 上卷积模块
    # L/4 * L/4 * 128 -> L/2 * L/2 * 128
    # L/2 * L/2 * 128 -> L/2 * L/2 * 64
    x = up_conv_block(inputs=x, filters=64)

    # merge
    x = layers.Concatenate(axis=3)([p2, x])

    # 卷积模块
    x = conv_block(inputs=x, filters=64)

    # ================= 第九模块 =================
    # 输入矩阵大小： L/2 * L/2 * 64
    # 输出矩阵大小： L * L * 2

    # 上卷积模块
    # L/2 * L/2 * 64 -> L * L * 64
    # L * L * 64 -> L * L * 32
    x = up_conv_block(inputs=x, filters=32)

    # merge
    x = layers.Concatenate(axis=3)([p1, x])

    # 卷积模块
    x = conv_block(inputs=x, filters=32)

    # 第四卷积层
    # L * L * 32 -> L * L * 2
    #x = conv_block(inputs=x, filters=2)

    # ================= 输出层 =================
    # 输入矩阵大小： L * L * 2
    # 输出矩阵大小： L * L * 1

    outputs = layers.Conv2D(kernel_size=1, strides=1, filters=1, activation=keras.activations.sigmoid)(x)
    model = keras.Model(inputs, outputs, name='u-net')
    return model
