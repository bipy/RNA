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

    # ================= 第一模块 =================
    # 输入矩阵大小： L * L
    # 输出矩阵大小： L/2 * L/2 * 32

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 32
    # 即将输入矩阵卷积成对应过滤器个数的深度（每个过滤器中深度叠加）
    x = layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p1 = x
    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第二模块 =================
    # 输入矩阵大小： L/2 * L/2 * 32
    # 输出矩阵大小： L/4 * L/4 * 64

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 64
    x = layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p2 = x
    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第三模块 =================
    # 输入矩阵大小： L/4 * L/4 * 64
    # 输出矩阵大小： L/8 * L/8 * 128

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 128
    x = layers.Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p3 = x
    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第四模块 =================
    # 输入矩阵大小： L/8 * L/8 * 128
    # 输出矩阵大小： L/16 * L/16 * 256

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 256
    x = layers.Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    p4 = x
    # 池化层，选择 2 × 2 最大池化
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # ================= 第五模块 =================
    # 输入矩阵大小： L/16 * L/16 * 256
    # 输出矩阵大小： L/16 * L/16 * 512
    # 无池化层

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，激活函数为 ReLU，过滤器个数为 512
    x = layers.Conv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # ================= 第六模块 =================
    # 输入矩阵大小： L/16 * L/16 * 512
    # 输出矩阵大小： L/8 * L/8 * 256

    # 上采样层
    # L/16 * L/16 * 512 -> L/8 * L/8 * 512
    x = layers.UpSampling2D(size=(2, 2))(x)
    # 第一卷积层
    # L/8 * L/8 * 512 -> L/8 * L/8 * 256
    x = layers.Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # merge
    x = layers.Concatenate(axis=3)([p4, x])
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第三卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # ================= 第七模块 =================
    # 输入矩阵大小： L/8 * L/8 * 256
    # 输出矩阵大小： L/4 * L/4 * 128

    # 上采样层
    # L/8 * L/8 * 256 -> L/4 * L/4 * 256
    x = layers.UpSampling2D(size=(2, 2))(x)
    # 第一卷积层
    # L/4 * L/4 * 256 -> L/4 * L/4 * 128
    x = layers.Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # merge
    x = layers.Concatenate(axis=3)([p3, x])
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第三卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # ================= 第八模块 =================
    # 输入矩阵大小： L/4 * L/4 * 128
    # 输出矩阵大小： L/2 * L/2 * 64

    # 上采样层
    # L/4 * L/4 * 128 -> L/2 * L/2 * 128
    x = layers.UpSampling2D(size=(2, 2))(x)
    # 第一卷积层
    # L/2 * L/2 * 128 -> L/2 * L/2 * 64
    x = layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # merge
    x = layers.Concatenate(axis=3)([p2, x])
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第三卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # ================= 第九模块 =================
    # 输入矩阵大小： L/2 * L/2 * 64
    # 输出矩阵大小： L * L * 2

    # 上采样层
    # L/2 * L/2 * 64 -> L * L * 64
    x = layers.UpSampling2D(size=(2, 2))(x)
    # 第一卷积层
    # L * L * 64 -> L * L * 32
    x = layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # merge
    x = layers.Concatenate(axis=3)([p1, x])
    # 第二卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第三卷积层
    x = layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # 第四卷积层
    # L * L * 32 -> L * L * 2
    x = layers.Conv2D(kernel_size=3, strides=1, filters=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # ================= 输出层 =================
    # 输入矩阵大小： L * L * 2
    # 输出矩阵大小： L * L * 1
    # 39， 40

    outputs = layers.Conv2D(kernel_size=1, strides=1, filters=1, activation='softmax')(x)
    # layers.Dense(units=10, activation='softmax')
    model = keras.Model(inputs, outputs, name='u-net')
    return model
