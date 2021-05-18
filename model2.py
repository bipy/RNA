from tensorflow import keras
from tensorflow.keras import layers, optimizers


def get_model(shape):
    stack = []
    inputs = keras.Input(shape=shape)
    x = inputs
    pre = inputs

    for filters in [32, 64, 128, 256]:
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        stack.append(x)

        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

        residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same')(pre)
        x = layers.add([x, residual])
        pre = x

    x = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    for filters in [256, 128, 64, 32]:
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Concatenate(axis=3)([x, stack.pop()])

        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        residual = layers.UpSampling2D(size=(2, 2))(pre)
        residual = layers.Conv2D(filters=filters, kernel_size=1, padding='same')(residual)
        x = layers.add([x, residual])
        pre = x

    outputs = layers.Conv2D(1, kernel_size=3, activation='softmax', padding='same')(x)

    model = keras.Model(inputs, outputs)
    return model

