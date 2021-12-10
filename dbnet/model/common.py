import tensorflow as tf
from tensorflow import keras
from . import backbone


def conv(x, filters, kernel_size, kernel_init, mean_init="zeros"):
    if kernel_size == 1:
        x = keras.layers.Conv2D(filters, kernel_size,
                                padding='same',
                                use_bias=True,
                                kernel_initializer=kernel_init)(x)
    else:
        x = keras.layers.SeparableConv2D(filters, kernel_size,
                                         padding='same',
                                         use_bias=True,
                                         depthwise_initializer=kernel_init,
                                         pointwise_initializer=kernel_init)(x)
    x = keras.layers.BatchNormalization(moving_mean_initializer=mean_init)(x)
    x = keras.layers.ReLU()(x)
    return x


def upsample_add(x, y):
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Add()([x, y])
    return x


def upsample(x, rep):
    for _ in range(rep):
        x = keras.layers.UpSampling2D()(x)
    return x


def get_map(x, name, fpn_channels, kernel_init, mean_init="zeros"):
    x = conv(x, fpn_channels // 4, 3, kernel_init=kernel_init, mean_init=mean_init)
    x = keras.layers.Conv2DTranspose(
        fpn_channels // 4,
        kernel_size=2,
        strides=2,
        use_bias=False,
        kernel_initializer=kernel_init)(x)
    x = keras.layers.BatchNormalization(moving_mean_initializer=mean_init)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2DTranspose(1, kernel_size=2, strides=2, use_bias=False)(x)
    x = keras.layers.Activation('sigmoid', name=name, dtype='float32')(x)
    return x


def get_backbone(conf, input_tensor):
    fn = getattr(backbone, conf["name"])
    base = fn(input_tensor=input_tensor, **conf["options"])
    return base
