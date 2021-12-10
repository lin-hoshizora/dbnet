import tensorflow as tf
from tensorflow import keras
from .common import conv, upsample_add, upsample, get_map, get_backbone
from ..utils import load_conf


def dbnet_fpn(input_shape,
              backbone,
              fpn_channels,
              k,
              prob_init,
              threshold_init,
              kernel_init, 
              batch_size=None,
              dynamic_shape=False):

  if dynamic_shape: input_shape = (None, None, 3)
  input_tensor = keras.Input(shape=input_shape, name="image", batch_size=batch_size)
  backbone_conf = load_conf(f"./conf/{backbone}.yaml")
  base = get_backbone(backbone_conf, input_tensor)

  feature_names = backbone_conf["feature_names"]
  feature_d32 = base.get_layer(feature_names["downsample32"]).output
  feature_d16 = base.get_layer(feature_names["downsample16"]).output
  feature_d8 = base.get_layer(feature_names["downsample8"]).output
  feature_d4 = base.get_layer(feature_names["downsample4"]).output

  feature_d32 = conv(feature_d32, fpn_channels // 4, 1, kernel_init)
  feature_d16 = conv(feature_d16, fpn_channels // 4, 1, kernel_init)
  merge_d16 = upsample_add(feature_d32, feature_d16)
  merge_d16 = conv(merge_d16, fpn_channels // 4, 3, kernel_init)

  feature_d8 = conv(feature_d8, fpn_channels // 4, 1, kernel_init)
  merge_d8 = upsample_add(merge_d16, feature_d8)
  merge_d8 = conv(merge_d8, fpn_channels // 4, 3, kernel_init)

  feature_d4 = conv(feature_d4, fpn_channels // 4, 1, kernel_init)
  merge_d4 = upsample_add(merge_d8, feature_d4)
  merge_d4 = conv(merge_d4, fpn_channels // 4, 3, kernel_init)

  # upsample to stide 4
  c5 = upsample(feature_d32, 3)
  c4 = upsample(merge_d16, 2)
  c3 = upsample(merge_d8, 1)
  fpn_feature = keras.layers.Concatenate()([merge_d4, c3, c4, c5])
  fpn_feature = conv(fpn_feature, fpn_channels, 3, kernel_init)

  prob_maps = get_map(fpn_feature,
                      "prob",
                      fpn_channels,
                      kernel_init=kernel_init,
                      mean_init=keras.initializers.Constant(value=prob_init))
  threshold_maps = get_map(fpn_feature,
                           "threshold",
                           fpn_channels,
                           kernel_init=kernel_init,
                           mean_init=keras.initializers.Constant(value=threshold_init))
  binary_maps = keras.layers.Lambda(
    lambda x: tf.math.reciprocal(1 + tf.math.exp(-k * (x[0] - x[1]))), name="binary"
  )([prob_maps, threshold_maps])

  model = keras.Model(inputs=base.inputs, outputs=[prob_maps, threshold_maps, binary_maps])
  return model
