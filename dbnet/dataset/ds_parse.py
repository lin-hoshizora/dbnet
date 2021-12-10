"""
Helper functions to parse tfrecord datasets
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from ..utils import load_conf


feature_description = {
  "image": tf.io.FixedLenFeature([], tf.string),
  "h": tf.io.FixedLenFeature([], tf.int64),
  "w": tf.io.FixedLenFeature([], tf.int64),
  "rs_h": tf.io.FixedLenFeature([], tf.int64),
  "rs_w": tf.io.FixedLenFeature([], tf.int64),
  "prob": tf.io.VarLenFeature(tf.float32),
  "threshold": tf.io.VarLenFeature(tf.float32),
}

@tf.function
def parse_ex(raw, folder):
  """
  Parse a single raw tfrecord example
  Args:
    raw: raw example
    folder: use images from this folder
  """
  parsed = tf.io.parse_single_example(raw, feature_description)
  h = tf.cast(parsed["h"], tf.int32)
  w = tf.cast(parsed["w"], tf.int32)
  rs_h = tf.cast(parsed["rs_h"], tf.int32)
  rs_w = tf.cast(parsed["rs_w"], tf.int32)
  img = tf.io.read_file(folder + parsed["image"])
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (rs_h, rs_w), method="area", antialias=True)
  img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
  prob = tf.reshape(tf.sparse.to_dense(parsed["prob"]), (h, w, 1))
  threshold = tf.reshape(tf.sparse.to_dense(parsed["threshold"]), (h, w, 2))
  return {"image": img}, {"prob": prob, "threshold": threshold}


@tf.function
def augment(x, y, max_degree=5):
  """
  Augment images in training dataset
  """
  img = tf.cast(x["image"], tf.uint8)
  prob_map = y["prob"]
  th_map = y["threshold"][:, :, :1]
  th_mask = y["threshold"][:, :, 1:]

  img = tf.image.random_jpeg_quality(img, min_jpeg_quality=50, max_jpeg_quality=100)
  img = tf.clip_by_value(tf.cast(img, tf.float32), clip_value_min=0., clip_value_max=255.)
  # rand brightness is turned off b/c training data already has a variety of lighting
#   img = tf.image.random_brightness(img, max_delta=0.3)
#   img = tf.clip_by_value(img, 0. , 255.)
  img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
  img = tf.clip_by_value(img, 0. , 255.)
  img = tf.image.random_hue(img, max_delta=0.25)
  img = tf.clip_by_value(img, 0. , 255.)
  img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
  img = tf.clip_by_value(img, 0. , 255.)

  # random crop
  rs_ratio = tf.random.uniform([], minval=0.9, maxval=1.5)
  s = tf.cast(tf.shape(img), tf.float32)
  new_h = tf.cast(s[0] / rs_ratio // 32. * 32., tf.int32)
  new_w = tf.cast(s[1] / rs_ratio // 32. * 32., tf.int32)
  img = tf.image.resize(img, (new_h, new_w), method="area")
  prob_map = tf.image.resize(prob_map, (new_h, new_w), method="area", antialias=True)
  th_map = tf.image.resize(th_map, (new_h, new_w), method="area", antialias=True)
  th_mask = tf.image.resize(th_mask, (new_h, new_w), method="area", antialias=True)

  # random flip
  if tf.random.uniform([], minval=0., maxval=1.) > 0.5:
    img = tf.image.flip_left_right(img)
    prob_map = tf.image.flip_left_right(prob_map)
    th_map = tf.image.flip_left_right(th_map)
    th_mask = tf.image.flip_left_right(th_mask)
  if tf.random.uniform([], minval=0., maxval=1.) > 0.5:
    img = tf.image.flip_up_down(img)
    prob_map = tf.image.flip_up_down(prob_map)
    th_map = tf.image.flip_up_down(th_map)
    th_mask = tf.image.flip_up_down(th_mask)

  # random rotation
  max_rad = tf.constant(max_degree / 180 * np.pi)
  angle = tf.random.uniform([], minval=-max_rad, maxval=max_rad)
  if tf.abs(angle) > max_rad / 50:
    img = tfa.image.rotate(img, angle, interpolation="BILINEAR")

  prob_map = tf.cast(tf.greater(tfa.image.rotate(prob_map, angle, interpolation="BILINEAR"), 0), tf.float32)
  th_map = tfa.image.rotate(th_map, angle, interpolation="BILINEAR")
  th_mask = tf.cast(tf.greater(tfa.image.rotate(th_mask, angle, interpolation="BILINEAR"), 0), tf.float32)
  th_pack = tf.concat([th_map, th_mask], axis=-1)
  return {"image": img}, {"prob": prob_map, "threshold": th_pack, "binary": prob_map}


@tf.function
def val_resize(x, y):
  img = x["image"]
  s = tf.cast(tf.shape(img), tf.float32)
  h_new = 0
  w_new = 0
  if s[0] == 1504:
    h_new = 1280
    w_new = tf.cast(h_new / s[0] * s[1] // 32. * 32., tf.int32)
  elif s[1] == 1504:
    w_new = 1280
    h_new = tf.cast(w_new / s[1] * s[0] // 32. * 32., tf.int32)
  else:
    if s[0] < s[1]:
      h_new = 640
      w_new = tf.cast(h_new / s[0] * s[1] // 32. * 32., tf.int32)
    else:
      w_new = 896
      h_new = tf.cast(w_new / s[1] * s[0] // 32. * 32., tf.int32)
  img = tf.image.resize(img, (h_new, w_new), method="area", antialias=True)
  img = tf.clip_by_value(img, clip_value_min=0., clip_value_max=255.)
  prob_map = tf.image.resize(y["prob"], (h_new, w_new), method="area", antialias=True)
  prob_map = tf.cast(tf.greater(prob_map, 0), tf.float32)
  th_map = tf.image.resize(y["threshold"][..., :1], (h_new, w_new), method="area", antialias=True)
  th_mask = tf.image.resize(y["threshold"][..., 1:], (h_new, w_new), method="area", antialias=True)
  th_mask = tf.cast(tf.greater(th_mask, 0), tf.float32)
  th_pack = tf.concat([th_map, th_mask], axis=-1)
  return {"image": img}, {"prob": prob_map, "threshold": th_pack, "binary": prob_map}


@tf.function
def parse(raw, folder, tags, train):
  # randomly choose image folder during training
  folders = tf.constant([folder + tag + "/" for tag in tags])
  if train and len(folders) > 1:
    folder_idx = tf.random.uniform([], minval=0, maxval=len(folders) - 1, dtype=tf.int32)
  else:
    folder_idx = 0
  folder = folders[folder_idx]

  x, y = parse_ex(raw, folder)

  if train:
    x, y = augment(x, y)
  else:
    x, y = val_resize(x, y)
  return x, y
