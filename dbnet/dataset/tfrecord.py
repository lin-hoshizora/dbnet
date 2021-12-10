"""
Helper functions to genereate tfrecord files based on VIA labels
"""

import cv2
import numpy as np
import tensorflow as tf
from .shrink import MakeShrinkMap
from .border import MakeBorderMap


clipper = MakeBorderMap()
shrinker = MakeShrinkMap()


def div32(x: int) -> int:
  """
  Make sure a number is divisible by 32
  """
  return (x // 32 + 1) * 32 if x % 32 != 0 else x


def pad(img: np.ndarray) -> np.ndarray:
  """
  Pad an image to make sure both its height and width are divisible by 32
  """
  h_new = div32(img.shape[0])
  w_new = div32(img.shape[1])
  padded = np.zeros((h_new, w_new, 3), dtype=img.dtype)
  padded[:img.shape[0], :img.shape[1], :] = img
  return padded, (img.shape[0], img.shape[1])


def parse(record, folder, resize_h=None):
  img = cv2.imread(f'{folder}/{record["filename"]}')[..., ::-1]
  ratio = 0
  if resize_h is None:
    if img.shape[0] > img.shape[1] * 1.25:
      w_new = 1024
      ratio = w_new / img.shape[1]
      h_new = int(ratio * img.shape[0])
    else:
      h_new = 736
      ratio = h_new / img.shape[0]
      w_new = int(ratio * img.shape[1])
  else:
    if img.shape[0] < img.shape[1]:
      h_new = resize_h
      ratio = h_new / img.shape[0]
      w_new = int(ratio * img.shape[1])
    else:
      w_new = resize_h
      ratio = w_new / img.shape[1]
      h_new = int(ratio * img.shape[0])

  img = cv2.resize(img, (w_new, h_new), cv2.INTER_AREA)
  img, (rs_h, rs_w) = pad(img)
  polygons = []
  for region in record['regions']:
    assert region['shape_attributes']['name'] in ["rect", "polygon"]
    if region["shape_attributes"]["name"] == "rect":
      x = region['shape_attributes']['x']
      y = region['shape_attributes']['y']
      w = region['shape_attributes']['width']
      h = region['shape_attributes']['height']
      polygons.append((np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) * ratio).astype(int))
    if region["shape_attributes"]["name"] == "polygon":
      xs = np.array(region['shape_attributes']['all_points_x'])
      ys = np.array(region['shape_attributes']['all_points_y'])
      pts = np.vstack((xs, ys)).transpose([1, 0])
      polygons.append((pts * ratio).astype(int))
  ignores = [False] * len(polygons)
  polygons = np.array(polygons)
  data = {'img': img, 'text_polys': polygons, 'ignore_tags': ignores, 'filename': record['filename'], 'rs_h': rs_h, 'rs_w': rs_w}
  data = clipper(data)
  data = shrinker(data)
  return data


def gen_tfr(ds: list, folder: str, save_path: str, resize_h: int = None) -> None:
  """
  Generate tfrecord file based VIA labels
  """
  with tf.io.TFRecordWriter(save_path) as f:
    for idx, v in enumerate(ds):
      data = parse(v, folder=folder, resize_h=resize_h)
      th_pack = np.concatenate((data["threshold_map"][..., np.newaxis], data["threshold_mask"][..., np.newaxis]), axis=-1)
      print('data: ',data["filename"].encode(),data["shrink_map"].shape[0],
                                  data["shrink_map"].shape[1],data["rs_h"],
            data["rs_w"],data['shrink_map'].flatten().shape,th_pack.flatten().shape)
    
      example = tf.train.Example(features=tf.train.Features(feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data["filename"].encode()])),
        "h": tf.train.Feature(int64_list=tf.train.Int64List(value=[data["shrink_map"].shape[0]])),
        "w": tf.train.Feature(int64_list=tf.train.Int64List(value=[data["shrink_map"].shape[1]])),
        "rs_h": tf.train.Feature(int64_list=tf.train.Int64List(value=[data["rs_h"]])),
        "rs_w": tf.train.Feature(int64_list=tf.train.Int64List(value=[data["rs_w"]])),
        "prob": tf.train.Feature(float_list=tf.train.FloatList(value=data['shrink_map'].flatten())),
        "threshold": tf.train.Feature(float_list=tf.train.FloatList(value=th_pack.flatten())),
      }))
      f.write(example.SerializeToString())
      print(f"\r{idx + 1} / {len(ds)} done", end="")


def split(path: str, n_val: int) -> None:
  """
  Split a tfrecord file to a train set and a validation set
  """
  ds = tf.data.TFRecordDataset(path)
  with tf.io.TFRecordWriter(path.replace(".tfrecord", "_train.tfrecord")) as f_train:
    with tf.io.TFRecordWriter(path.replace(".tfrecord", "_val.tfrecord")) as f_val:
      for idx, data in enumerate(ds):
        if idx < n_val:
          f_val.write(data.numpy())
        else:
          f_train.write(data.numpy())
        print(f"\r{idx + 1} done", end="")
  print()
