from pathlib import Path
import tensorflow as tf
from .ds_parse import parse
from ..utils import load_conf


class TFRecordLoader:
  def get_ds_from(self, conf_path):
    conf = load_conf(conf_path)
    if not Path(conf["tfrecord"]).exists():
      raise ValueError(f"{conf['tfrecord']} does not exist")
    if not Path(conf["parse_params"]["folder"]).exists():
      raise ValueError(f"{conf['parse_params']['folder']} does not exist")
    ds = tf.data.TFRecordDataset(conf["tfrecord"])
    if conf["train"]:
      ds = ds.shuffle(conf["buf_size"])

    @tf.function
    def cur_parse(raw):
      return parse(raw, train=conf["train"], **conf["parse_params"])

    ds = ds.map(cur_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(conf["batch_size"])
    if conf["train"]: ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

  def merge_datasets(self, datasets, choice, repeat):
    choice = tf.data.Dataset.from_tensor_slices(choice)
    ds = tf.data.experimental.choose_from_datasets(datasets, choice)
    if repeat: ds = ds.repeat()
    return ds
