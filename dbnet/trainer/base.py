import tensorflow as tf
import tensorflow_addons as tfa
from . import optimizer
from . import callback
from ..utils import get_timestamp


class BaseTrainer:
  def get_optimizer(self):
    opt = getattr(optimizer, self.train_conf["optimizer"]["name"])(**self.train_conf["optimizer"]["options"])
    if self.train_conf["optimizer"]["lookahead"]:
      opt = tfa.optimizers.Lookahead(opt)
    if self.train_conf["optimizer"]["mixed_precision"]:
      opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    return opt


  def get_callbacks(self):
    cbs = []
    for k, v in self.train_conf["callbacks"].items():
      if "filepath" in v:
        v["filepath"] = f"./ckpt/{v['filepath']}/{get_timestamp()}/" + "/{epoch:02d}-{loss:.3f}.ckpt"
      cbs.append(getattr(callback, k)(**v))
    return cbs


  def _override(self, conf, **kwargs):
    for k, v in conf.items():
      if k in kwargs:
        self.train_conf[k] = kwargs[k]
      if isinstance(v, dict):
        for subk in v:
          if subk in kwargs:
            conf[k][subk] = kwargs[subk]


  def save_history(self, path):
    with open(path, "wb") as f:
      pickle.dump(self.history.history, f)
