import pickle
from pathlib import Path
import tensorflow as tf
try:
  import wandb
  from wandb.keras import WandbCallback
except ModuleNotFoundError as e:
  print("wandb is not installed, skip importing...")
from .base import BaseTrainer
from . import loss
from ..utils import load_conf, print_conf, merge_conf
from .. import model
from ..dataset import TFRecordLoader


class SimpleTrainer(BaseTrainer):
  def __init__(self, conf_path, **kwargs):
    # load conf
    self.train_conf = load_conf(conf_path)
    self.conf_folder = Path(conf_path).parent
    self.model_conf = load_conf(str(self.conf_folder / f"{self.train_conf['model']}.yaml"))
    self.backbone_conf = load_conf(str(self.conf_folder / f"{self.model_conf['options']['backbone']}.yaml"))
    # override config
    self._override(self.model_conf, **kwargs)


    # compile model　
    self.model = getattr(model, self.model_conf["builder"])(dynamic_shape=True, **self.model_conf["options"])
    losses = {k: getattr(loss, v) for k, v in self.train_conf["loss"].items()}
    print('30 ',losses)
    self.model.compile(
      optimizer=self.get_optimizer(),
      loss=losses,
      loss_weights=self.train_conf["weights"]
    )
    
    self.model.summary()

    # load datasets
    self.loader = TFRecordLoader()
    self.ds_train = self.loader.get_ds_from(str(self.conf_folder / f"{self.train_conf['datasets']['train']}.yaml"))
    self.ds_val = self.loader.get_ds_from(str(self.conf_folder / f"{self.train_conf['datasets']['val']}.yaml"))


  def train(self, wandb_runname="", **kwargs):
    # override config
    self._override(self.train_conf, **kwargs)

    # handle wandb
    if "WandbCallback" in self.train_conf["callbacks"]:
      all_conf = merge_conf([self.backbone_conf, self.model_conf, self.train_conf])
      wandb.init(project=self.train_conf["callbacks"]["WandbCallback"].pop("project"), config=all_conf)
      if wandb_runname:
        wandb.run.name = wandb_runname
        wandb.run.save()

    # start fitting
    self.history = self.model.fit(
      self.ds_train,
      validation_data=self.ds_val,
      callbacks=self.get_callbacks(),
      **self.train_conf["fit"])


  def __repr__(self):
    res = ""
    res += print_conf(self.backbone_conf, space=0, tag="Backbone Config")
    res += print_conf(self.model_conf, space=0, tag="Model Config")
    res += print_conf(self.train_conf, space=0, tag="Train Config")
    return res
