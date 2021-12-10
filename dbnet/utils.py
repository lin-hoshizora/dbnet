"""
Helper functions
"""

from datetime import datetime
import yaml


def load_conf(p: str) -> dict:
  with open(p) as f:
    res = yaml.safe_load(f)
  return res


def print_conf(conf: dict, space: int, tag: str = None, res: str = "") -> None:
  if tag: res += "="*50 + "\n" + tag + "\n" + "="*50 + "\n"
  for k, v in conf.items():
    if isinstance(v, dict):
      res += " " * space + str(k) + ":" + "\n"
      res = print_conf(v, space=space + 2, tag=None, res=res)
      continue
    res += " " * space + f"{k}: {v}\n"
  return res

def flatten_dict(d, prefix="", res={}):
  for k, v in d.items():
    if isinstance(v, dict):
      flatten_dict(v, prefix=str(k) + "_", res=res)
      continue
    if isinstance(v, list):
      for idx, lv in enumerate(v):
        res[prefix + k + "_" + str(idx)] = lv
      continue
    res[prefix + k] = v
  return res

def merge_conf(confs, prefix=""):
  return {k: v for conf in confs for k, v in flatten_dict(conf).items()}


def get_timestamp() -> str:
  timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  return timestamp
