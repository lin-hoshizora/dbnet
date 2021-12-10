import tensorflow as tf


if tf.__version__ != "2.4.0":
  print(f"WARNING: DBNet model has only been tested in TF 2.4.0. Current version:{tf.__version__}")
