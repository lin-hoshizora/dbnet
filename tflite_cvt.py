import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import yaml
import tensorflow as tf
from dbnet.model import dbnet_fpn, infer_strip


default_conf_path = "./conf/db192mini.yaml"
default_weights_path = "./ckpt/mynum-mobilev3-1.0-192-min-imagenet/11-0.581.ckpt"
default_save_path = "./test.tflite"
conf_path = input(f"Input config file path (default: {default_conf_path})")
if not conf_path: conf_path = default_conf_path
weights_path = input(f"Input weight file path (default: {default_weights_path})")
if not weights_path: weights_path = default_weights_path
save_path = input(f"Input save path for tflite model file (default: {default_save_path})")
if not save_path: save_path = default_save_path

with open(conf_path) as f:
  conf = yaml.safe_load(f)

model = dbnet_fpn(**conf, batch_size=1)
model.load_weights(weights_path)
infer_model = infer_strip(model)
infer_model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
with open(save_path, "wb") as f:
  f.write(converter.convert())
