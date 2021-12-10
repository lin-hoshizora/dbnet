import sys
import time
import tensorflow as tf
import numpy as np
from openvino.inference_engine import IENetwork, IECore

assert len(sys.argv) == 4
tf_path = sys.argv[1]
openvino_path = sys.argv[2]
dev = sys.argv[3]

n_iter = 10

# OpenVINO inference
ie_core = IECore()
net = ie_core.read_network(model=openvino_path+".xml", weights=openvino_path+".bin")
exe = ie_core.load_network(net, dev, num_requests=1)
input_name = next(iter(exe.input_info))
input_shape = exe.input_info[input_name].input_data.shape
input_data = (np.random.random(input_shape) * 255.).astype(np.float32)
output_name = next(iter(exe.outputs))
openvino_times = []
for _ in range(n_iter):
  t0 = time.time()
  req = exe.start_async(exe.get_idle_request_id(), {input_name: input_data})
  req.wait()
  openvino_out = req.output_blobs[output_name].buffer
  t = time.time() - t0
  openvino_times.append(t)
  print(f"OpenVINO inference: {t:.3f}s")

# TF inference
tf_times = []
model = tf.keras.models.load_model(tf_path)
for _ in range(n_iter):
  t0 = time.time()
  tf_out = model.predict(input_data)
  t = time.time() - t0
  tf_times.append(t)
  print(f"TF inference: {t:.3f}s")

print(f"OpenVINO({dev}) avg time: {np.mean(openvino_times):.3f}s")
print(f"TF avg time: {np.mean(tf_times):.3f}s")

err = np.abs((openvino_out - tf_out) / tf_out)
err_mean = err.mean()
err_max = err.max()
threshold = 0.01 if dev == "CPU" else 0.05
print(f"Mean err: {err_mean * 100:.3f}%  Max err: {err_max * 100:.3f}%")
#assert err_max < threshold
