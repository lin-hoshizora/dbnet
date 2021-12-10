from tensorflow import keras


def infer_strip(model):
  model_infer = keras.Model(inputs=model.input, outputs=model.outputs[0])
  return model_infer
