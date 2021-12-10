from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt


def check(x, y, figsize):
  fig, axes = plt.subplots(1, 2, figsize=figsize)
  axes[0].imshow(array_to_img(x["image"][0]))
  axes[0].imshow(y["prob"][0, ..., 0], alpha=0.5)
  axes[0].set_title("Prob")
  axes[1].imshow(array_to_img(x["image"][0]))
  axes[1].imshow(y["threshold"][0, ..., 0], alpha=0.5)
  axes[1].set_title("Threshold")
  plt.show()
