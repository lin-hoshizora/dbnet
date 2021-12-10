import tensorflow as tf


# for threshold map
def l1(y_true, y_pred, eps=1e-6):
  mask = y_true[:, :, :, 1:]
  y_true = y_true[:, :, :, :1]
  loss = tf.reduce_sum(tf.abs(y_pred - y_true) * mask) / tf.reduce_sum(mask)
  return loss
