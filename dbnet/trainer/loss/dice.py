import tensorflow as tf


# for binary map
def dice(y_true, y_pred, eps=1e-6):
  inter = tf.reduce_sum(y_pred * y_true)
  union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + eps
  loss = 1 - 2.0 * inter / union
  return loss
