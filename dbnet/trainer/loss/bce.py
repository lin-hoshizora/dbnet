import tensorflow as tf


# for shrink map
def bce(y_true, y_pred, neg_ratio=3.0, eps=1e-6):
  neg = 1 - y_true
  pos_count = tf.reduce_sum(y_true)
  neg_count = tf.minimum(tf.reduce_sum(neg), pos_count * neg_ratio)
  loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
  pos_loss = loss * tf.squeeze(y_true, axis=-1)
  neg_loss = loss * tf.squeeze(neg, axis=-1)
  neg_loss, _ = tf.math.top_k(tf.reshape(neg_loss, (-1,)), k=tf.cast(neg_count, tf.int32))
  loss = (tf.reduce_sum(neg_loss) + tf.reduce_sum(pos_loss)) / (pos_count + neg_count + eps)
  return loss
