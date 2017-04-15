import numpy as np
import tensorflow as tf

def W_var(shape, stddev, layer_reg_scale=1.0, name="W"):
  initial = tf.random_normal(shape, stddev=stddev)
  var = tf.Variable(initial, name=name)
  wloss = layer_reg_scale * tf.nn.l2_loss(var)
  tf.add_to_collection('losses', wloss)
  return var

def b_var(length, name="b"):
  initial = tf.constant(0.0, shape=[length])
  return tf.Variable(initial, name=name)

def batch_norm(X, is_training):
  out = tf.contrib.layers.batch_norm(inputs=X, 
      center=True, scale=True, is_training=is_training)
  return out                                      

def affine(X, W_shape, stddev):
  W = W_var(W_shape, stddev, layer_reg_scale=1.0)
  b = b_var(W_shape[-1])
  X_reshaped = tf.reshape(X, [-1, W_shape[0]])
  out = tf.matmul(X_reshaped, W) + b
  return out

def relu(X):
  out = tf.nn.relu(X)
  return out

def dropout(X, is_training, keep_prob_value):
  keep_prob = tf.cond(is_training,
      lambda: tf.constant(1.0),
      lambda: tf.constant(keep_prob_value))
  out = tf.nn.dropout(X, keep_prob)
  return out

def loss(scores, y, reg):
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - scores))
    sumW2 = tf.add_n(tf.get_collection('losses'))
    loss = loss + reg * sumW2
    tf.summary.scalar("loss", loss) 
    # NOTE! Include at least one summary 
    # or evaluating merged will result in error:
    # Fetch argument None has invalid type <class 'NoneType'>
    return loss

def rmse(scores, y):
  with tf.name_scope("RMSE"):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y - scores)))
    return rmse

def r2(scores, y):
  with tf.name_scope("R2"):
    sse = tf.reduce_sum(tf.squared_difference(y, scores))
    sst = tf.reduce_sum(tf.squared_difference(y, tf.reduce_mean(y)))
    r2 = 1 - tf.div(sse, sst)
    return r2

def train_op(loss, learning_rate, global_step):  
  opt = tf.train.AdamOptimizer(learning_rate)
  grads_and_vars = opt.compute_gradients(loss)
  apply_grads_op = opt.apply_gradients(grads_and_vars, global_step=global_step) # global_step incremented

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  with tf.control_dependencies([apply_grads_op] + update_ops):
    train_op = tf.no_op(name='train') # does nothing, only for control

  return train_op
  