import tensorflow as tf
from math import ceil

from Utils import _variable_with_weight_decay, _variable_on_cpu, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage

from Inputs import *
from config import *


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def loss(logits, labels):
  """
      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, (-1,NUM_CLASSES))
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def cal_loss(logits, labels):
    if (NUM_CLASSES ==32):
        loss_weight = np.array([
        0.2595, 4.5640, 0.1417, 0.6823,
        0.1826, 4.5640, 0.1417, 0.6823,
        4.5640, 4.5640, 0.1417, 0.6823,
        0.1417, 4.5640, 0.1417, 0.6823,
        0.9051, 4.5640, 0.1417, 0.6823,
        0.3826, 4.5640, 0.1417, 0.6823,
        9.6446, 4.5640, 0.1417, 0.6823,
        1.8418, 4.5640, 0.1417, 0.6823,
        ]) # class 0~32
    if (NUM_CLASSES == 11):
        loss_weight = np.array([
        0.2595, 0.1826, 4.5640, 0.1417,
        0.9051, 0.3826, 9.6446, 1.8418,
        0.6823, 6.2478, 7.3614,
        ]) # class 0~10


    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

#=====================================================

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    #in_channel = shape[2]
    out_channel = shape[3]
    #k_size = shape[0]
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


## ============================================================================

def prediction_myModel(images, batch_size=1, phase_train=False):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
    # conv1
    conv1_1 = conv_layer_with_bn(norm1, [3, 3, images.get_shape().as_list()[3], 64], phase_train, name="conv1_1")
    conv1_2 = conv_layer_with_bn(conv1_1, [3, 3, 64, 64], phase_train, name="conv1_2")
    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    # conv2
    conv2_1 = conv_layer_with_bn(pool1, [3, 3, 64, 128], phase_train, name="conv2_1")
    conv2_2 = conv_layer_with_bn(conv2_1, [3, 3, 128, 128], phase_train, name="conv2_2")
    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3_1 = conv_layer_with_bn(pool2, [3, 3, 128, 256], phase_train, name="conv3_1")
    conv3_2 = conv_layer_with_bn(conv3_1, [3, 3, 256, 256], phase_train, name="conv3_2")
    conv3_3 = conv_layer_with_bn(conv3_2, [3, 3, 256, 256], phase_train, name="conv3_3")

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4_1 = conv_layer_with_bn(pool3, [3, 3, 256, 512], phase_train, name="conv4_1")
    conv4_2 = conv_layer_with_bn(conv4_1, [3, 3, 512, 512], phase_train, name="conv4_2")
    conv4_3 = conv_layer_with_bn(conv4_2, [3, 3, 512, 512], phase_train, name="conv4_3")
    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    # conv5
    conv5_1 = conv_layer_with_bn(pool4, [3, 3, 512, 512], phase_train, name="conv5_1")
    conv5_2 = conv_layer_with_bn(conv5_1, [3, 3, 512, 512], phase_train, name="conv5_2")
    conv5_3 = conv_layer_with_bn(conv5_2, [3, 3, 512, 512], phase_train, name="conv5_3")
    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    """ End of encoder """
    """ start upsample """
    # upsample5
    upsample5 = deconv_layer(pool5, [2, 2, 512, 512], [batch_size, 23, 30, 512], 2, "up5")
    # decode 4
    conv_decode5_3 = conv_layer_with_bn(upsample5, [3, 3, 512, 512], phase_train, False, name="conv_decode5_3")
    conv_decode5_2 = conv_layer_with_bn(conv_decode5_3, [3, 3, 512, 512], phase_train, False, name="conv_decode5_2")
    conv_decode5_1 = conv_layer_with_bn(conv_decode5_2, [3, 3, 512, 512], phase_train, False, name="conv_decode5_1")
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(conv_decode5_1, [2, 2, 512, 512], [batch_size, 45, 60, 512], 2, "up4")
    # decode 4
    conv_decode4_3 = conv_layer_with_bn(upsample4, [3, 3, 512, 512], phase_train, False, name="conv_decode4_3")
    conv_decode4_2 = conv_layer_with_bn(conv_decode4_3, [3, 3, 512, 512], phase_train, False, name="conv_decode4_2")
    conv_decode4_1 = conv_layer_with_bn(conv_decode4_2, [3, 3, 512, 256], phase_train, False, name="conv_decode4_1")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4_1, [2, 2, 256, 256], [batch_size, 90, 120, 256], 2, "up3")
    # decode 3
    conv_decode3_3 = conv_layer_with_bn(upsample3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_3")
    conv_decode3_2 = conv_layer_with_bn(conv_decode3_3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_2")
    conv_decode3_1 = conv_layer_with_bn(conv_decode3_2, [3, 3, 256, 128], phase_train, False, name="conv_decode3_1")
    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3_1, [2, 2, 128, 128], [batch_size, 180, 240, 128], 2, "up2")
    # decode 2
    conv_decode2_2 = conv_layer_with_bn(upsample2, [3, 3, 128, 128], phase_train, False, name="conv_decode2_2")
    conv_decode2_1 = conv_layer_with_bn(conv_decode2_2, [3, 3, 128, 64], phase_train, False, name="conv_decode2_1")
    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2_1, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    conv_decode1_2 = conv_layer_with_bn(upsample1, [3, 3, 64, 64], phase_train, False, name="conv_decode1_2")
    conv_decode1 = conv_layer_with_bn(conv_decode1_2, [3, 3, 64, 64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, NUM_CLASSES],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier

    return logit

## ============================================================================

def prediction_segnet(images, batch_size=1, phase_train=False):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # pool4
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, NUM_CLASSES],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier

    return logit

# ===============================

def inference(images, labels, batch_size, phase_train):

    if (cnn_model == 'SegNet'):
        logit = prediction_segnet(images, batch_size, phase_train)
    else:
        logit = prediction_myModel(images, batch_size, phase_train)
    loss = cal_loss(logit, labels)

    return loss, logit
