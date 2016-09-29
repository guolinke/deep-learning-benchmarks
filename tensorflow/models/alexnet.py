import tensorflow.python.platform
import tensorflow as tf


data_format = 'NCHW'

image_size = 224

if data_format == 'NCHW':
	featureDim = (3, image_size + 3, image_size + 3)
else:
	featureDim = (image_size + 3, image_size + 3, 3)

numClasses = 1000


conv_counter = 1
pool_counter = 1
affine_counter = 1
parameters = []


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        if data_format == 'NCHW':
          strides = [1, 1, dH, dW]
        else:
          strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                            data_format=data_format)
        biases = tf.Variable(tf.constant(0.1, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                         data_format=data_format),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1

def _affine(inpOp, nIn, nOut, is_relu = True):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        if is_relu:
        	affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        else:
        	affine1 = tf.nn.xw_plus_b(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          data_format=data_format,
                          name=name)

def inference(images):
    conv1 = _conv (images, 3, 96, 11, 11, 4, 4, 'VALID')
    pool1 = _mpool(conv1,  3, 3, 2, 2)
    conv2 = _conv (pool1,  96, 256, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool(conv2,  3, 3, 2, 2)
    conv3 = _conv (pool2,  256, 384, 3, 3, 1, 1, 'SAME')
    conv4 = _conv (conv3,  384, 384, 3, 3, 1, 1, 'SAME')
    conv5 = _conv (conv4,  384, 256, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool(conv5,  3, 3, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    affn1 = _affine(resh1, 256 * 6 * 6, 4096)
    affn2 = _affine(affn1, 4096, 4096)
    affn3 = _affine(affn2, 4096, numClasses, is_relu=False)

    return affn3

def build_model(feature):
	return inference(feature)
