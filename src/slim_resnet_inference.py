# Piece:
#   TensorFlow streaming metrics workflow
#
# watermark:
#   CPython 3.6.6
#   IPython 6.4.0
#   tensorflow 1.8.0

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

# config
experiment_folder = './experiment'
checkpoint_file = './resnet_v2_50.ckpt'
sample_images = ['sample.jpg']
input_shape = (299, 299, 3)


def build_resnet(images_tn, num_classes, is_training):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_logits, end_points = resnet_v2.resnet_v2_50(images_tn, is_training=is_training)

    resnet_init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, [var for var in tf.global_variables()])

    with tf.variable_scope('ClassificationPart'):
        resnet_flat = tf.layers.flatten(resnet_logits)
        dense2 = tf.layers.dense(resnet_flat, 1024, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(dense2, 0.75, training=is_training)
        logits = tf.layers.dense(dropout2, num_classes, activation=tf.nn.relu)
    return resnet_init_fn, logits


def simple_inference_resnet():
    num_classes = 3

    # create model
    images_tn = tf.placeholder(tf.float32, shape=(None,) + input_shape, name='input_image')
    init_fn, logits = build_resnet(images_tn, num_classes, False)

    # init variables
    init_vars = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ClassificationPart') +
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='FinalPart'))
    with tf.Session() as sess:
        # init
        sess.run(init_vars)
        init_fn(sess)

        # inference
        for image in sample_images:
            im = Image.open(image).resize(input_shape[:2])
            im = np.array(im)
            im = im.reshape(1, *input_shape)
            print(im.shape)
            logit_values = sess.run(logits, feed_dict={images_tn: im})
            print(np.array(logit_values).shape)
            print(np.max(logit_values))
            print(np.argmax(logit_values))


if '__main__' in __name__:
    simple_inference_resnet()
