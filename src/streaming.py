# Piece:
#   TensorFlow streaming metrics workflow
#
# watermark:
#   CPython 3.6.6
#   IPython 6.4.0
#   tensorflow 1.8.0

import tensorflow as tf

# config
experiment_folder = './experiment'


def train():
    logits = tf.placeholder(tf.int64, [2, 3])
    labels = tf.Variable([[0, 1, 0], [1, 0, 1]])

    # create two different ops
    with tf.name_scope('train'):
        train_acc, train_acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                                      predictions=tf.argmax(logits, 1))
    with tf.name_scope('valid'):
        valid_acc, valid_acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                                      predictions=tf.argmax(logits, 1))

    tf.summary.scalar('tr_acc', train_acc)
    tf.summary.scalar('val_acc', valid_acc)
    summ_op = tf.summary.merge_all()
    stream_vars_valid = [v for v in tf.local_variables() if 'valid/' in v.name]
    stream_vars_train = [v for v in tf.local_variables() if 'train/' in v.name]
    with tf.Session() as sess:

        # initialize the local variables has it holds the variables used for metrics calculation.
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(experiment_folder, graph=tf.get_default_graph())

        # initial state
        print('initial state')
        print(sess.run(train_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
        print(sess.run(valid_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))

        for i in range(3):
            print('training')
            sess.run(tf.variables_initializer(stream_vars_train))
            for _ in range(10):
                sess.run(train_acc_op, {logits: [[0, 1, 0], [1, 0, 1]]})
            summ_str = sess.run(summ_op)
            writer.add_summary(summ_str, i)
            print(sess.run(train_acc))
            print(sess.run(valid_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
            writer.flush()

            print('validation')
            sess.run(tf.variables_initializer(stream_vars_valid))
            for _ in range(10):
                sess.run(valid_acc_op, {logits: [[0, 1, 0], [0, 1, 0]]})
            summ_str = sess.run(summ_op)
            writer.add_summary(summ_str, i)
            print(sess.run(valid_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
            print(sess.run(train_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
            writer.flush()
            writer.close()

if '__main__' in __name__:
    train()
