# Piece:
#   tf.keras with tf.estimator.Estimator
#   ResNet50 pretrained
#   tf.data.Datasets API for i/o
#
# watermark:
#   CPython 3.6.6
#   IPython 6.4.0
#   tensorflow 1.8.0

import json
from os import path
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications.resnet50 import ResNet50

# config
experiment_folder = './experiment'
train_path = './input/train'
test_path = './input/test'
input_shape = [299, 299, 3]

# routines


def train_ke(train_files, train_labels, test_files, test_labels):
    """
    Trains keras model.
    Args:
        train_files (list): filenames for train process.
        train_labels (list): labels corresponding to train_files.
        test_files (list): filenames for train process.
        test_labels (list): labels corresponding to test_files.

    Returns:
        Nothing.
    """
    model = create_model()

    est_catvsdog = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                         model_dir=experiment_folder)

    train_input_fn = lambda: imgs_input_fn(train_files, train_labels,
                                           perform_shuffle=True, repeat_count=2, batch_size=32)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2)

    eval_input_fn = lambda: imgs_input_fn(test_files, labels=test_labels, perform_shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(est_catvsdog, train_spec, eval_spec)


def create_model():
    model = ResNet50(include_top=False, weights='imagenet', pooling='max')
    inp = Input(input_shape, name='input_image')
    out = model(inp)
    out = Flatten()(out)
    out = Dense(1024)(out)
    out = Dropout(0.75)(out)
    out = Dense(1)(out)
    model = Model(inputs=[inp], outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    return model


def imgs_input_fn(filenames, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):
    """
    Creates tf.data.Dataset object.
    Args:
        filenames (list): images paths. 
        labels (list):  gt labels.
        perform_shuffle (bool): specifies whether it is necessary to shuffle.
        repeat_count (int): specifies number of dataset repeats.  
        batch_size (int): specifies batch size.

    Returns (tuple):
        Tuple contains images batch and corresponding labels batch.  

    """

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image.set_shape([None, None, 3])  # important!
        image = tf.image.resize_images(image, input_shape[:2])
        image = tf.div(tf.subtract(image, 127.5), 127.5)  # normalization
        return image, label, filename

    if labels is None:
        labels = [0]*len(filenames)  # fakes
    labels=np.array(labels)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).repeat(repeat_count)
    return dataset.make_one_shot_iterator().get_next()


def main():
    def get_inputs(dataset_folder):
        data_descr = json.load(open(dataset_folder, 'r'))
        filenames = [path.join(dataset_folder, record['image']) for record in data_descr]
        labels = [record['label'] for record in data_descr]
        return filenames, labels
    train_x, train_y = get_inputs(train_path)
    test_x, test_y = get_inputs(test_path)
    train_ke(train_x, train_y, test_x, test_y)

if '__main__' in __name__:
    main()
