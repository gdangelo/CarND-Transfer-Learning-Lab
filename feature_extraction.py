import pickle
import numpy as np
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('epoch', 50, "Number of epochs used for training")
flags.DEFINE_string('batch_size', 256, "Batch size used for training")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    
    '''inputs = Input(shape=X_train.shape[1:])
    x = Flatten()(inputs)
    pred = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs, pred)'''

    model = Sequential()

    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    print(model.summary())
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epoch, validation_data=(X_val, y_val))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
