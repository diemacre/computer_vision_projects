"""
    @Title:Assigment 4 of CS-512 - Computer Vision
    @author: Diego Martin Crespo
    @Term: Fall 2018
    @Id: A20432558
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.callbacks import TensorBoard
import functools
import tensorflow as tf

############################# configuration to launch tensorboard in localhost #############################
!wget https: // bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')


! curl - s http: // localhost: 4040/api/tunnels | python3 - c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

############################# Beginning of the model configuration ##################################
batch_size = 64
#odd and even numbers/clases
num_classes = 2
epochs = 5

#function for classify in odd and even

def relabel(labels):
    for idx, item in enumerate(labels):
        if item % 2 == 0:
            labels[idx] = 0     # even number
        else:
            labels[idx] = 1     # odd number
    return labels


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#change to even and odd
y_train = relabel(y_train)
y_test = relabel(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# layer 1 convolution, 32 filters
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu', input_shape=input_shape))
# pooling with downsample by 2 layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 2 convolution, 64 filters
model.add(Conv2D(64, (5, 5), activation='relu'))
# pooling with downsample by 2, layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout of 40%
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
# gradient optimization using learning rate of 0.001
sdg = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
adam = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#implementation of tensorflow precision and recall
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


precision2 = as_keras_metric(tf.metrics.precision)
recall2 = as_keras_metric(tf.metrics.recall)

#implementation of keras precision and recall
def precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
  pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives/(pred_positives + K.epsilon())
  return precision


def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
  pred_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives/(pred_positives + K.epsilon())
  return recall

#compilation of model to get accuracy, loss,  precision and recall
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sdg,
              metrics=['accuracy', precision, recall])

#tensorboard callback
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)

#fit the model and call tensorboard
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tbCallBack])

############################# save model to .h5 file #############################
model.save("../models/model.h5")
print("Saved model to disk")
