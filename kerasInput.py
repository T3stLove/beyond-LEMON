import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import pandas as pd
from tensorflow.python.keras.backend import shape
from colors import *

_, (x_test, y_test) = keras.datasets.mnist.load_data()
def get_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test
# print(y_test)
# print('=======')
x_test = get_mnist_data(x_test)
y_test = keras.utils.to_categorical(y_test, num_classes=10)