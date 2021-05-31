import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
from colors import *

modelpath = '/home/lisa/newmodel.h5'
model = keras.models.load_model(modelpath)

print(model.summary())