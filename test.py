# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.backend as K
# import pandas as pd
# from colors import *
# import os

# # modelspath = '/home/lisa/origin_model/'
# # models = os.listdir(modelspath)
# # # f = open('info.py', 'w')
# # # f.write('\'\'\'')
# # for model in models:
# #     modelpath = os.path.join(modelspath, model)
# #     print(f'>{model}')
# #     model = keras.models.load_model(modelpath)
# #     model.summary()

# modelpath = '/home/lisa/origin_model/lenet5-mnist_origin.h5'
# model = keras.models.load_model(modelpath)
# layer = model.layers[0]
# print(layer.output.shape)



# #     f.write(f'> {model}\n')
# #     f.write(str(type(model.summary())))
# #     f.write('\n')
# # f.write('\'\'\'')
# # f.close()


# # model = keras.Sequential([
# #     keras.layers.InputLayer(input_shape=(4, 4, 3)),
# #     keras.layers.Conv2D(filters=2, kernel_size=(4,4), activation='relu', padding='valid', \
# #         data_format='channels_last', strides=(1,1), dilation_rate=(1,1)),
# #     keras.layers.BatchNormalization(),
# #     keras.layers.Dropout(rate=0.3),
# #     keras.layers.Flatten(),
# #     keras.layers.BatchNormalization(),
# #     keras.layers.Dropout(rate=0.3),
# #     keras.layers.Dense(32, activation='relu')
# # ])
# # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
# # model.summary()




import configparser
parser = configparser.ConfigParser()
parser.read('config.conf')
print(parser['params']['OPspool'].split('\n'))