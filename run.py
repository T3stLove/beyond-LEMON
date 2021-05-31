import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
from colors import *
from mutators import addOneLayer
import os, subprocess


modelnames = [
            'vgg16-imagenet',
            # 'densenet121-imagenet',
            'lenet5-mnist',
            'vgg19-imagenet',
            'inception.v3-imagenet',
            'resnet50-imagenet',
            'xception-imagenet',

            
            'alexnet-cifar10',
            # 'lstm0-sinewave',
            'lstm2-price',
            'mobilenet.1.00.224-imagenet',
            'lenet5-fashion-mnist'
            ]

for modelname in modelnames:

    modelpath = f'/share_container/share_host_hy2/origin_model/{modelname}_origin.h5'
    model = keras.models.load_model(modelpath)
    # print(model.summary())

    # for layer in model.layers:
    #     print(layer.get_config())

    # while True:

    destination = '/share_container/share_host_hy2/MHYmutation_2021_5_13'

    for order in [1, 2, 3, 4, 5]:
        for cnt in range(1, 51):
            if order == 1:
                newmodel = addOneLayer(model)
                dest = os.path.join(destination, str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, f'lenet5-mnist_{str(cnt)}.h5')
                newmodel.save(h5dest)
            else:
                order1_dest = os.path.join(destination, str(order-1))
                h5files = os.listdir(order1_dest)
                # htfile = np.random.choice(h5files)
                htfile = ''
                for file in h5files:
                    if file.endswith(f'_{cnt}.h5'):
                        htfile = file
                        break
                if not htfile:
                    raise Exception('Find no h5file')
                htfile_name = htfile[:-3]
                htfile_newname = htfile_name + '_' + str(cnt) + '.h5'

                newmodel = addOneLayer(model)
                dest = os.path.join(destination, str(order))
                if not os.path.exists(dest):
                    subprocess.check_output(f'mkdir -p {dest}', shell=True)
                h5dest = os.path.join(dest, htfile_newname)
                newmodel.save(h5dest)






# print(newmodel.summary())

# print(newmodel.save('/home/lisa/newmodel.h5'))

# for layer in newmodel.layers:
#     print('=======================')
#     print(layer.weights)

'''
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_9 (Conv2D)            (None, 28, 28, 6)         156
_________________________________________________________________
average_pooling2d_9 (Average (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 10, 10, 16)        2416
_________________________________________________________________
average_pooling2d_10 (Averag (None, 5, 5, 16)          0
_________________________________________________________________
flatten_5 (Flatten)          (None, 400)               0
_________________________________________________________________
dense_13 (Dense)             (None, 120)               48120
_________________________________________________________________
dropout_5 (Dropout)          (None, 120)               0
_________________________________________________________________
dense_14 (Dense)             (None, 84)                10164
_________________________________________________________________
dense_15 (Dense)             (None, 10)                850
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
'''