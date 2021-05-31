import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from colors import *

def myDenseLayer(layer, inputshape, definite=True):
    config = layer.get_config()
    newlayer = keras.layers.Dense(units=config['units'], trainable=config['trainable'], \
                                    dtype=config['dtype'],use_bias=config['use_bias'], \
                                        activation=config['activation'])
    inputdim = inputshape[-1] if inputshape else layer.input.shape[-1]
    newlayer.build(inputdim)
    # print('>>>', inputdim, layer.output.shape[-1])
    # newlayer.add_weight(shape=(inputdim, layer.output.shape[-1]), initializer="random_normal", trainable=True)
    # if config['use_bias']:
    #     newlayer.add_weight(shape=(layer.output.shape[-1], ), initializer="random_normal", trainable=True)

    if definite and ((inputshape and inputshape[-1] == layer.input.shape[-1]) or not inputshape):
        newlayer.set_weights(layer.get_weights())

    return newlayer

def myConv2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    config = layer.get_config()
    param_inputshape = layer.input.shape
    setweights = True

    if indefinite_conv_pooling_kwargs:
        kernel_size, padding, strides, dilation_rate = indefinite_conv_pooling_kwargs['kerpool_size'], \
                                                       indefinite_conv_pooling_kwargs['padding'], \
                                                       indefinite_conv_pooling_kwargs['strides'], \
                                                       indefinite_conv_pooling_kwargs['dilation_rate']
        filters = np.random.randint(1, 11) 
        activation = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        dtype = 'float32'
        trainable = True
        use_bias = True
        setweights = False
    else:
        filters = config['filters']
        kernel_size = config['kernel_size']
        strides = config['strides']
        padding = config['padding']
        activation = config['activation']
        dtype = config['dtype']
        trainable = config['trainable']
        use_bias = config['use_bias']
        dilation_rate = config['dilation_rate']

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:

        if inputshape[1:-1] == layer.input.shape[1:-1]:
            param_inputshape = inputshape
            setweights = False
        else:
            kernel_size = (inputshape[1]-layer.output.shape[1]+1,\
                            inputshape[2]-layer.output.shape[2]+1)
            param_inputshape = inputshape
            setweights = False
            dilation_rate = (1,1)
            strides = (1,1)

    newlayer = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,\
                                        strides=strides, padding=padding,\
                                        activation=activation, dtype=dtype,\
                                        trainable=trainable, use_bias=use_bias,\
                                        dilation_rate = dilation_rate)
    
    newlayer.build(param_inputshape)
                            
    # print(Red(str([kernel_size[0], kernel_size[1], param_inputshape[-1], filters])))
    # newlayer.add_weight(shape=(kernel_size[0], kernel_size[1], param_inputshape[-1], filters), \
    #                     initializer="random_normal", trainable=True)
    # if use_bias:
    #     newlayer.add_weight(shape=(filters, ), initializer="random_normal", trainable=True)
    if setweights:
        newlayer.set_weights(layer.get_weights())
    return newlayer

def myAveragePooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    config = layer.get_config()
    

    if indefinite_conv_pooling_kwargs:
        pool_size, padding, strides = indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']
        dtype = 'float32'
        trainable = True

    else:
        pool_size = config['pool_size']
        strides = config['strides']
        padding = config['padding']
        trainable = config['trainable']
        dtype = config['dtype']

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            pass
        else:
            pool_size = (inputshape[1]-layer.output.shape[1]+1, \
                        inputshape[2]-layer.output.shape[2]+1)
            strides = (1,1)
            padding = 'valid'

    newlayer = keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, \
                                             trainable=trainable, dtype=dtype)
    return newlayer

def myMaxPooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    config = layer.get_config()
    

    if indefinite_conv_pooling_kwargs:
        pool_size, padding, strides = indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']
        dtype = 'float32'
        trainable = True
    else:
        pool_size = config['pool_size']
        strides = config['strides']
        padding = config['padding']
        trainable = config['trainable']
        dtype = config['dtype']

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            pass
        else:
            pool_size = (inputshape[1]-layer.output.shape[1]+1, \
                        inputshape[2]-layer.output.shape[2]+1)
            strides = (1,1)
            padding = 'valid'

    newlayer = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, \
                                             trainable=trainable, dtype=dtype)
    return newlayer

def myFlattenLayer(layer):
    config = layer.get_config()
    trainable = config['trainable']
    dtype = config['dtype']
    newlayer = keras.layers.Flatten(trainable=trainable, dtype=dtype)
    return newlayer

def myDropoutLayer(layer):
    config = layer.get_config()
    rate = config['rate']
    newlayer = keras.layers.Dropout(rate=rate)
    return newlayer
