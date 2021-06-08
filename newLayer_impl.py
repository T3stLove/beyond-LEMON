import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from colors import *
from globalInfos import LAYER_NAME

def _setName(layer):
    className = layer.__class__.__name__
    id = -1
    if className not in LAYER_NAME:
        LAYER_NAME[className] = 1
        id = 1
    else:
        id = LAYER_NAME[className] 
        id += 1
        LAYER_NAME[className] = id
    return className + '_' + str(id)

def myDenseLayer(layer, inputshape, definite=True):
    config = layer.get_config()
    if not definite:
        config['units'] = np.random.randint(1, 101)
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
    inputdim = inputshape[-1] if inputshape else layer.input.shape[-1]
    newlayer = keras.layers.Dense.from_config(config)
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
        config['kernel_size'], config['padding'], config['strides'], config['dilation_rate'] = \
                                                       indefinite_conv_pooling_kwargs['kerpool_size'], \
                                                       indefinite_conv_pooling_kwargs['padding'], \
                                                       indefinite_conv_pooling_kwargs['strides'], \
                                                       indefinite_conv_pooling_kwargs['dilation_rate']
        config['filters'] = np.random.randint(1, 11) 
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        config['dtype'] = 'float32'
        config['trainable'] = True
        config['use_bias'] = True
        config['setweights'] = False
    
    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:

        if inputshape[1:-1] == layer.input.shape[1:-1]:
            param_inputshape = inputshape
            setweights = False
        else:
            config['kernel_size'] = (inputshape[1]-layer.output.shape[1]+1,\
                            inputshape[2]-layer.output.shape[2]+1)
            param_inputshape = inputshape
            setweights = False
            config['dilation_rate'] = (1,1)
            config['strides'] = (1,1)

    config['name'] = _setName(layer)
    newlayer = keras.layers.Conv2D.from_config(config)
    
    newlayer.build(param_inputshape)
    if setweights:
        newlayer.set_weights(layer.get_weights())
    return newlayer

def myAveragePooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    config = layer.get_config()
    
    if indefinite_conv_pooling_kwargs:
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']
        config['dtype'] = 'float32'
        config['trainable'] = True

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            pass
        else:
            config['pool_size'] = (inputshape[1]-layer.output.shape[1]+1, \
                        inputshape[2]-layer.output.shape[2]+1)
            config['strides'] = (1,1)
            config['padding'] = 'valid'

    config['name'] = _setName(layer)
    newlayer = keras.layers.MaxPooling2D.from_config(config)
    return newlayer

def myMaxPooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    config = layer.get_config()
    if indefinite_conv_pooling_kwargs:
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']
        config['dtype'] = 'float32'
        config['trainable'] = True

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            pass
        else:
            config['pool_size'] = (inputshape[1]-layer.output.shape[1]+1, \
                        inputshape[2]-layer.output.shape[2]+1)
            config['strides'] = (1,1)
            config['padding'] = 'valid'
    
    config['name'] = _setName(layer)
    newlayer = keras.layers.MaxPooling2D.from_config(config)
    return newlayer

def myFlattenLayer(layer):
    config = layer.get_config()
    config['name'] = _setName(layer)
    newlayer = keras.layers.Flatten.from_config(config)
    return newlayer

def myDropoutLayer(layer, definite=True):
    config = layer.get_config()
    if not definite:
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    config['name'] = _setName(layer)
    newlayer = keras.layers.Dropout.from_config(config)
    return newlayer

def mySpatialDropout2DLayer(layer, definite=True):
    config = layer.get_config()
    if not definite:
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    config['name'] = _setName(layer)
    newlayer = keras.layers.Dropout.from_config(config)
    return newlayer

# def myGlobalMaxPooling2DLayer(layer):

def myBatchNormalizationLayer(layer, inputshape, definite=True):
    # config = layer.get_config()
    # if not 
    pass