import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from colors import *

def _setName(layerclass):
    className = layerclass.__name__
    id = -1
    from globalInfos import LAYER_NAME
    if className not in LAYER_NAME:
        LAYER_NAME[className] = 1
        id = 1
    else:
        id = LAYER_NAME[className] 
        id += 1
        LAYER_NAME[className] = id
    return className + '_' + str(id)

def _getConfig(layerclass):
    className = layerclass.__name__
    if className == 'Conv2D':
        return {'name': 'conv2d', 'trainable': True, 'dtype': 'float32', 'filters': 1, \
            'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last', \
                'dilation_rate': (1, 1), 'groups': 1, 'activation': 'relu', 'use_bias': True, \
                    'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, \
                        'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, \
                            'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
    if className == 'SeparableConv2D':
        return {'name': 'separable_conv2d', 'trainable': True, 'dtype': 'float32', 'filters': 1,
                'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                'dilation_rate': (1, 1), 'groups': 1, 'activation': 'relu', 'use_bias': True,
                'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None,
                'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
                'bias_constraint': None, 'depth_multiplier': 1,
                'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None,
                'pointwise_constraint': None}

    elif className == 'Dense':
        return {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 10, 'activation': 'linear', \
            'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, \
                'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, \
                    'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
    elif className == 'SpatialDropout2D':
        return {'name': 'spatial_dropout2d', 'trainable': True, 'dtype': 'float32', 'rate': 0.3, 'noise_shape': None, 'seed': None}
    elif className == 'Dropout':
        return {'name': 'dropout', 'trainable': True, 'dtype': 'float32', 'rate': 0.3, 'noise_shape': None, 'seed': None}
    elif className == 'BatchNormalization':
        return {'name': 'batch_normalization', 'trainable': True, 'dtype': 'float32', 'axis': -1, \
            'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, \
                'beta_initializer': {'class_name': 'Zeros', 'config': {}}, \
                    'gamma_initializer': {'class_name': 'Ones', 'config': {}}, \
                        'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, \
                            'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, \
                                'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}
    elif className == 'AveragePooling2D':
        return {'name': 'average_pooling2d', 'trainable': True, 'dtype': 'float32', 'pool_size': (1, 1), \
            'padding': 'valid', 'strides': (1, 1), 'data_format': 'channels_last'}
    elif className == 'MaxPooling2D':
        return {'name': 'max_pooling2d', 'trainable': True, 'dtype': 'float32', 'pool_size': (1, 1), \
            'padding': 'valid', 'strides': (1, 1), 'data_format': 'channels_last'}
    elif className == 'LayerNormalization':
        return {'name': 'layer_normalization', 'trainable': True, 'dtype': 'float32', 'axis': -1, 'epsilon': 0.001, \
            'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, \
                'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, \
                    'beta_constraint': None, 'gamma_constraint': None}
    elif className == 'Flatten':
        return {'name': 'flatten', 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}
    else:
        raise Exception(Cyan(f'Unknown className: {className}'))

def _setExtraConfigInfo(layerclass, config):
    className = layerclass.__name__
    if className == 'Conv2D' or className == 'SeparableConv2D' \
            or className == 'AveragePooling2D' or className == 'MaxPooling2D':
        from globalInfos import DATA_FORMAT
        config['data_format'] = DATA_FORMAT
    from globalInfos import DTYPE
    config['dtype'] = DTYPE

def myDenseLayer(layer, inputshape, definite=True):
    if not definite:
        config = _getConfig(keras.layers.Dense)
        config['units'] = np.random.randint(1, 101)
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
    else:
        config = layer.get_config()
    config['name'] = _setName(keras.layers.Dense)
    _setExtraConfigInfo(keras.layers.Dense, config)
    newlayer = keras.layers.Dense.from_config(config)
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
    param_inputshape = layer.input.shape
    setweights = True

    if indefinite_conv_pooling_kwargs:
        config = _getConfig(keras.layers.Conv2D)
        config['kernel_size'], config['padding'], config['strides'], config['dilation_rate'] = \
                                                       indefinite_conv_pooling_kwargs['kerpool_size'], \
                                                       indefinite_conv_pooling_kwargs['padding'], \
                                                       indefinite_conv_pooling_kwargs['strides'], \
                                                       indefinite_conv_pooling_kwargs['dilation_rate']
        config['filters'] = np.random.randint(1, 11) 
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        setweights = False
    else:
        config = layer.get_config()

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        setweights = False
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            param_inputshape = inputshape
        else:
            config['kernel_size'] = (inputshape[1]-layer.output.shape[1]+1,\
                                    inputshape[2]-layer.output.shape[2]+1)
            param_inputshape = inputshape
            config['dilation_rate'] = (1,1)
            config['strides'] = (1,1)

    config['name'] = _setName(keras.layers.Conv2D)
    _setExtraConfigInfo(keras.layers.Conv2D, config)
    newlayer = keras.layers.Conv2D.from_config(config)
    newlayer.build(param_inputshape)
                            
    # print(Red(str([kernel_size[0], kernel_size[1], param_inputshape[-1], filters])))
    # newlayer.add_weight(shape=(kernel_size[0], kernel_size[1], param_inputshape[-1], filters), \
    #                     initializer="random_normal", trainable=True)
    # if use_bias:
    #     newlayer.add_weight(shape=(filters, ), initializer="random_normal", trainable=True)
    if setweights:
        newlayer.set_weights(layer.get_weights())
    return newlayer

def mySeparableConv2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    param_inputshape = layer.input.shape
    setweights = True

    if indefinite_conv_pooling_kwargs:
        config = _getConfig(keras.layers.SeparableConv2D)
        config['kernel_size'], config['padding'], config['strides'], config['dilation_rate'] = \
            indefinite_conv_pooling_kwargs['kerpool_size'], \
            indefinite_conv_pooling_kwargs['padding'], \
            indefinite_conv_pooling_kwargs['strides'], \
            indefinite_conv_pooling_kwargs['dilation_rate']
        config['filters'] = np.random.randint(1, 11)
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        setweights = False
    else:
        config = layer.get_config()

    if not inputshape or inputshape[1:] == layer.input.shape[1:]:
        pass
    else:
        setweights = False
        if inputshape[1:-1] == layer.input.shape[1:-1]:
            param_inputshape = inputshape
        else:
            config['kernel_size'] = (inputshape[1]-layer.output.shape[1]+1, \
                                     inputshape[2]-layer.output.shape[2]+1)
            param_inputshape = inputshape
            config['dilation_rate'] = (1,1)
            config['strides'] = (1,1)

    config['name'] = _setName(keras.layers.SeparableConv2D)
    _setExtraConfigInfo(keras.layers.Conv2D, config)
    newlayer = keras.layers.SeparableConv2D.from_config(config)
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

    if indefinite_conv_pooling_kwargs:
        config = _getConfig(keras.layers.AveragePooling2D)
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']

    else:
        config = layer.get_config()

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
    config['name'] = _setName(keras.layers.AveragePooling2D)
    _setExtraConfigInfo(keras.layers.AveragePooling2D, config)
    newlayer = keras.layers.AveragePooling2D.from_config(config)
    return newlayer

def myMaxPooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs):
    if indefinite_conv_pooling_kwargs:
        config = _getConfig(keras.layers.MaxPooling2D)
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_conv_pooling_kwargs['kerpool_size'], \
                                      indefinite_conv_pooling_kwargs['padding'], \
                                      indefinite_conv_pooling_kwargs['strides']
    else:
        config = layer.get_config()

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
    config['name'] = _setName(keras.layers.MaxPooling2D)
    _setExtraConfigInfo(keras.layers.MaxPooling2D, config)
    newlayer = keras.layers.MaxPooling2D.from_config(config)
    return newlayer

def myFlattenLayer(layer):
    config = layer.get_config()
    config['name'] = _setName(keras.layers.Flatten)
    _setExtraConfigInfo(keras.layers.Flatten, config)
    newlayer = keras.layers.Flatten.from_config(config)
    return newlayer

def myDropoutLayer(layer, definite=True):
    if definite:
        config = layer.get_config()
    else:
        config = _getConfig(keras.layers.Dropout)
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    config['name'] = _setName(keras.layers.Dropout)
    _setExtraConfigInfo(keras.layers.Dropout, config)
    newlayer = keras.layers.Dropout.from_config(config)
    return newlayer

def mySpatialDropout2DLayer(layer, definite=True):
    if definite:
        config = layer.get_config()
    else:
        config = _getConfig(keras.layers.SpatialDropout2D)
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    config['name'] = _setName(keras.layers.SpatialDropout2D)
    _setExtraConfigInfo(keras.layers.SpatialDropout2D, config)
    newlayer = keras.layers.SpatialDropout2D.from_config(config)
    return newlayer

def myBatchNormalizationLayer(layer, inputshape, definite=True):
    if not definite:
        config = _getConfig(keras.layers.BatchNormalization)
        config['momentum'] = np.random.rand()
        '''axis will be assigned automatically'''
        # if DATA_FORMAT == 'channels_last':
        #     config['axis'] = 3
        # elif DATA_FORMAT == 'channels_first':
        #     config['axis'] = 1
        # else:
        #     raise Exception(Cyan(f'Unkown axis: {str(axis)}'))
    else:
        config = layer.get_config()
    config['name'] = _setName(keras.layers.BatchNormalization)
    _setExtraConfigInfo(keras.layers.BatchNormalization, config)
    newlayer = keras.layers.BatchNormalization.from_config(config)
    inputdim = inputshape if inputshape else layer.input.shape
    newlayer.build(inputdim)
    if definite and ((inputshape and inputshape[-1] == layer.input.shape[-1]) or not inputshape):
        newlayer.set_weights(layer.get_weights())
    return newlayer
  
def myLayerNormalizationLayer(layer, inputshape, definite=True):
    if not definite:
        config = _getConfig(keras.layers.LayerNormalization)
    else:
        config = layer.get_config()
    config['name'] = _setName(keras.layers.LayerNormalization)
    _setExtraConfigInfo(keras.layers.LayerNormalization, config)
    newlayer = keras.layers.LayerNormalization.from_config(config)
    inputdim = inputshape if inputshape else layer.input.shape
    newlayer.build(inputdim)
    if definite and ((inputshape and inputshape[-1] == layer.input.shape[-1]) or not inputshape):
        newlayer.set_weights(layer.get_weights())
    return newlayer
