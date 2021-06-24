import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from colors import *

def _setName(layerclass):
    className = layerclass.__name__
    id = -1
    from globalInfos import LAYER_CLASS_NAMES
    if className not in LAYER_CLASS_NAMES:
        LAYER_CLASS_NAMES[className] = 1
        id = 1
    else:
        id = LAYER_CLASS_NAMES[className] 
        id += 1
        LAYER_CLASS_NAMES[className] = id
    return 'hyPLUSqc' + '-' + className + '_' + str(id)

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
                'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last',
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
    elif className == 'GaussianDropout':
        return {'name': 'gaussian_dropout', 'trainable': True, 'dtype': 'float32', 'rate': 0.9}
    elif className == 'Add':
        return {'name': 'add', 'trainable': True, 'dtype': 'float32'}
    elif className == 'Reshape':
        return {'name': 'reshape', 'trainable': True, 'dtype': 'float32', 'target_shape': (27,)}
    elif className == 'ZeroPadding2D':
        return {'name': 'zero_padding2d', 'trainable': True, 'dtype': 'float32', 'padding': ((1, 2), (2, 1)), 'data_format': 'channels_last'}
    elif className == 'Cropping2D':
        return {'name': 'cropping2d', 'trainable': True, 'dtype': 'float32', 'cropping': ((1, 2), (2, 1)), 'data_format': 'channels_last'}
    elif className == 'Maximum':
        return {'name': 'maximum', 'trainable': True, 'dtype': 'float32'}
    elif className == 'Minimum':
        return {'name': 'minimum', 'trainable': True, 'dtype': 'float32'}
    elif className == 'Average':
        return {'name': 'average', 'trainable': True, 'dtype': 'float32'}
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

def myDenseLayer(layer, inputshape, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Dense)
        config['units'] = np.random.randint(1, 101)
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        config['name'] = _setName(keras.layers.Dense)
        _setExtraConfigInfo(keras.layers.Dense, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Dense.from_config(config)
    inputdim = inputshape[-1] if inputshape else layer.input.shape[-1]
    newlayer.build(inputdim)
    # print('>>>', inputdim, layer.output.shape[-1])
    # newlayer.add_weight(shape=(inputdim, layer.output.shape[-1]), initializer="random_normal", trainable=True)
    # if config['use_bias']:
    #     newlayer.add_weight(shape=(layer.output.shape[-1], ), initializer="random_normal", trainable=True)

    if copy and ((inputshape and inputshape[-1] == layer.input.shape[-1]) or not inputshape):
        newlayer.set_weights(layer.get_weights())

    return newlayer

def myDepthwiseConv2DLayer(layer, inputshape):
    param_inputshape = layer.input.shape
    setweights = True
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
            config['padding'] = 'valid'
    newlayer = keras.layers.DepthwiseConv2D.from_config(config)
    newlayer.build(param_inputshape)
    if setweights:
        newlayer.set_weights(layer.get_weights())
    return newlayer

def myConv2DLayer(layer, inputshape, **indefinite_kwargs):
    param_inputshape = layer.input.shape
    setweights = True

    if indefinite_kwargs:
        config = _getConfig(keras.layers.Conv2D)
        config['kernel_size'], config['padding'], config['strides'], config['dilation_rate'] = \
                                                       indefinite_kwargs['kerpool_size'], \
                                                       indefinite_kwargs['padding'], \
                                                       indefinite_kwargs['strides'], \
                                                       indefinite_kwargs['dilation_rate']
        config['filters'] = np.random.randint(1, 11) 
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        setweights = False
        config['name'] = _setName(keras.layers.Conv2D)
        _setExtraConfigInfo(keras.layers.Conv2D, config)
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
            config['padding'] = 'valid'

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

def mySeparableConv2DLayer(layer, inputshape, **indefinite_kwargs):
    param_inputshape = layer.input.shape
    setweights = True

    if indefinite_kwargs:
        config = _getConfig(keras.layers.SeparableConv2D)
        config['kernel_size'], config['padding'], config['strides'], config['dilation_rate'] = \
            indefinite_kwargs['kerpool_size'], \
            indefinite_kwargs['padding'], \
            indefinite_kwargs['strides'], \
            indefinite_kwargs['dilation_rate']
        config['filters'] = np.random.randint(1, 11)
        config['activation'] = np.random.choice(['relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        setweights = False
        config['name'] = _setName(keras.layers.SeparableConv2D)
        _setExtraConfigInfo(keras.layers.SeparableConv2D, config)
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
            config['padding'] = 'valid'

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

def myAveragePooling2DLayer(layer, inputshape, **indefinite_kwargs):

    if indefinite_kwargs:
        config = _getConfig(keras.layers.AveragePooling2D)
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_kwargs['kerpool_size'], \
                                      indefinite_kwargs['padding'], \
                                      indefinite_kwargs['strides']
        config['name'] = _setName(keras.layers.AveragePooling2D)
        _setExtraConfigInfo(keras.layers.AveragePooling2D, config)

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
    newlayer = keras.layers.AveragePooling2D.from_config(config)
    return newlayer

def myMaxPooling2DLayer(layer, inputshape, **indefinite_kwargs):
    if indefinite_kwargs:
        config = _getConfig(keras.layers.MaxPooling2D)
        config['pool_size'], config['padding'], config['strides'] = \
                                      indefinite_kwargs['kerpool_size'], \
                                      indefinite_kwargs['padding'], \
                                      indefinite_kwargs['strides']
        config['name'] = _setName(keras.layers.MaxPooling2D)
        _setExtraConfigInfo(keras.layers.MaxPooling2D, config)
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
    newlayer = keras.layers.MaxPooling2D.from_config(config)
    return newlayer

def myFlattenLayer(layer):
    return layer.__class__.from_config(layer.get_config())

def myDropoutLayer(layer, copy=True):
    if copy:
        config = layer.get_config()
        config['name'] = _setName(keras.layers.Dropout)
        _setExtraConfigInfo(keras.layers.Dropout, config)
    else:
        config = _getConfig(keras.layers.Dropout)
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    newlayer = keras.layers.Dropout.from_config(config)
    return newlayer

def mySpatialDropout2DLayer(layer, copy=True):
    if copy:
        config = layer.get_config()
        config['name'] = _setName(keras.layers.SpatialDropout2D)
        _setExtraConfigInfo(keras.layers.SpatialDropout2D, config)
    else:
        config = _getConfig(keras.layers.SpatialDropout2D)
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    newlayer = keras.layers.SpatialDropout2D.from_config(config)
    return newlayer

def myGaussianDropoutLayer(layer, copy=True):
    if copy:
        config = layer.get_config()
        config['name'] = _setName(keras.layers.GaussianDropout)
        _setExtraConfigInfo(keras.layers.GaussianDropout, config)
    else:
        config = _getConfig(keras.layers.GaussianDropout)
        config['rate'] = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    newlayer = keras.layers.GaussianDropout.from_config(config)
    return newlayer

def myBatchNormalizationLayer(layer, inputshape, copy=True, inlayer=None):
    if not copy:
        config = _getConfig(keras.layers.BatchNormalization)
        config['momentum'] = np.random.rand()
        config['name'] = _setName(keras.layers.BatchNormalization)
        _setExtraConfigInfo(keras.layers.BatchNormalization, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.BatchNormalization.from_config(config)
    if inlayer:
        inputdim = inputshape if inputshape else inlayer.output.shape
    else:
        inputdim = inputshape if inputshape else layer.input.shape
    newlayer.build(inputdim)
    if inlayer:
        if copy and ((inputshape and inputshape[-1] == inlayer.output.shape[-1]) or not inputshape):
            newlayer.set_weights(layer.get_weights())
    else:
        if copy and ((inputshape and inputshape[-1] == layer.output.shape[-1]) or not inputshape):
            newlayer.set_weights(layer.get_weights())
    return newlayer
  
def myLayerNormalizationLayer(layer, inputshape, copy=True, inlayer=None):
    if not copy:
        config = _getConfig(keras.layers.LayerNormalization)
        config['name'] = _setName(keras.layers.LayerNormalization)
        _setExtraConfigInfo(keras.layers.LayerNormalization, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.LayerNormalization.from_config(config)
    if inlayer:
        inputdim = inputshape if inputshape else inlayer.output.shape
    else:
        inputdim = inputshape if inputshape else layer.input.shape
    newlayer.build(inputdim)
    if inlayer:
        if copy and ((inputshape and inputshape[-1] == inlayer.output.shape[-1]) or not inputshape):
            newlayer.set_weights(layer.get_weights())
    else:
        if copy and ((inputshape and inputshape[-1] == layer.output.shape[-1]) or not inputshape):
            newlayer.set_weights(layer.get_weights())
    return newlayer

def myAddLayer(layer, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Add)
        config['name'] = _setName(keras.layers.Add)
        _setExtraConfigInfo(keras.layers.Add, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Add.from_config(config)
    return newlayer

def myMinimumLayer(layer, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Minimum)
        config['name'] = _setName(keras.layers.Minimum)
        _setExtraConfigInfo(keras.layers.Minimum, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Minimum.from_config(config)
    return newlayer

def myMaximumLayer(layer, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Maximum)
        config['name'] = _setName(keras.layers.Maximum)
        _setExtraConfigInfo(keras.layers.Maximum, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Maximum.from_config(config)
    return newlayer

def myAverageLayer(layer, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Average)
        config['name'] = _setName(keras.layers.Average)
        _setExtraConfigInfo(keras.layers.Average, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Average.from_config(config)
    return newlayer

def myReshapeLayer(layer, target_shape=None, copy=True):
    if not copy:
        config = _getConfig(keras.layers.Reshape)
        config['target_shape'] = target_shape
        config['name'] = _setName(keras.layers.Reshape)
        _setExtraConfigInfo(keras.layers.Reshape, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Reshape.from_config(config)
    return newlayer

def myZeroPadding2DLayer(layer, **indefinite_kwargs):
    if indefinite_kwargs:
        config = _getConfig(keras.layers.ZeroPadding2D)
        config['padding'] = indefinite_kwargs['padding']
        config['name'] = _setName(keras.layers.ZeroPadding2D)
        _setExtraConfigInfo(keras.layers.ZeroPadding2D, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.ZeroPadding2D.from_config(config)
    return newlayer

def myCropping2DLayer(layer, **indefinite_kwargs):
    if indefinite_kwargs:
        config = _getConfig(keras.layers.Cropping2D)
        config['cropping'] = indefinite_kwargs['cropping']
        config['name'] = _setName(keras.layers.Cropping2D)
        _setExtraConfigInfo(keras.layers.Cropping2D, config)
    else:
        config = layer.get_config()
    newlayer = keras.layers.Cropping2D.from_config(config)
    return newlayer

def myConcatenateLayer(layer):
    return layer.__class__.from_config(layer.get_config())

def myActivationLayer(layer):
    return layer.__class__.from_config(layer.get_config())

def myReluLayer(layer):
    return layer.__class__.from_config(layer.get_config())

def myGlobalAveragePooling2DLayer(layer):
    return layer.__class__.from_config(layer.get_config())