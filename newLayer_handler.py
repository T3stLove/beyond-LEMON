import tensorflow.keras as keras
from newLayer_impl import *


SHAPE_CHANGING_LAYERS_CONV2D = [
    myConv2DLayer,
    myAveragePooling2DLayer,
    myMaxPooling2DLayer
]

def _myConv2DLayer_indefinite_conv_pooling(layer, inputshape, **indefinite_conv_pooling_kwargs):

    kerasLayer = np.random.choice(SHAPE_CHANGING_LAYERS_CONV2D)
    newlayer = None
    print(Yellow(f'kerasLayer is {str(kerasLayer)}'))

    newlayer = kerasLayer(layer, inputshape, **indefinite_conv_pooling_kwargs)

    # if kerasLayer == keras.layers.Conv2D:
    #     newlayer = myConv2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs)
    # elif kerasLayer == keras.layers.AveragePooling2D:
    #     newlayer = myAveragePooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs)
    # elif kerasLayer == keras.layers.MaxPooling2D:
    #     newlayer = myMaxPooling2DLayer(layer, inputshape, **indefinite_conv_pooling_kwargs)

    if not newlayer:
        raise Exception('newlayer is of unexpected type!')

    return newlayer

def _myConv2DLayer_indefinite_dense(layer, inputshape):
    return myDenseLayer(layer, inputshape, definite=False)

def _myConv2DLayer_indefinite_dropout():
    pass
def _myConv2DLayer_indefinite_batchnormalization():
    pass

def _myConv2DLayer_definite(layer, inputshape):
    newlayer = None
    if isinstance(layer, keras.layers.Dense):
        newlayer = myDenseLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.Conv2D):
        newlayer = myConv2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.AveragePooling2D):
        newlayer = myAveragePooling2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.Flatten):
        newlayer = myFlattenLayer(layer)
    elif isinstance(layer, keras.layers.Dropout):
        newlayer = myDropoutLayer(layer)
    elif isinstance(layer, keras.layers.MaxPooling2D):
        newlayer = myMaxPooling2DLayer(layer, inputshape)
    if not newlayer:
        raise Exception(Cyan('newlayer is of unexpected type!'))

    return newlayer
    

def myConv2dLayer(layer, definite, subType, inputshape, **indefinite_conv_pooling_kwargs):

    if definite:
        return _myConv2DLayer_definite(layer, inputshape)
    else:
        if subType == 'conv and pooling':
            return _myConv2DLayer_indefinite_conv_pooling(layer, inputshape, **indefinite_conv_pooling_kwargs)
        elif subType == 'dense':
            return _myConv2DLayer_indefinite_dense(layer, inputshape)
        else:
            raise Exception(Cyan('Unknown subType'))

def myLayer(layer, modelType, definite, subType=None, inputshape=None, **indefinite_conv_pooling_kwargs):

    if modelType == 'conv2d':
        return myConv2dLayer(layer, definite, subType, inputshape, **indefinite_conv_pooling_kwargs)

    else:
        raise Exception(Cyan('Unknown modelType'))
