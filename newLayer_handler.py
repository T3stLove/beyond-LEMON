import tensorflow.keras as keras
from newLayer_impl import *
# from globalInfos 

def _myConv2DLayer_op_to_newlayer(op):
    if op == 'Dense': return myDenseLayer
    elif op == 'Conv2D': return myConv2DLayer
    elif op == 'SeparableConv2D':return mySeparableConv2DLayer
    elif op == 'AveragePooling2D': return myAveragePooling2DLayer
    elif op == 'MaxPooling2D': return myMaxPooling2DLayer
    elif op == 'Dropout': return myDropoutLayer
    elif op == 'SpatialDropout2D': return mySpatialDropout2DLayer
    elif op == 'BatchNormalization': return myBatchNormalizationLayer
    elif op == 'LayerNormalization': return myLayerNormalizationLayer
    elif op == 'GaussianDropout': return myGaussianDropoutLayer
    else:
        raise Exception(Cyan(f'The op {op} does not correspond to any new layer.'))
    

def _myConv2DLayer_indefinite_1(layer, inputshape, mode, op, **indefinite_conv_pooling_kwargs):

    if mode == 'fixed':
        kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    elif mode == 'random':    
        from globalInfos import CONV2D_TYPE_1_POOL
        kerasLayer = _myConv2DLayer_op_to_newlayer(np.random.choice(CONV2D_TYPE_1_POOL))
    else:
        raise Exception(Cyan(f'Unkown mode: {mode}'))
    # UPDATE
    if kerasLayer != mySpatialDropout2DLayer:
        newlayer = kerasLayer(layer, inputshape, **indefinite_conv_pooling_kwargs)
    else:
         newlayer = kerasLayer(layer, definite=False)
    return newlayer

def _myConv2DLayer_indefinite_2(layer, inputshape):
    return myDenseLayer(layer, inputshape, definite=False)

def _myConv2DLayer_indefinite_3(layer, inputshape, mode, op):
    if mode == 'fixed':
        kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    elif mode == 'random':
        from globalInfos import CONV2D_TYPE_3_POOL
        kerasLayer = _myConv2DLayer_op_to_newlayer(np.random.choice(CONV2D_TYPE_3_POOL))
    
    if kerasLayer == myDropoutLayer or kerasLayer == myGaussianDropoutLayer:
        newlayer = kerasLayer(layer, definite=False)
    else:
        newlayer = kerasLayer(layer, inputshape, definite=False)
    return newlayer

def _myConv2DLayer_definite(layer, inputshape):
    newlayer = None
    if isinstance(layer, keras.layers.Dense):
        newlayer = myDenseLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.Conv2D):
        newlayer = myConv2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.SeparableConv2D):
        newlayer = mySeparableConv2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.AveragePooling2D):
        newlayer = myAveragePooling2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.Flatten):
        newlayer = myFlattenLayer(layer)
    elif isinstance(layer, keras.layers.Dropout):
        newlayer = myDropoutLayer(layer)
    elif isinstance(layer, keras.layers.MaxPooling2D):
        newlayer = myMaxPooling2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.BatchNormalization):
        newlayer = myBatchNormalizationLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.LayerNormalization):
        newlayer = myLayerNormalizationLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.SeparableConv2D):
        newlayer = mySpatialDropout2DLayer(layer)
    elif isinstance(layer, keras.layers.GaussianDropout):
        newlayer = myGaussianDropoutLayer(layer)
    if not newlayer:
        raise Exception(Cyan('newlayer is of unexpected type!'))

    return newlayer
    

def myConv2dLayer(layer, definite, subType, inputshape, mode, op, **indefinite_conv_pooling_kwargs):

    if definite:
        return _myConv2DLayer_definite(layer, inputshape)
    else:
        if subType == 1: 
            return _myConv2DLayer_indefinite_1(layer, inputshape, mode, op, **indefinite_conv_pooling_kwargs)
        elif subType == 2:
            return _myConv2DLayer_indefinite_2(layer, inputshape)
        elif subType == 3:
            return _myConv2DLayer_indefinite_3(layer, inputshape, mode, op)
        else:
            raise Exception(Cyan('Unknown subType'))

def myLayer(layer, modelType, definite, subType=None, inputshape=None, mode='random', op=None, **indefinite_conv_pooling_kwargs):

    if modelType == 'conv2d':
        return myConv2dLayer(layer, definite, subType, inputshape, mode, op, **indefinite_conv_pooling_kwargs)

    else:
        raise Exception(Cyan('Unknown modelType'))
