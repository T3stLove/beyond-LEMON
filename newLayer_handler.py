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
    elif op == 'Add': return myAddLayer
    elif op == 'ZeroPadding2D': return myZeroPadding2DLayer
    elif op == 'Cropping2D': return myCropping2DLayer
    else:
        raise Exception(Cyan(f'The op {op} does not correspond to any new layer.'))
    

def _myConv2DLayer_indefinite_1(layer, inputshape, op, **indefinite_kwargs):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    # UPDATE
    if kerasLayer in [myConv2DLayer, mySeparableConv2DLayer, myAveragePooling2DLayer, \
        myMaxPooling2DLayer]:
        newlayer = kerasLayer(layer, inputshape, **indefinite_kwargs)
    elif kerasLayer in [myZeroPadding2DLayer, myCropping2DLayer]:
        newlayer = kerasLayer(layer, **indefinite_kwargs)
    else:
         newlayer = kerasLayer(layer, definite=False)
    return newlayer

def _myConv2DLayer_indefinite_2(layer, inputshape):
    return myDenseLayer(layer, inputshape, definite=False)

def _myConv2DLayer_indefinite_3(layer, inputshape, op):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    
    if kerasLayer in [myDropoutLayer, myGaussianDropoutLayer]:
        newlayer = kerasLayer(layer, definite=False)
    else:
        newlayer = kerasLayer(layer, inputshape, definite=False)
    return newlayer

def _myConv2DLayer_indefinite_4(layer, op):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    newlayer = kerasLayer(layer, definite=False)    
    return newlayer

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
    elif isinstance(layer, keras.layers.BatchNormalization):
        newlayer = myBatchNormalizationLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.LayerNormalization):
        newlayer = myLayerNormalizationLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.SeparableConv2D):
        newlayer = mySpatialDropout2DLayer(layer)
    elif isinstance(layer, keras.layers.GaussianDropout):
        newlayer = myGaussianDropoutLayer(layer)
    elif isinstance(layer, keras.layers.Add):
        newlayer = myAddLayer(layer)
    elif isinstance(layer, keras.layers.Reshape):
        newlayer = myReshapeLayer(layer)
    if not newlayer:
        raise Exception(Cyan('newlayer is of unexpected type!'))

    return newlayer
    

def myConv2dLayer(layer, definite, subType, inputshape, op, **indefinite_kwargs):

    if definite:
        return _myConv2DLayer_definite(layer, inputshape)
    else:
        if subType == 1: 
            return _myConv2DLayer_indefinite_1(layer, inputshape, op, **indefinite_kwargs)
        elif subType == 2:
            return _myConv2DLayer_indefinite_2(layer, inputshape)
        elif subType == 3:
            return _myConv2DLayer_indefinite_3(layer, inputshape, op)
        elif subType == 4:
            return _myConv2DLayer_indefinite_4(layer, op)
        else:
            raise Exception(Cyan('Unknown subType'))

def myLayer(layer, modelType, definite, subType=None, inputshape=None, op=None, **indefinite_kwargs):

    if modelType == 'conv2d':
        return myConv2dLayer(layer, definite, subType, inputshape, op, **indefinite_kwargs)

    else:
        raise Exception(Cyan('Unknown modelType'))
