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
    elif op == 'Minimum': return myMinimumLayer
    elif op == 'Maximum': return myMaximumLayer
    elif op == 'Average': return myAverageLayer
    elif op == 'ZeroPadding2D': return myZeroPadding2DLayer
    elif op == 'Cropping2D': return myCropping2DLayer
    else:
        raise Exception(Cyan(f'The op {op} does not correspond to any new layer.'))
    

def _myConv2DLayer_notcopy_1(layer, inputshape, op, **notcopy_kwargs):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    # UPDATE
    if kerasLayer in [myConv2DLayer, mySeparableConv2DLayer, myAveragePooling2DLayer, \
        myMaxPooling2DLayer]:
        newlayer = kerasLayer(layer, inputshape, **notcopy_kwargs)
    elif kerasLayer in [myZeroPadding2DLayer, myCropping2DLayer]:
        newlayer = kerasLayer(layer, **notcopy_kwargs)
    else:
         newlayer = kerasLayer(layer, copy=False)
    return newlayer

def _myConv2DLayer_notcopy_2(layer, inputshape):
    return myDenseLayer(layer, inputshape, copy=False)

def _myConv2DLayer_notcopy_3(layer, inputshape, op, **notcopy_kwargs):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    
    if kerasLayer in [myDropoutLayer, myGaussianDropoutLayer]:
        newlayer = kerasLayer(layer, copy=False)
    elif kerasLayer in [myBatchNormalizationLayer, myLayerNormalizationLayer]:
        newlayer = kerasLayer(layer, inputshape, False, notcopy_kwargs['inlayer'])
    else:
        newlayer = kerasLayer(layer, inputshape, copy=False)
    return newlayer

def _myConv2DLayer_notcopy_4(layer, op):

    kerasLayer = _myConv2DLayer_op_to_newlayer(op)
    newlayer = kerasLayer(layer, copy=False)    
    return newlayer

def _myConv2DLayer_copy(layer, inputshape):
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
        newlayer = mySeparableConv2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.GaussianDropout):
        newlayer = myGaussianDropoutLayer(layer)
    elif isinstance(layer, keras.layers.Add):
        newlayer = myAddLayer(layer)
    elif isinstance(layer, keras.layers.Minimum):
        newlayer = myMinimumLayer(layer)
    elif isinstance(layer, keras.layers.Maximum):
        newlayer = myMaximumLayer(layer)
    elif isinstance(layer, keras.layers.Average):
        newlayer = myAverageLayer(layer)
    elif isinstance(layer, keras.layers.Reshape):
        newlayer = myReshapeLayer(layer)
    elif isinstance(layer, keras.layers.SpatialDropout2D):
        newlayer = mySpatialDropout2DLayer(layer)
    elif isinstance(layer, keras.layers.Activation):
        newlayer = myActivationLayer(layer)
    elif isinstance(layer, keras.layers.Concatenate):
        newlayer = myConcatenateLayer(layer)
    elif isinstance(layer, keras.layers.ReLU):
        newlayer = myReluLayer(layer)
    elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
        newlayer = myGlobalAveragePooling2DLayer(layer)
    elif isinstance(layer, keras.layers.DepthwiseConv2D):
        newlayer = myDepthwiseConv2DLayer(layer, inputshape)
    elif isinstance(layer, keras.layers.ZeroPadding2D):
        newlayer = myZeroPadding2DLayer(layer)
    if not newlayer:
        raise Exception(Cyan('newlayer is of unexpected type!'))

    return newlayer
    

def myConv2dLayer(layer, copy, subType, inputshape, op, **notcopy_kwargs):

    if copy:
        return _myConv2DLayer_copy(layer, inputshape)
    else:
        if subType == 1: 
            return _myConv2DLayer_notcopy_1(layer, inputshape, op, **notcopy_kwargs)
        elif subType == 2:
            return _myConv2DLayer_notcopy_2(layer, inputshape)
        elif subType == 3:
            return _myConv2DLayer_notcopy_3(layer, inputshape, op, **notcopy_kwargs)
        elif subType == 4:
            return _myConv2DLayer_notcopy_4(layer, op)
        else:
            raise Exception(Cyan('Unknown subType'))

def myLayer(layer, modelType, copy, subType=None, inputshape=None, op=None, **notcopy_kwargs):

    if modelType == 'conv2d':
        return myConv2dLayer(layer, copy, subType, inputshape, op, **notcopy_kwargs)

    else:
        raise Exception(Cyan('Unknown modelType'))
