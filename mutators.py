import numpy as np
from numpy.lib.function_base import insert
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.merge import minimum
from colors import *
import globalInfos
from newLayer_handler import myLayer
from collections import namedtuple
import globalInfos

def addOneLayer(model, mode='random', op = None):
    return _addOneLayer_Conv2D(model, mode, op)

def _addOneLayer_Conv2D_addConv2DOrPooling_minSize(layers, index):

    nextLayer = layers[index]
    # haoyang only considers Conv2D, maxpooling and averagepooling
    if len(nextLayer.output.shape) == 2:
       minimage_d1, minimage_d2 = 1, 1
    else:
        minimage_d1, minimage_d2 = nextLayer.output.shape[1], nextLayer.output.shape[2]

    return minimage_d1, minimage_d2

def _decideConv2DOrPoolingParams_strides(image_d1, image_d2, minikerpool_size, maxikerpool_size, output_d1, output_d2):

    if output_d1 == 1:
        kerpool_size_d1 = image_d1
        strides_d1 = 1
    else:
        kerpool_size_d1s = []
        for ks_d1 in range(minikerpool_size[0], maxikerpool_size[0]+1):
            if (image_d1 - ks_d1) % (output_d1 - 1) == 0:
                kerpool_size_d1s.append(ks_d1)
        kerpool_size_d1 = np.random.choice(kerpool_size_d1s)
        strides_d1 = (image_d1 - ks_d1) // (output_d1 - 1)

    if output_d2 == 1:
        kerpool_size_d2 = image_d2
        strides_d2 = 1
    else:
        kerpool_size_d2s = []
        for ks_d2 in range(minikerpool_size[1], maxikerpool_size[1]+1):
            if (image_d2 - ks_d2) % (output_d2 - 1) == 0:
                kerpool_size_d2s.append(ks_d2)
        kerpool_size_d2 = np.random.choice(kerpool_size_d2s)
        strides_d2 = (image_d2 - ks_d2) // (output_d2 - 1)

    kerpool_size = (kerpool_size_d1, kerpool_size_d2)
    strides = (strides_d1, strides_d2)
    return kerpool_size, strides

def _decideConv2DOrPoolingParams_dilation_rate(image_d1, image_d2, minikerpool_size, maxikerpool_size, output_d1, output_d2):

    kerpool_size_d1s = []
    for ks_d1 in range(minikerpool_size[0], maxikerpool_size[0]+1): 
        if ks_d1 == 1:
            kerpool_size_d1s.append(ks_d1)
            continue
        if (image_d1-output_d1+1-ks_d1)%(ks_d1-1) == 0:
            kerpool_size_d1s.append(ks_d1)

    kerpool_size_d1 = np.random.choice(kerpool_size_d1s)
    dilation_rate_1 = (image_d1-output_d1+1-kerpool_size_d1)//(kerpool_size_d1-1)+1

    kerpool_size_d2s = []
    for ks_d2 in range(minikerpool_size[1], maxikerpool_size[1]+1):
        if ks_d2 == 1:
            kerpool_size_d2s.append(ks_d2)
            continue
        if (image_d2-output_d2+1-ks_d2)%(ks_d2-1) == 0:
            kerpool_size_d2s.append(ks_d2)

    kerpool_size_d2 = np.random.choice(kerpool_size_d2s)
    dilation_rate_2 = (image_d2-output_d2+1-kerpool_size_d2)//(kerpool_size_d2-1)+1

    dilation_rate = (dilation_rate_1, dilation_rate_2)
    kerpool_size = (kerpool_size_d1, kerpool_size_d2)
    return kerpool_size, dilation_rate


def _decideConv2DOrPoolingParams(op, minimage_d1, minimage_d2, image_d1, image_d2):

    if minimage_d1 > image_d1 or minimage_d2 > image_d2:
        raise Exception(Cyan('Cannot insert a conv or pooling layer here. Required minimum output size is larger than the input size!'))

    # output dimension of the added conv or pooling layer
    output_d1 = minimage_d1 if minimage_d1 == image_d1 else np.random.randint(minimage_d1, image_d1)
    output_d2 = minimage_d1 if minimage_d2 == image_d2 else np.random.randint(minimage_d2, image_d2)
    padding = 'valid'
    strides, dilation_rate = (1, 1), (1, 1) 

    if op == 'AveragePooling2D' or op == 'MaxPooling2D':
        dilation_or_strides = 'strides'
    elif op == 'Conv2D' or op == 'SeparableConv2D':
        dilation_or_strides = np.random.choice(['dilation_rate', 'strides'])
    else:
        raise Exception(Cyan(f'Unexpected op: {op}'))

    # maxikerpool_size exists when strides = (1,1)
    maxikerpool_size = (image_d1-output_d1+1, image_d2-output_d2+1)
    minikerpool_size = (1, 1)

    if dilation_or_strides == 'strides':
        kerpool_size, strides = _decideConv2DOrPoolingParams_strides(image_d1, image_d2, minikerpool_size, \
                                        maxikerpool_size, output_d1, output_d2)
    else:
        
        kerpool_size, dilation_rate = _decideConv2DOrPoolingParams_dilation_rate(image_d1, image_d2, \
                                minikerpool_size, maxikerpool_size, output_d1, output_d2)

    return kerpool_size, padding, strides, dilation_rate


def _addOneLayer_Conv2D_addConv2DOrPooling(layers, layer, index, layerType, mode, op):

    inputshape = layer.input.shape
    image_d1 = inputshape[1]
    image_d2 = inputshape[2]
    minimage_d1, minimage_d2 = _addOneLayer_Conv2D_addConv2DOrPooling_minSize(layers, index)
    kerpool_size, padding, strides, dilation_rate = _decideConv2DOrPoolingParams(op, minimage_d1, minimage_d2, image_d1, image_d2)

    return myLayer(layer, modelType='conv2d', definite=False, subType=layerType, mode=mode, op=op, \
        kerpool_size=kerpool_size, padding=padding, strides=strides, dilation_rate=dilation_rate)

def _addOneLayer_Conv2D_addOperation(layers, layer, index, layerType, mode, op):
    # if _convOrPooling(layer):
    print(layer, layerType)
    if layerType == 1:
        # UPDATE
        if op != 'SpatialDropout2D':
            return _addOneLayer_Conv2D_addConv2DOrPooling(layers, layer, index, layerType, mode, op)
        else:
            return myLayer(layer, modelType='conv2d', definite=False, subType=layerType, mode=mode, op=op)
    elif layerType == 2 or layerType == 3:
        return myLayer(layer, modelType='conv2d', definite=False, subType=layerType, mode=mode, op=op)
    else:
        raise Exception(Cyan(f'Unkown layerType: {str(layerType)}'))
    
def _addOneLayer_Conv2D_analyze_layerType(op):

    from globalInfos import CONV2D_TYPE_1_POOL, CONV2D_TYPE_2_POOL, CONV2D_TYPE_3_POOL
    if op in CONV2D_TYPE_1_POOL: return 1
    elif op in CONV2D_TYPE_2_POOL: return 2
    elif op in CONV2D_TYPE_3_POOL: return 3
    else:
        raise Exception(Cyan(f'Unkown op: {op}'))

def _addOneLayer_Conv2D_decide_layerType_and_head_tail(mode, op):
    
    if mode == 'random':
        layerTypes = []
        from globalInfos import CONV2D_TYPE_1_POOL, CONV2D_TYPE_2_POOL, CONV2D_TYPE_3_POOL
        if CONV2D_TYPE_1_POOL: layerTypes.append(1)
        if CONV2D_TYPE_2_POOL: layerTypes.append(2)
        if CONV2D_TYPE_3_POOL: layerTypes.append(3)
        layerType = np.random.choice(layerTypes)
    elif mode == 'fixed':
        layerType = _addOneLayer_Conv2D_analyze_layerType(op)  
        # print('op = ', op, ' layerType = ', layerType)
    else:
        raise Exception(Cyan(f'Unkown mode: {mode}'))
    heads, tails = [], []
    from globalInfos import CONV2D_LAYERTYPE1_HEADS,\
                            CONV2D_LAYERTYPE1_TAILS,\
                            CONV2D_LAYERTYPE2_HEADS,\
                            CONV2D_LAYERTYPE2_TAILS,\
                            CONV2D_LAYERTYPE3_HEADS,\
                            CONV2D_LAYERTYPE3_TAILS
    if layerType == 1:
        heads, tails = CONV2D_LAYERTYPE1_HEADS, CONV2D_LAYERTYPE1_TAILS
    elif layerType == 2:
        heads, tails = CONV2D_LAYERTYPE2_HEADS, CONV2D_LAYERTYPE2_TAILS
    elif layerType == 3:
        heads, tails = CONV2D_LAYERTYPE3_HEADS, CONV2D_LAYERTYPE3_TAILS
    else:
        raise Exception(Cyan('Unkown '))
    head_tail_id = np.random.randint(0, len(heads))
    head, tail = heads[head_tail_id], tails[head_tail_id]
    return layerType, head, tail

def decide_data_form(layers):
    for layer in layers:
        config = layer.get_config()
        if 'data_format' in config:
            globalInfos.DATA_FORMAT = config['data_format'] 
            break

def decide_data_type(layers):
    for layer in layers:
        config = layer.get_config()
        if 'dtype' in config:
            globalInfos.DTYPE = config['dtype'] 
            break

def _addOneLayer_Conv2D(model, mode, op):

    layers = model.layers
    layerType, head, tail = _addOneLayer_Conv2D_decide_layerType_and_head_tail(mode, op)
    insertIndex = np.random.randint(head, tail)
    # insertIndex = 4

    print(Red(f'insertIndex = {str(insertIndex)}'))

    decide_data_form(layers)
    decide_data_type(layers)
    for i, layer in enumerate(layers):
        if i == 0:
            inputs = keras.Input(shape=layer.input.shape[1:], dtype=globalInfos.DTYPE)
            model_inputs = inputs
        print('i =', i)

        if i < insertIndex:
            mylayer =  myLayer(layer, modelType='conv2d', definite=True)
        elif i == insertIndex:
            print(Red('Artificial layer construction begins.'))
            print(Blue(layer.__class__.__name__))
            mylayer_ = _addOneLayer_Conv2D_addOperation(layers, layer, i, layerType, mode, op)
            inputs = mylayer_(inputs)
            # newmodel.add(mylayer_)
            # print(Yellow(str(mylayer_.get_config())))
            print(Red('Artificial layer construction finished.'))
            previous_inputshape = mylayer_.output.shape

            # print('previous_inputshape1 =', previous_inputshape)
            mylayer = myLayer(layer, modelType='conv2d', definite=True, inputshape=previous_inputshape)
            # print(Yellow(str(mylayer.get_config())))
        else:
            mylayer = myLayer(layer, modelType='conv2d', definite=True, inputshape=previous_inputshape)

        if i == len(layers) - 1:
            outputs = mylayer(inputs)
        else:
            inputs = mylayer(inputs)
        # newmodel.add(mylayer)
        previous_inputshape = mylayer.output.shape
        # print('previous_inputshape2 =', previous_inputshape)
        # for node in mylayer._inbound_nodes:
        #     print(node.inbound_layers)
    newmodel = keras.Model(inputs=model_inputs, outputs=outputs)
    return newmodel
