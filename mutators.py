import numpy as np
from numpy.lib.function_base import insert
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.merge import minimum
from colors import *
from newLayer_handler import myLayer
from collections import namedtuple
from globalInfos import LAYERTYPE1_HEADS,\
                                LAYERTYPE1_TAILS,\
                                LAYERTYPE2_HEADS,\
                                LAYERTYPE2_TAILS,\
                                LAYERTYPE3_HEADS,\
                                LAYERTYPE3_TAILS

def addOneLayer(model, mode='random', op = None):
    return _addOneLayer_Conv2D(model, mode, op)

def _addOneLayer_Conv2D_addConv2DOrPooling_minSize(layers, layersNumber, index):
    # infos = namedtuple('infos', ['kerpool_size', 'strides', 'padding', 'dilation_rate'])
    # infolist = []
    # print('index, layersNumber: ', index, layersNumber)
    # for j in range(index, layersNumber):
    #     _layer_ = layers[j]
    #     if not _convOrPooling(_layer_):
    #         break
    #     config = _layer_.get_config()
    #     if 'Conv2D' in _layer_.__class__.__name__:
    #         infolist.append(infos(config['kernel_size'], config['strides'], config['padding'], config['dilation_rate']))
    #     elif 'Pooling2D' in _layer_.__class__.__name__ and 'Global' not in _layer_.__class__.__name__:
    #         infolist.append(infos(config['pool_size'], config['strides'], config['padding'], None))
    #     else:
    #         raise Exception(Cyan('Unprocessed _layer_.__class__'))
    # minimage_d1 = 1
    # minimage_d2 = 1
    # print('infolist length:', len(infolist))
    # for info in infolist[::-1]:
    #     kerpool_size = info.kerpool_size
    #     strides = info.strides
    #     padding = info.padding
    #     dilation_rate = info.dilation_rate
    #     if padding == 'valid':
    #         if dilation_rate:
    #             if dilation_rate != (1,1) and strides == (1,1):
    #                 minimage_d1 = minimage_d1-1+kerpool_size[0]+(kerpool_size[0]-1)*(dilation_rate[0]-1)
    #                 minimage_d2 = minimage_d2-1+kerpool_size[1]+(kerpool_size[1]-1)*(dilation_rate[1]-1)
    #             elif dilation_rate == (1,1) and strides != (1,1):
    #                 minimage_d1 = (minimage_d1-1)*strides[0]+kerpool_size[0]
    #                 minimage_d2 = (minimage_d2-1)*strides[1]+kerpool_size[1]
    #             elif dilation_rate == (1,1) and strides == (1,1):
    #                 minimage_d1 = minimage_d1-1+kerpool_size[0]
    #                 minimage_d2 = minimage_d2-1+kerpool_size[1]
    #             else:
    #                 raise Exception(Cyan('Conflict between dilation_rate and strides'))
    #         else:
    #             minimage_d1 = (minimage_d1-1)*strides[0]+kerpool_size[0]
    #             minimage_d2 = (minimage_d2-1)*strides[1]+kerpool_size[1]

    #     elif padding == 'same':
    #         minimage_d1 = max(minimage_d1, kerpool_size[0])
    #         minimage_d2 = max(minimage_d2, kerpool_size[1])
    #     else:
    #         raise Exception(Cyan('Unprocessed padding type'))
        
    #     print('>>>>>', minimage_d1, minimage_d1)
    
    nextLayer = layers[index]
    # haoyang only considers Conv2D, maxpooling and averagepooling
    if not isinstance(nextLayer, keras.layers.Conv2D) and \
       not isinstance(nextLayer, keras.layers.AveragePooling2D) and \
       not isinstance(nextLayer, keras.layers.MaxPooling2D):
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


def _decideConv2DOrPoolingParams(layer, minimage_d1, minimage_d2, image_d1, image_d2):

    if minimage_d1 > image_d1 or minimage_d2 > image_d2:
        raise Exception(Cyan('Cannot insert a conv or pooling layer here. Required minimum output size is larger than the input size!'))

    # output dimension of the added conv or pooling layer
    output_d1 = minimage_d1 if minimage_d1 == image_d1 else np.random.randint(minimage_d1, image_d1)
    output_d2 = minimage_d1 if minimage_d2 == image_d2 else np.random.randint(minimage_d2, image_d2)
    padding = 'valid'
    strides, dilation_rate = (1, 1), (1, 1) 

    if 'Pooling2D' in layer.__class__.__name__ and 'Global' not in layer.__class__.__name__:
        dilation_or_strides = 'strides'
    elif 'Conv2D' in layer.__class__.__name__:
        dilation_or_strides = np.random.choice(['dilation_rate', 'strides'])
    else:
        raise Exception(Cyan('Unprocessed _layer_.__class__'))

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


def _addOneLayer_Conv2D_addConv2DOrPooling(layers, layer, layersNumber, index, layerType, mode, op):

    inputshape = layer.input.shape
    image_d1 = inputshape[1]
    image_d2 = inputshape[2]
    minimage_d1, minimage_d2 = _addOneLayer_Conv2D_addConv2DOrPooling_minSize(layers, layersNumber, index)
    kerpool_size, padding, strides, dilation_rate = _decideConv2DOrPoolingParams(layer, minimage_d1, minimage_d2, image_d1, image_d2)

    return myLayer(layer, modelType='conv2d', definite=False, subType=layerType, mode=mode, op=op, \
        kerpool_size=kerpool_size, padding=padding, strides=strides, dilation_rate=dilation_rate)

def _addOneLayer_Conv2D_addOperation(layers, layer, layersNumber, index, layerType, mode, op):
    # if _convOrPooling(layer):
    if layerType == 1:
        return _addOneLayer_Conv2D_addConv2DOrPooling(layers, layer, layersNumber, index, layerType, mode, op)
    elif layerType == 2:
        return myLayer(layer, modelType='conv2d', definite=False, subType=layerType)
    # TODO
    
def _addOneLayer_Conv2D_analyze_layerType(op):
    LayerType1 = ['Conv2D', 'AveragePooling2D', 'MaxPooling2D', 'SpatialDropout2D']
    LayerType2 = ['Dense']
    LayerType3 = ['Dropout', 'BatchNormalization']
    if op in LayerType1: return 1
    elif op in LayerType2: return 2
    elif op in LayerType3: return 3
    else:
        raise Exception(Cyan(f'Unkown op: {op}'))

def _addOneLayer_Conv2D_decide_layerType_and_head_tail(mode, op):
    
    if mode == 'random':
        layerType = np.random.randint(1, 4)
    elif mode == 'fixed':
        layerType = _addOneLayer_Conv2D_analyze_layerType(op)  
    else:
        raise Exception(Cyan(f'Unkown mode: {mode}'))
    heads, tails = [], []
    if layerType == 1:
        heads, tails = LAYERTYPE1_HEADS, LAYERTYPE1_TAILS
    elif layerType == 2:
        heads, tails = LAYERTYPE2_HEADS, LAYERTYPE2_TAILS
    elif layerType == 3:
        heads, tails = LAYERTYPE3_HEADS, LAYERTYPE3_TAILS
    else:
        raise Exception(Cyan('Unkown '))
    head_tail_id = np.random.randint(0, len(heads))
    head, tail = heads[head_tail_id], tails[head_tail_id]
    return layerType, head, tail

def _addOneLayer_Conv2D(model, mode, op):

    newmodel = keras.Sequential()
    layers = model.layers
    layersNumber = len(layers)
    
    layerType, head, tail = _addOneLayer_Conv2D_decide_layerType_and_head_tail(mode, op)
    insertIndex = np.random.randint(head, tail)

    print(Red(f'insertIndex = {str(insertIndex)}'))

    for i, layer in enumerate(layers):
        if i == 0:
            inputshape = layer.input.shape
            newmodel.add(keras.layers.InputLayer(input_shape=inputshape[1:])) 

        print('i =', i)

        if i < insertIndex:
            mylayer =  myLayer(layer, modelType='conv2d', definite=True)
        elif i == insertIndex:
            print(Red('Artificial layer construction begins.'))
            print(layer)
            mylayer_ = _addOneLayer_Conv2D_addOperation(layers, layer, layersNumber, i, layerType, mode, op)
            newmodel.add(mylayer_)
            # print(Yellow(str(mylayer_.get_config())))
            print(Red('Artificial layer construction finished.'))
            previous_inputshape = mylayer_.output.shape

            # print('previous_inputshape1 =', previous_inputshape)
            mylayer = myLayer(layer, modelType='conv2d', definite=True, inputshape=previous_inputshape)
            # print(Yellow(str(mylayer.get_config())))
        else:
            mylayer = myLayer(layer, modelType='conv2d', definite=True, inputshape=previous_inputshape)

        newmodel.add(mylayer)
        previous_inputshape = mylayer.output.shape
        # print('previous_inputshape2 =', previous_inputshape)
        # for node in mylayer._inbound_nodes:
        #     print(node.inbound_layers)
    return newmodel
