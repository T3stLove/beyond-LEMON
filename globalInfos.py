from colors import *
import configparser

'''
@haoyang
Reconstruct the logic of generating insertIndex:
now we can insert conv2d and pooling(not global) 
layer before any layers for which the output shape 
of the previous layer  is 4-demensional.

For instance,
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_9 (Conv2D)            (None, 28, 28, 6)         156
_________________________________________________________________
average_pooling2d_9 (Average (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 10, 10, 16)        2416
_________________________________________________________________
average_pooling2d_10 (Averag (None, 5, 5, 16)          0
_________________________________________________________________
flatten_5 (Flatten)          (None, 400)               0
_________________________________________________________________
dense_13 (Dense)             (None, 120)               48120
_________________________________________________________________
dropout_5 (Dropout)          (None, 120)               0
_________________________________________________________________
dense_14 (Dense)             (None, 84)                10164
_________________________________________________________________
dense_15 (Dense)             (None, 10)                850
=================================================================
In this model, we can insert conv2d and pooling before flatten_5

layerType:
> 1: conv2d
    pooling(not global)
    SpatialDropout2D
only handles the situation where the input shape is 4-dimensional
> 2: dense
only for 2-dimensional input shape
> 3: dropout layer and batchnormalization
without shape limitation

'''
DATA_FORMAT = None
DTYPE = None

LAYER_NAME = {}

CONV2D_LAYERTYPE1_HEADS = []
CONV2D_LAYERTYPE1_TAILS = []

CONV2D_LAYERTYPE2_HEADS = []
CONV2D_LAYERTYPE2_TAILS = []

CONV2D_LAYERTYPE3_HEADS = []
CONV2D_LAYERTYPE3_TAILS = []

MODELNAMES = None
MODE = None
ORIGIN_PATH = None
MUTANT_PATH = None
ORDERS = None
OPSPOOL = None
TOTALNUMBER = None
EACHNUMBER = None

CONV2D_TYPE_1_POOL = []
CONV2D_TYPE_2_POOL = []
CONV2D_TYPE_3_POOL = []

def config_extraction():
    global MODELNAMES, MODE, ORIGIN_PATH, MUTANT_PATH, ORDERS, OPSPOOL, TOTALNUMBER, EACHNUMBER
    parser = configparser.ConfigParser()
    parser.read('./config.conf')
    params = parser['params']
    MODELNAMES = params['models'].split('\n')
    MODE = params['mode']
    ORIGIN_PATH = params['origin_path']
    MUTANT_PATH = params['mutant_path']
    ORDERS = params['orders'].split('\n')
    OPSPOOL = params['opspool'].split('\n')
    TOTALNUMBER = parser['random']['totalNumber']
    EACHNUMBER = parser['fixed']['eachNumber']

def extra_info_extraction():
    for op in OPSPOOL:
        if op == 'Conv2D' or op == 'SeparableConv2D' or op == 'AveragePooling2D' or op == 'MaxPooling2D' \
            or op == 'SpatialDropout2D':
            CONV2D_TYPE_1_POOL.append(op)
        elif op == 'Dense':
            CONV2D_TYPE_2_POOL.append(op)
        elif op == 'Dropout' or op == 'BatchNormalization' or op == 'LayerNormalization':
            CONV2D_TYPE_3_POOL.append(op)
        else:
            raise Exception(Cyan(f'Unkown op: {op}'))

def _type1_heads_tails_extraction(layers):
    global CONV2D_LAYERTYPE1_HEADS, CONV2D_LAYERTYPE1_TAILS
    on = False
    for i in range(len(layers)):
        if i == 0:
            on = True
            CONV2D_LAYERTYPE1_HEADS.append(i)
        else:
            preLayer = layers[i-1]
            if len(preLayer.output.shape) == 4:
                if not on:
                    on = True
                    CONV2D_LAYERTYPE1_HEADS.append(i)
            elif len(preLayer.output.shape) == 2:
                if on:
                    on = False
                    CONV2D_LAYERTYPE1_TAILS.append(i)
    print(Red('1'), CONV2D_LAYERTYPE1_HEADS, CONV2D_LAYERTYPE1_TAILS)
    if CONV2D_LAYERTYPE1_TAILS == []:
        raise Exception(Cyan('tails == []'))
    if len(CONV2D_LAYERTYPE1_TAILS) != len(CONV2D_LAYERTYPE1_HEADS):
        raise Exception(Cyan(f'heads length: {str(len(CONV2D_LAYERTYPE1_HEADS))}, tails length: {str(len(CONV2D_LAYERTYPE1_TAILS))}, they are inconsistent'))

def _type2_heads_tails_extraction(layers):
    global CONV2D_LAYERTYPE2_HEADS, CONV2D_LAYERTYPE2_TAILS
    on = False
    for i in range(len(layers)):
        if i == 0:
            pass
        else:
            preLayer = layers[i-1]
            if len(preLayer.output.shape) == 2:
                if not on:
                    on = True
                    CONV2D_LAYERTYPE2_HEADS.append(i)
            elif len(preLayer.output.shape) == 4:
                if on:
                    on = False
                    CONV2D_LAYERTYPE2_TAILS.append(i)
    CONV2D_LAYERTYPE2_TAILS.append(len(layers))
    print(Red('2'), CONV2D_LAYERTYPE2_HEADS, CONV2D_LAYERTYPE2_TAILS)
    if CONV2D_LAYERTYPE2_TAILS == []:
        raise Exception(Cyan('tails == []'))
    if len(CONV2D_LAYERTYPE2_TAILS) != len(CONV2D_LAYERTYPE2_HEADS):
        raise Exception(Cyan(f'heads length: {str(len(CONV2D_LAYERTYPE2_HEADS))}, tails length: {str(len(CONV2D_LAYERTYPE2_TAILS))}, they are inconsistent'))

def _type3_heads_tails_extraction(layersNumber):
    global CONV2D_LAYERTYPE3_HEADS, CONV2D_LAYERTYPE3_TAILS
    CONV2D_LAYERTYPE3_HEADS.append(0)
    CONV2D_LAYERTYPE3_TAILS.append(layersNumber)

def type123_heads_tails_extraction(layers, layersNumber):
    _type1_heads_tails_extraction(layers)
    _type2_heads_tails_extraction(layers)
    _type3_heads_tails_extraction(layersNumber)