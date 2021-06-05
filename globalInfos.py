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

LAYERTYPE1_HEADS = []
LAYERTYPE1_TAILS = []

LAYERTYPE2_HEADS = []
LAYERTYPE2_TAILS = []

LAYERTYPE3_HEADS = []
LAYERTYPE3_TAILS = []

MODELNAMES = None
MOD = None
ORIGIN_PATH = None
MUTANT_PATH = None
ORDERS = None
OPSPOOL = None
TOTALNUMBER = None
EACHNUMBER = None

def config_extraction():
    global MODELNAMES, MOD, ORIGIN_PATH, MUTANT_PATH, ORDERS, OPSPOOL, TOTALNUMBER, EACHNUMBER
    parser = configparser.ConfigParser()
    parser.read('config.conf')
    params = parser['params']
    MODELNAMES = params['models'].split('\n')
    MOD = params['mod']
    ORIGIN_PATH = params['origin_path']
    MUTANT_PATH = params['mutant_path']
    ORDERS = params['orders'].split('\n')
    OPSPOOL = params['opspool'].split('\n')
    TOTALNUMBER = parser['random']['totalNumber']
    EACHNUMBER = parser['fixed']['eachNumber']


def _type1_heads_tails_extraction(layers):
    global LAYERTYPE1_HEADS, LAYERTYPE1_TAILS
    on = False
    for i in range(len(layers)):
        if i == 0:
            on = True
            LAYERTYPE1_HEADS.append(i)
        else:
            preLayer = layers[i-1]
            if len(preLayer.output.shape) == 4:
                if not on:
                    on = True
                    LAYERTYPE1_HEADS.append(i)
            elif len(preLayer.output.shape) == 2:
                if on:
                    on = False
                    LAYERTYPE1_TAILS.append(i+1)
    if LAYERTYPE1_TAILS == []:
        raise Exception(Cyan('tails == [] -> Cannot find flatten layer in Conv2D model.'))
    if len(LAYERTYPE1_TAILS) != len(LAYERTYPE1_HEADS):
        raise Exception(Cyan(f'heads length: {str(len(LAYERTYPE1_HEADS))}, tails length: {str(len(LAYERTYPE1_TAILS))}, they are inconsistent'))

def _type2_heads_tails_extraction(layers):
    global LAYERTYPE2_HEADS, LAYERTYPE2_TAILS
    on = False
    for i in range(len(layers)):
        if i == 0:
            on = True
            LAYERTYPE2_HEADS.append(i)
        else:
            preLayer = layers[i-1]
            if len(preLayer.output.shape) == 2:
                if not on:
                    on = True
                    LAYERTYPE2_HEADS.append(i)
            elif len(preLayer.output.shape) == 4:
                if on:
                    on = False
                    LAYERTYPE2_TAILS.append(i+1)
    if LAYERTYPE2_TAILS == []:
        raise Exception(Cyan('tails == [] -> Cannot find flatten layer in Conv2D model.'))
    if len(LAYERTYPE2_TAILS) != len(LAYERTYPE2_HEADS):
        raise Exception(Cyan(f'heads length: {str(len(LAYERTYPE2_HEADS))}, tails length: {str(len(LAYERTYPE2_TAILS))}, they are inconsistent'))

def _type3_heads_tails_extraction(layersNumber):
    global LAYERTYPE3_HEADS, LAYERTYPE3_TAILS
    LAYERTYPE3_HEADS.append(0)
    LAYERTYPE3_TAILS.append(layersNumber)

def type123_heads_tails_extraction(layers, layersNumber):
    _type1_heads_tails_extraction(layers)
    _type2_heads_tails_extraction(layers)
    _type3_heads_tails_extraction(layersNumber)