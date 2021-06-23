import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.merge import minimum
from colors import *
from newLayer_handler import myLayer
from collections import namedtuple
import globalInfos

def addOneLayer(model, mode='random', op = None):
    return _addOneLayer_Conv2D(model, mode, op)

def _addOneLayer_Conv2D_minSize(nextLayer):
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
    if kerpool_size_d1 == 1:
        dilation_rate_1 = 1
    else:
        dilation_rate_1 = (image_d1-output_d1+1-kerpool_size_d1)//(kerpool_size_d1-1)+1

    kerpool_size_d2s = []
    for ks_d2 in range(minikerpool_size[1], maxikerpool_size[1]+1):
        if ks_d2 == 1:
            kerpool_size_d2s.append(ks_d2)
            continue
        if (image_d2-output_d2+1-ks_d2)%(ks_d2-1) == 0:
            kerpool_size_d2s.append(ks_d2)

    kerpool_size_d2 = np.random.choice(kerpool_size_d2s)
    if kerpool_size_d2 == 1:
        dilation_rate_2 = 1
    else:
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


def _addOneLayer_Conv2D_addConv2DOrPooling(layer, layerType, mode, op):
    print(layer.name)
    inputshape = layer.input.shape
    image_d1 = inputshape[1]
    image_d2 = inputshape[2]
    minimage_d1, minimage_d2 = _addOneLayer_Conv2D_minSize(layer)
    kerpool_size, padding, strides, dilation_rate = _decideConv2DOrPoolingParams(op, minimage_d1, minimage_d2, image_d1, image_d2)

    return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op, \
        kerpool_size=kerpool_size, padding=padding, strides=strides, dilation_rate=dilation_rate)

def _addOneLayer_Conv2D_addZeroPadding2D(layer, layerType, mode, op):
    frn = [
        np.random.randint(1, 11),
        np.random.randint(1, 11),
        np.random.randint(1, 11),
        np.random.randint(1, 11)
    ]
    return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op,\
        padding=((frn[0], frn[1]), (frn[2], frn[3])))

def _addOneLayer_Conv2D_addCropping2D(layer, layerType, mode, op):
    minimage_d1, minimage_d2 = _addOneLayer_Conv2D_minSize(layer)
    inputshape = layer.input.shape
    image_d1 = inputshape[1]
    image_d2 = inputshape[2]
    output_d1 = minimage_d1 if minimage_d1 == image_d1 else np.random.randint(minimage_d1, image_d1)
    output_d2 = minimage_d1 if minimage_d2 == image_d2 else np.random.randint(minimage_d2, image_d2)
    diff_d1 = image_d1 - output_d1
    diff_d2 = image_d2 - output_d2
    top_crop = diff_d1 // 2
    bottom_crop = diff_d1 - top_crop
    left_crop = diff_d2 // 2
    right_crop = diff_d2 - left_crop
    frn = [
        np.random.randint(0, top_crop+1),
        np.random.randint(0, bottom_crop+1),
        np.random.randint(0, left_crop+1),
        np.random.randint(0, right_crop+1)
    ]
    return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op,\
        cropping=((frn[0], frn[1]), (frn[2], frn[3])))

def _addOneLayer_Conv2D_addOperation(layer, layerType, mode, op, inlayer):
    if layerType == 1:
        if mode == 'fixed':
           pass
        elif mode == 'random':    
            from globalInfos import CONV2D_TYPE_1_POOL
            op = np.random.choice(CONV2D_TYPE_1_POOL)
        else:
            raise Exception(Cyan(f'Unkown mode: {mode}'))
        # UPDATE
        if op != 'SpatialDropout2D' and op != 'ZeroPadding2D' and op != 'Cropping2D':
            return _addOneLayer_Conv2D_addConv2DOrPooling(layer, layerType, mode, op)
        else:
            if op == 'ZeroPadding2D':
                return _addOneLayer_Conv2D_addZeroPadding2D(layer, layerType, mode, op)
            elif op == 'Cropping2D':
                return _addOneLayer_Conv2D_addCropping2D(layer, layerType, mode, op)
            else:
                return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op)

    elif layerType == 2 or layerType == 3 or layerType == 4:

        if layerType == 2:
            if mode == 'fixed':
                pass
            elif mode == 'random':    
                from globalInfos import CONV2D_TYPE_2_POOL
                op = np.random.choice(CONV2D_TYPE_2_POOL)
            else:
                raise Exception(Cyan(f'Unkown mode: {mode}'))
        
        elif layerType == 3:
            if mode == 'fixed':
                pass
            elif mode == 'random':    
                from globalInfos import CONV2D_TYPE_3_POOL
                op = np.random.choice(CONV2D_TYPE_3_POOL)
            else:
                raise Exception(Cyan(f'Unkown mode: {mode}'))
        
        elif layerType == 4:
            if mode == 'fixed':
                pass
            elif mode == 'random':    
                from globalInfos import CONV2D_TYPE_4_POOL
                op = np.random.choice(CONV2D_TYPE_4_POOL)
            else:
                raise Exception(Cyan(f'Unkown mode: {mode}'))
        if op == 'BatchNormalization' or op == 'LayerNormalization':
            return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op, inlayer=inlayer)
        return myLayer(layer, modelType='conv2d', copy=False, subType=layerType, mode=mode, op=op)
    else:
        raise Exception(Cyan(f'Unkown layerType: {str(layerType)}'))
    
def _addOneLayer_Conv2D_analyze_layerType(op):

    from globalInfos import CONV2D_TYPE_1_POOL, CONV2D_TYPE_2_POOL, CONV2D_TYPE_3_POOL, CONV2D_TYPE_4_POOL
    if op in CONV2D_TYPE_1_POOL: return 1
    elif op in CONV2D_TYPE_2_POOL: return 2
    elif op in CONV2D_TYPE_3_POOL: return 3
    elif op in CONV2D_TYPE_4_POOL: return 4
    else:
        raise Exception(Cyan(f'Unkown op: {op}'))

def _addOneLayer_Conv2D_decide_layerType(mode, op):
    
    if mode == 'random':
        layerTypes = []
        from globalInfos import CONV2D_TYPE_1_POOL, CONV2D_TYPE_2_POOL, CONV2D_TYPE_3_POOL, CONV2D_TYPE_4_POOL
        if CONV2D_TYPE_1_POOL: layerTypes.append(1)
        if CONV2D_TYPE_2_POOL: layerTypes.append(2)
        if CONV2D_TYPE_3_POOL: layerTypes.append(3)
        if CONV2D_TYPE_4_POOL: layerTypes.append(4)
        layerType = np.random.choice(layerTypes)
    elif mode == 'fixed':
        layerType = _addOneLayer_Conv2D_analyze_layerType(op)  
        # print('op = ', op, ' layerType = ', layerType)
    else:
        raise Exception(Cyan(f'Unkown mode: {mode}'))
    return layerType

def _addOneLayer_Conv2D_decide_inLayers_outLayer_crop_edge(layerType):
    from globalInfos import CONV2D_TYPE_1_AVAILABLE_EDGES,\
                            CONV2D_TYPE_2_AVAILABLE_EDGES,\
                            CONV2D_TYPE_3_AVAILABLE_EDGES,\
                            CONV2D_TYPE_4_AVAILABLE_EDGES
    available_edges = [CONV2D_TYPE_1_AVAILABLE_EDGES,\
                       CONV2D_TYPE_2_AVAILABLE_EDGES,\
                       CONV2D_TYPE_3_AVAILABLE_EDGES,\
                       CONV2D_TYPE_4_AVAILABLE_EDGES][layerType-1]

    randID = np.random.randint(0, len(available_edges))
    randID = 4
    print(Blue(f'randID = {str(randID)}'))
    selected_edges = available_edges[randID]
    print('layerType:', layerType)
    if layerType != 4:
        inLayers = [selected_edges[0]]
        outLayer = selected_edges[1]
        # print('=================',outLayer.input)
        crop_edge = selected_edges
    else:
        inLayers = []
        for edge in selected_edges:
            inLayers.append(edge[0])
        outLayer = selected_edges[-1][1]
        crop_edge = selected_edges[-1]
    return inLayers, outLayer, crop_edge

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

def _in_inLayers(qlayer, inLayers):
    for inlayer in inLayers:
        if qlayer.name == inlayer.name:
            return True
    return False

def _addOneLayer_Conv2D_init_q_inputs():
    from globalInfos import INPUTLAYERS
    import queue
    q = queue.Queue()
    inputs = []
    for inputlayer in INPUTLAYERS:
        input = keras.Input(shape=inputlayer.input.shape[1:])
        inputs.append(input)

        if not isinstance(inputlayer, keras.layers.InputLayer):
            newlayer = inputlayer.__class__.from_config(inputlayer.get_config())
            q.put((inputlayer, newlayer(input)))
        else:
            for node in inputlayer._outbound_nodes:
                out_layer = node.outbound_layer
                newlayer = out_layer.__class__.from_config(out_layer.get_config())
                q.put((out_layer, newlayer(input)))

    return inputs, q

def _addOneLayer_Conv2D(model, mode, op):
    layerType = _addOneLayer_Conv2D_decide_layerType(mode, op)
    inLayers, outLayer, crop_edge = _addOneLayer_Conv2D_decide_inLayers_outLayer_crop_edge(layerType)
    print('outLayer.name:', outLayer.name)
    print('inLayers: ', len(inLayers))
    for inlayer in inLayers:
        print('>>>', inlayer.name)
    newlayer = _addOneLayer_Conv2D_addOperation(outLayer, layerType, mode, op, inLayers[0])
    
    decide_data_form(model.layers)
    decide_data_type(model.layers)
    visited = {}
    invalues = [] if len(inLayers) > 1 else None
    outputs = []
    outputs_dict = {}
    CONV2D_TYPE_4_POOL_ALL = ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']
    inputs, q = _addOneLayer_Conv2D_init_q_inputs()

    while not q.empty():
        qlayer, output = q.get()
        print(Magenta(f'qlayer.name: {qlayer.name}'))
        # print(Magenta(str(output.shape)))
        outputs_dict[qlayer.name] = output
        if qlayer.name in visited:
            continue
        visited[qlayer.name] = True
        if qlayer._outbound_nodes == []:
            outputs.append(output)
        if _in_inLayers(qlayer, inLayers):
            if isinstance(invalues, list): 
                invalues.append(output)
            else: 
                invalues = output
        for node in qlayer._outbound_nodes:
            layer_out = node.outbound_layer
            print(Red(f'layer_out.name: {layer_out.name}'))
            if layer_out.name == outLayer.name:
                print(Green('yes'))
                print('len(invalues): ', len(invalues))
                output = newlayer(invalues)
                layer_out_ = myLayer(layer_out, modelType='conv2d', copy=True, inputshape=output.shape)
            else:

                if layer_out.__class__.__name__  not in ['Conv2D', 'AveragePooling2D', 'MaxPooling2D', 'SeparableConv2D', 'DepthwiseConv2D'] or \
                    output.shape == layer_out.input.shape:
                    layer_out_ = layer_out.__class__.from_config(layer_out.get_config())
                    if layer_out.get_weights() != []:
                        layer_out_.build(layer_out.input.shape)
                        layer_out_.set_weights(layer_out.get_weights())
                else:
                    layer_out_ = myLayer(layer_out, modelType='conv2d', copy=True, inputshape=output.shape) 
            # print(layer_out.__class__.__name__, layer_out.__class__.__name__ in CONV2D_TYPE_4_POOL_ALL)
            if layer_out.__class__.__name__ in CONV2D_TYPE_4_POOL_ALL:
                tmp_inputs = []
                inputlayer_alltraversed = True
                for node in layer_out._inbound_nodes:
                    for l in node.inbound_layers:
                        if l.name != qlayer.name:
                            if l.name in outputs_dict:
                                tmp_inputs.append(outputs_dict[l.name])
                            else:
                                inputlayer_alltraversed = False
                if inputlayer_alltraversed:
                    tmp_inputs.append(output)
                    q.put((layer_out, layer_out_(tmp_inputs)))
            else:
                # print(Yellow(str(output.shape)))
                q.put((layer_out, layer_out_(output)))
    # print(inputs, outputs)
    newmodel = keras.Model(inputs=inputs, outputs=outputs)
    return newmodel
