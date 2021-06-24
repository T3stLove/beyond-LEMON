from colors import *
import configparser
import queue
import tensorflow.keras as keras

DATA_FORMAT = None
DTYPE = None

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
CONV2D_TYPE_4_POOL = []

CONV2D_TYPE_1_AVAILABLE_EDGES = []
CONV2D_TYPE_2_AVAILABLE_EDGES = []
CONV2D_TYPE_3_AVAILABLE_EDGES = []
CONV2D_TYPE_4_AVAILABLE_EDGES = []

EDGES = []
LAYER_CLASS_NAMES = {}
INPUTLAYERS = []
OUTPUTLAYERS = []

CONV2D_TYPE_4_LAYERS = []

def _isInputLayer(model, layer, id):
    if model.__class__.__name__ == 'Sequential':
        if id == 0:
            return True
    else:
        if isinstance(layer, keras.layers.InputLayer):
            return True
    return False

def _isOutputLayer(layer):
    if layer._outbound_nodes == []:
        return True
    return False 

def edge_collection(model):
    global EDGES, INPUTLAYERS, OUTPUTLAYERS
    INPUTLAYERS = []
    OUTPUTLAYERS = []
    layers = model.layers
    EDGES = []
    layer_set = set()
    visited = set()
    CONV2D_TYPE_4_POOL_ALL = ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']
    q = queue.Queue()
    for id, layer in enumerate(layers):
        if _isInputLayer(model, layer, id):
            INPUTLAYERS.append(layer)
            q.put(layer)
        if _isOutputLayer(layer):
            OUTPUTLAYERS.append(layer)
    
    while not q.empty():
        qlayer = q.get()
        layer_set.add(qlayer.name)
        # print(Green(qlayer.name))
        if qlayer.name in CONV2D_TYPE_4_LAYERS:
            CONV2D_TYPE_4_LAYERS.append(qlayer)
        if qlayer.name in visited:
            continue
        visited.add(qlayer.name)
        for node in qlayer._outbound_nodes:
            layer_out = node.outbound_layer
            if layer_out.__class__.__name__ not in CONV2D_TYPE_4_POOL_ALL:
                q.put(layer_out)
                EDGES.append((qlayer, layer_out))
            else:
                inputlayer_alltraversed = True
                for node in layer_out._inbound_nodes:
                    if isinstance(node.inbound_layers, list):
                        for l in node.inbound_layers:
                            if l.name not in layer_set:
                                inputlayer_alltraversed = False
                                break
                    else:
                        l = node.inbound_layers
                        if l.name not in layer_set:
                            inputlayer_alltraversed = False
                            break
                if inputlayer_alltraversed:
                    q.put(layer_out)
                    EDGES.append((qlayer, layer_out))
    print(EDGES == [])
def _dim4data_bigger(data1, data2):
    if data1[1] > data2[1] or data1[2] > data2[2]:
        return True
    return False

def _dim4data_equal(data1, data2):
    if data1[1] != data2[1] or data1[2] != data2[2] or data1[3] != data2[3]:
        return False
    return True

# '''
# type_4 layers have strict requirements on the consistency of input shapes
# So we should first weed out the layers that we cannot insert 4-dimensional
# shape-changed layers, such as Conv2d, to avoid the situation where
# the resultant input shapes are inconsistent
# '''
# def _type4_influence():
#     # layers before which we cannot insert 4-dimensional shape-changed layers
#     forbid2_layers = set()
#     visited = set()
#     from queue import Queue
#     q = Queue()

#     for layer in CONV2D_TYPE_4_LAYERS:
#         q.put(layer)
#     while not q.empty():
#         layer = q.get()
#         forbid2_layers.add(layer)
#         for node in layer._inbound_nodes:
#             for l in node.inbound_layers:
#                 if l in visited:
#                     continue
#                 visited.add(l)
#                 if l.__class__.__name__ not in ['Conv2D', 'AveragePooling2D', \
#                                             'MaxPooling2D', 'SeparableConv2D', \
#                                                 'ZeroPadding2D', 'Cropping2D']:
#                     q.put(l)

def _certificate_for_adding_4_dimensional_shape_changed_layers():
    from queue import Queue
    q = Queue()
    for layer in OUTPUTLAYERS:
        q.put((layer, False))
    certificate = set()
    visited = set()
    while not q.empty():
        layer, cert = q.get()
        if layer.name in visited:
            continue
        visited.add(layer.name)
        if cert:
            certificate.add(layer.name)
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                for l in node.inbound_layers:
                    if cert:
                        if l.__class__.__name__ in ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']:
                            q.put((l, False))
                        else:
                            q.put((l, True))
                    else:
                        if l.__class__.__name__ in ['Conv2D', 'SeparableConv2D', 'DepthwiseConv2D']:
                            q.put((l, True))
                        else:
                            q.put((l, False))
            else:
                l = node.inbound_layers
                if cert:
                    if l.__class__.__name__ in ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']:
                        q.put((l, False))
                    else:
                        q.put((l, True))
                else:
                    if l.__class__.__name__ in ['Conv2D', 'SeparableConv2D', 'DepthwiseConv2D']:
                        q.put((l, True))
                    else:
                        q.put((l, False))
    return certificate

def inlayer_not_exists(edges4_repo, inlayer):
    if edges4_repo == []:
        return True
    for edge in edges4_repo:
        if edge[0].name == inlayer.name:
            return False

    return True

def available_edges_extraction_for4types():

    certificate = _certificate_for_adding_4_dimensional_shape_changed_layers()

    global EDGES, CONV2D_TYPE_1_AVAILABLE_EDGES, CONV2D_TYPE_2_AVAILABLE_EDGES,\
           CONV2D_TYPE_3_AVAILABLE_EDGES, CONV2D_TYPE_4_AVAILABLE_EDGES
    CONV2D_TYPE_1_AVAILABLE_EDGES = []
    CONV2D_TYPE_2_AVAILABLE_EDGES = []
    CONV2D_TYPE_3_AVAILABLE_EDGES = []
    CONV2D_TYPE_4_AVAILABLE_EDGES = []

    if EDGES == []:
        raise Exception(Cyan('Oops! EDGES is empty...'))
    edges4_repo = []
    edges4_output_repo = []
    edges2_repo = []
    edges2_id = 0
    for edge in EDGES:
        inlayer, outlayer = edge
        if isinstance(inlayer, keras.layers.InputLayer):
            continue
        if outlayer.__class__.__name__ in ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']:
            continue
        if isinstance(inlayer.output, list):
            raise Exception(Cyan(f'inlayer {inlayer.name} has more than 1 output'))
        if len(outlayer.output.shape) == 4 and outlayer.name in certificate:
            CONV2D_TYPE_1_AVAILABLE_EDGES.append(edge)
            
            if len(edges4_repo) == 0:
                edges4_repo.append(edge)
                edges4_output_repo.append(inlayer)
            else:
                output = edges4_output_repo[0].output
                if not _dim4data_equal(output.shape.as_list(), inlayer.output.shape.as_list()):
                    if len(edges4_repo) > 1:
                        CONV2D_TYPE_4_AVAILABLE_EDGES.append(tuple(edges4_repo))
                    edges4_repo.clear()
                    edges4_output_repo.clear()
                # else:
                #     if output.shape.as_list()[1:3] != inlayer.output.shape.as_list()[1:3]:
                #         raise Exception(Cyan(f'Incorrect relationship between {str(edges4_output_repo[0].name)}.output.shape and {str(inlayer.name)}.output.shape: output.shape.as_list() = {str(output.shape.as_list())} while inlayer.output.shape.as_list() = {str(inlayer.output.shape.as_list())}'))
                if inlayer_not_exists(edges4_repo, inlayer):
                    edges4_repo.append(edge)
                    edges4_output_repo.append(inlayer)

        elif len(inlayer.output.shape) == 2:
            if outlayer.__class__.__name__ not in ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']:
                CONV2D_TYPE_2_AVAILABLE_EDGES.append(edge)        
            edges2_repo.append((edge, inlayer.output.shape[1], edges2_id))
            edges2_id += 1
        
        CONV2D_TYPE_3_AVAILABLE_EDGES.append(edge)

    if len(edges4_repo) > 1:
        CONV2D_TYPE_4_AVAILABLE_EDGES.append(tuple(edges4_repo))
    edges2_repo.sort(key=lambda x:(x[1], x[2]))
    v_, tmp_edges = -1, []
    for edge_, value_, id_ in edges2_repo:
        if v_ != value_:
            if len(tmp_edges) > 1:
                CONV2D_TYPE_4_AVAILABLE_EDGES.append(tuple(tmp_edges))
            tmp_edges.clear()
            tmp_edges.append(edge_)
            v_ = value_
        else:
            tmp_edges.append(edge_)

    if len(tmp_edges) > 1:
        CONV2D_TYPE_4_AVAILABLE_EDGES.append(tuple(tmp_edges))

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
    global CONV2D_TYPE_1_POOL, CONV2D_TYPE_2_POOL, CONV2D_TYPE_3_POOL, CONV2D_TYPE_4_POOL
    for op in OPSPOOL:
        if op in ['Conv2D', 'SeparableConv2D', 'AveragePooling2D', 'MaxPooling2D',\
            'SpatialDropout2D', 'SeparableConv2D', 'ZeroPadding2D', 'Cropping2D']:
            CONV2D_TYPE_1_POOL.append(op)
        elif op == 'Dense':
            CONV2D_TYPE_2_POOL.append(op)
        elif op in ['Dropout', 'BatchNormalization', 'LayerNormalization', 'GaussianDropout']:
            CONV2D_TYPE_3_POOL.append(op)
        elif op in ['Add', 'Concatenate', 'Average', 'Maximum', 'Minimum', 'Subtract', 'Multiply', 'Dot']:
            CONV2D_TYPE_4_POOL.append(op)
        else:
            raise Exception(Cyan(f'Unkown op: {op}'))


