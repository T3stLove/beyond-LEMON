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

def _isInputLayer(model, layer, id):
    if model.__class__.__name__ == 'Sequential':
        if id == 0:
            return True
    else:
        if isinstance(layer, keras.layers.InputLayer):
            return True
    return False

def edge_collection(model):
    global EDGES
    layers = model.layers
    EDGES = []
    visited = []
    q = queue.Queue()
    for id, layer in enumerate(layers):
        if _isInputLayer(model, layer, id):
            q.put(layer)
    
    while not q.empty():
        qlayer = q.get()
        if qlayer.name in visited:
            continue
        visited.append(qlayer.name)
        for node in qlayer._outbound_nodes:
            q.put(node.outbound_layer)
            EDGES.append((qlayer, node.outbound_layer))

def _dim4data_bigger(data1, data2):
    if data1[1] > data2[1] or data1[2] > data2[2]:
        return True
    return False

def available_edges_extraction_for4types():
    global EDGES
    if EDGES == []:
        raise Exception(Cyan('Oops! EDGES is empty...'))
    edges4_repo = []
    edges4_output_repo = []
    edges2_repo = []
    edges2_id = 0
    for edge in EDGES:
        inlayer, outlayer = edge
        if isinstance(inlayer.output, list):
            raise Exception(Cyan(f'inlayer {inlayer.name} has more than 1 output'))
        if len(inlayer.output.shape) == 4:
            if outlayer.__class__.__name__ not in CONV2D_TYPE_4_POOL:
                CONV2D_TYPE_1_AVAILABLE_EDGES.append(edge)

            if len(edges4_repo) == 0:
                edges4_repo.append(edge)
                edges4_output_repo.append(inlayer.output)
            else:
                output = edges4_output_repo[0]
                if _dim4data_bigger(output.shape.as_list(), inlayer.output.shape.as_list()):
                    if len(edges4_repo) > 1:
                        CONV2D_TYPE_4_AVAILABLE_EDGES.append(tuple(edges4_repo))
                    edges4_repo.clear()
                    edges4_output_repo.clear()
                else:
                    if output.shape.as_list() != inlayer.output.shape.as_list():
                        raise Exception(Cyan(f'Incorrect relationship between output.shape and inlayer.output\
                            : output.shape.as_list() = {str(output.shape.as_list())} while \
                                inlayer.output.as_list() = {str(inlayer.output.as_list())}'))
                edges4_repo.append(edge)
                edges4_output_repo.append(inlayer.output)

        elif len(inlayer.output.shape) == 2:
            if outlayer.__class__.__name__ not in CONV2D_TYPE_4_POOL:
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


