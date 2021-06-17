import tensorflow.keras as keras
from globalInfos import edge_collection, available_edges_extraction_for4types, extra_info_extraction, config_extraction
from colors import *

config_extraction()
extra_info_extraction()

modelpath = '/home/lisa/origin_model/lenet5-mnist_origin.h5'
keras_model = keras.models.load_model(modelpath)
print(Yellow('keras_model layers'))
for layer in keras_model.layers:
    print(layer.output.shape)
edge_collection(keras_model)
from globalInfos import EDGES
print(Green('EDGES'))
for edge in EDGES:
    print(edge[0].name, edge[1].name)
print('===================================')
available_edges_extraction_for4types()
from globalInfos import CONV2D_TYPE_1_AVAILABLE_EDGES,\
                        CONV2D_TYPE_2_AVAILABLE_EDGES,\
                        CONV2D_TYPE_3_AVAILABLE_EDGES,\
                        CONV2D_TYPE_4_AVAILABLE_EDGES
print(Red('CONV2D_TYPE_1_AVAILABLE_EDGES'))
for edge in CONV2D_TYPE_1_AVAILABLE_EDGES:
    print(edge[0].name, edge[1].name)
print(Red('CONV2D_TYPE_2_AVAILABLE_EDGES'))
for edge in CONV2D_TYPE_2_AVAILABLE_EDGES:
    print(edge[0].name, edge[1].name)
print(Red('CONV2D_TYPE_3_AVAILABLE_EDGES'))
for edge in CONV2D_TYPE_3_AVAILABLE_EDGES:
    print(edge[0].name, edge[1].name)
print(Red('CONV2D_TYPE_4_AVAILABLE_EDGES'))
for edgestuple in CONV2D_TYPE_4_AVAILABLE_EDGES:
    print(Blue('<><><><>'))
    for edge in edgestuple:
        print(edge[0].name, edge[1].name)