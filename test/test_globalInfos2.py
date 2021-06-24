import tensorflow.keras as keras
from globalInfos import edge_collection, _certificate_for_adding_4_dimensional_shape_changed_layers
from colors import *

def _certificate_for_adding_4_dimensional_shape_changed_layers_offline(modelname):
    if modelname == 'alexnet-cifar10':
        return {'conv2d_5', 'batch_normalization_2', 'conv2d_1', 'batch_normalization_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_3', 'max_pooling2d_2', 'conv2d_1_input', 'conv2d_3', 'conv2d_4'}
    elif modelname == 'lenet5-mnist':
        return {'conv2d_10', 'conv2d_9_input', 'average_pooling2d_10', 'conv2d_9', 'average_pooling2d_9'}
    elif modelname == 'resnet50-imagenet':
        return {'activation_37', 'bn4e_branch2a', 'activation_28', 'res4b_branch2b', 'res4f_branch2a', 'activation_42', 'res4e_branch2b', 'activation_24', 'activation_1', 'res4b_branch2c', 'bn3c_branch2a', 'bn3d_branch2b', 'bn4c_branch2a', 'bn4d_branch2b', 'activation_20', 'activation_15', 'res5b_branch2b', 'res2a_branch1', 'res3a_branch2b', 'bn2a_branch2b', 'res3c_branch2b', 'res2a_branch2b', 'res3b_branch2a', 'bn4d_branch2a', 'bn4a_branch2b', 'activation_6', 'res2c_branch2c', 'res4d_branch2c', 'res5a_branch2b', 'activation_38', 'bn3a_branch2b', 'res4f_branch2b', 'activation_26', 'activation_21', 'activation_10', 'res2b_branch2a', 'res5a_branch2c', 'activation_18', 'res2b_branch2c', 'res4d_branch2b', 'activation_30', 'activation_44', 'bn4b_branch2a', 'res3d_branch2c', 'activation_29', 'bn3b_branch2a', 'res2b_branch2b', 'bn5a_branch2a', 'bn5a_branch2b', 'bn2c_branch2a', 'res5a_branch2a', 'res3c_branch2c', 'bn2a_branch2a', 'activation_46', 'res3b_branch2c', 'activation_17', 'res4a_branch1', 'bn5b_branch2b', 'activation_33', 'activation_12', 'activation_47', 'res2a_branch2c', 'res4c_branch2c', 'res4f_branch2c', 'res2c_branch2a', 'res3d_branch2b', 'res5a_branch1', 'activation_5', 'bn5c_branch2b', 'res4c_branch2a', 'bn5c_branch2a', 'res5b_branch2c', 'activation_43', 'activation_16', 'res4e_branch2a', 'res4c_branch2b', 'activation_3', 'activation_4', 'res4d_branch2a', 'activation_2', 'max_pooling2d_1', 'activation_27', 'activation_31', 'input_1', 'bn4e_branch2b', 'bn3c_branch2b', 'activation_48', 'activation_45', 'bn2b_branch2a', 'activation_14', 'activation_7', 'bn4f_branch2a', 'activation_34', 'bn5b_branch2a', 'activation_35', 'bn3d_branch2a', 'res5c_branch2a', 'activation_22', 'activation_25', 'res3a_branch1', 'res4b_branch2a', 'pool1_pad', 'conv1_pad', 'activation_19', 'res3c_branch2a', 'bn3b_branch2b', 'res4a_branch2b', 'res5c_branch2b', 'res5b_branch2a', 'bn4a_branch2a', 'activation_9', 'activation_13', 'bn4f_branch2b', 'activation_41', 'res3a_branch2a', 'activation_11', 'res2a_branch2a', 'res5c_branch2c', 'res2c_branch2b', 'activation_39', 'bn4b_branch2b', 'bn2b_branch2b', 'res3b_branch2b', 'conv1', 'res3a_branch2c', 'activation_32', 'bn3a_branch2a', 'res3d_branch2a', 'activation_8', 'bn_conv1', 'activation_36', 'activation_23', 'bn2c_branch2b', 'bn4c_branch2b', 'activation_40', 'res4e_branch2c', 'res4a_branch2a', 'res4a_branch2c'}
    elif modelname == 'vgg16-imagenet':
        return {'block3_conv3', 'block5_pool', 'block4_conv2', 'block5_conv1', 'input_4', 'block5_conv3', 'block3_conv2', 'block2_conv1', 'block1_conv1', 'block4_pool', 'block4_conv1', 'block2_conv2', 'block5_conv2', 'block3_conv1', 'block3_pool', 'block1_pool', 'block2_pool', 'block1_conv2', 'block4_conv3'}
    elif modelname == 'inception.v3-imagenet':
        pass

modelpath = '/home/lisa/origin_model/inception.v3-imagenet_origin.h5'
model = keras.models.load_model(modelpath)
edge_collection(model)
_certificate_for_adding_4_dimensional_shape_changed_layers()
