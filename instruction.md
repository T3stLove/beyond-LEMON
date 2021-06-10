## How to add new layers for 2-dimensional image related model?
1. Add detailed implementation in `newLayer_impl.py`

   To perform layer addition, we provide two modes for all layers prepared: `definite` and `indefinite`
   except for: `Flatten`, `GlobalMaxPooling2D`, `GlobalAveragePooling2D` and all layers that work on
   changing **dimension**
   
   + definite: this added layer is a copy of the existing model
   + indefinite: this added layer is completely designed from scratch 

2. Add the layer name to `opspool` in `config.conf`
3. supplement function `extra_info_extraction` in `globalInfos.py` by pointing out which type the added layer belongs to
   
   + type1: can only handle 4-dimensional data
   + type2: can only handle 2-dimensional data
   + type3: can handle both the 4-dimensional and the 2-dimensional

4. For layers of type1: you should pay attention to `_decideConv2DOrPoolingParams` in `mutators.py`, which introduces some details about
   deciding range of `kernel_size`, `pool_size`, etc, which may change data shape flowing through the added layer.
   We need to add layer class name to the following code snippet if our new layer is related to `Conv2D` or `Pooling`
   ```Python
    if op == 'AveragePooling2D' or op == 'MaxPooling2D':
        dilation_or_strides = 'strides'
    elif op == 'Conv2D' or op == 'SeparableConv2D':
        dilation_or_strides = np.random.choice(['dilation_rate', 'strides'])
    else:
        raise Exception(Cyan(f'Unexpected op: {op}'))
   ``` 

5. Supplement `_myConv2DLayer_op_to_newlayer` and `_myConv2DLayer_definite` in `newLayer_handler.py`

6. For `_myConv2DLayer_indefinite_1`, `_myConv2DLayer_indefinite_2` and `_myConv2DLayer_indefinite_3`, we should
    pay attention to whether we pass correct number of parameters in the correct order