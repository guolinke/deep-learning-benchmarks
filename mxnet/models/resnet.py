"""
Reference:
https://github.com/tao-j/resnet/tree/master/mxnet
"""
import mxnet as mx

featureDim = (3, 224, 224)
labelDim = 1000

def ConvFactory(data, num_filter, kernel, stride, pad, no_bias=False):
    if no_bias == True:
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    else:
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data=bn, act_type='relu')
    return act

def ConvFactory2(data, num_filter, kernel, stride, pad, no_bias=False):
    if no_bias == True:
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    else:
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    return bn

def BottleneckFactory(data, num_filter, layer_idx, project=False):

    # first layer
    layer_idx += 1
    if project:
        proj = ConvFactory2(data=data, num_filter=num_filter*4, kernel=(1, 1), stride=(2, 2), pad=(0, 0), no_bias=True)
        layer1_stride = (2, 2)
    else:
        proj = data
        layer1_stride = (1, 1)
    data = ConvFactory(data, num_filter, kernel=(1, 1), stride=layer1_stride, pad=(0, 0), no_bias=True)

    # second layer
    layer_idx += 1
    data = ConvFactory(data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

    # third layer
    layer_idx += 1
    data = ConvFactory2(data, num_filter=num_filter*4, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
    esum = mx.symbol.ElementWiseSum(proj, data)
    act = mx.symbol.Activation(data=esum, act_type='relu')

    return layer_idx, act


def build_model(model_idx=0):

    layer_idx = 0
    data = mx.symbol.Variable(name='data')
    label = mx.sym.Variable("label")

    # stage conv1_x
    data = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True)

    # setup model parameters
    num_filter = (64, 128, 256, 512)
    model_cfgs = [
        (3, 4, 6, 3),
        (3, 4, 23, 3),
        (3, 8, 36, 3)
    ]
    model_cfg = model_cfgs[model_idx]

    # stage conv2_x to conv5_x, 4 stages
    data = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max')

    # special for conv2_x first block
    stage_filter_size = num_filter[0]
    proj   = ConvFactory2(data=data, num_filter=stage_filter_size*4, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
    block1 = ConvFactory(data=data, num_filter=stage_filter_size, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
    block2 = ConvFactory(data=block1, num_filter=stage_filter_size, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
    block3 = ConvFactory2(data=block2, num_filter=stage_filter_size*4, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
    esum = mx.symbol.ElementWiseSum(proj, block3)
    act = mx.symbol.Activation(esum, act_type='relu')
    necks = act
    layer_idx += 3

    for stage_neck_nums, stage_filter_size, stage_idx in zip(model_cfg, num_filter, range(len(model_cfg))):
        for neck_idx in range(stage_neck_nums):
            if neck_idx == 0 and stage_idx == 0:
                pass
            else:
                if neck_idx == 0 and stage_idx != 0:
                    project = True
                else:
                    project = False
                layer_idx, necks = BottleneckFactory(data=necks, num_filter=stage_filter_size, layer_idx=layer_idx, project=project)

    layer_idx += 1
    avg = mx.symbol.Pooling(data=necks, kernel=(7, 7), stride=(1, 1), name='global_pool', pool_type='avg')
    flatten = mx.sym.Flatten(data=avg, name="flatten")
    fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=labelDim, name='fc0')
    softmax = mx.sym.SoftmaxOutput(data=fc0, label=label)
    return softmax