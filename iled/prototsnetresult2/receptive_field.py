import math
from collections import namedtuple

import torch

ReceptiveFieldInfo = namedtuple('ReceptiveFieldInfo', ['conv_ts_len', 'size', 'center', 'jump'])


def compute_rf_prototype(ts_len, prototype_patch_index, protoL_rf_info):
    ts_idx = prototype_patch_index[0]
    fmap_proto_idx = prototype_patch_index[1]
    rf_indices = compute_rf_protoL_at_spatial_location(ts_len,
                                                       fmap_proto_idx,
                                                       protoL_rf_info)
    return [ts_idx, rf_indices[0], rf_indices[1]]


def compute_rf_protoL_at_spatial_location(ts_len, fmap_proto_idx, protoL_rf_info: ReceptiveFieldInfo):
    assert(fmap_proto_idx < protoL_rf_info.conv_ts_len)

    center = protoL_rf_info.center + (fmap_proto_idx * protoL_rf_info.jump)

    rf_start_idx = max(int(center - (protoL_rf_info.size/2)), 0)
    rf_end_idx = min(int(center + (protoL_rf_info.size/2)), ts_len)

    return [rf_start_idx, rf_end_idx]


def compute_proto_layer_rf_info(ts_len, latent_proto_len, layers):
    rf_info = ReceptiveFieldInfo(conv_ts_len=ts_len, size=1, center=0.5, jump=1)

    for l in layers:
        if type(l) in (torch.nn.Conv1d, torch.nn.MaxPool1d):
            rf_info = compute_layer_rf_info(layer_filter_size=l.kernel_size[0] if type(l.kernel_size) is tuple else l.kernel_size,
                                            layer_stride=l.stride[0] if type(l.stride) is tuple else l.stride,
                                            layer_padding=l.padding[0] if type(l.padding) is tuple else l.padding,
                                            previous_layer_rf_info=rf_info)

    # account for prototype length
    rf_info = compute_layer_rf_info(layer_filter_size=latent_proto_len,
                                    layer_stride=1,
                                    layer_padding='valid',
                                    previous_layer_rf_info=rf_info)

    return rf_info


def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info: ReceptiveFieldInfo):
    ts_len_in = previous_layer_rf_info.conv_ts_len

    if layer_padding == 'same':
        ts_len_out = math.ceil(float(ts_len_in) / float(layer_stride))
        if (ts_len_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (ts_len_in % layer_stride), 0)
        assert(ts_len_out == math.floor((ts_len_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (ts_len_out-1)*layer_stride - ts_len_in + layer_filter_size) # sanity check
    elif layer_padding == 'valid':
        ts_len_out = math.ceil(float(ts_len_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(ts_len_out == math.floor((ts_len_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (ts_len_out-1)*layer_stride - ts_len_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        ts_len_out = math.floor((ts_len_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    jump_out = previous_layer_rf_info.jump * layer_stride
    size_out = previous_layer_rf_info.size + (layer_filter_size - 1) * previous_layer_rf_info.jump
    center_out = previous_layer_rf_info.center + ((layer_filter_size - 1)/2 - pL) * previous_layer_rf_info.jump
    return ReceptiveFieldInfo(
        conv_ts_len=ts_len_out, size=size_out, center=center_out, jump=jump_out
    )
