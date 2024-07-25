# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision

from torch import nn
from typing import Dict, List
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelMaxPool)

import torch.nn.functional as F

from .transformer import _get_activation_fn
from ..util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (not train_backbone
                    or 'layer2' not in name
                    and 'layer3' not in name
                    and 'layer4' not in name):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        super().__init__(backbone, train_backbone,
                         return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class layer_backbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='relu', dropout=0.1):
        super(layer_backbone, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.dropout(self.activation(self.linear(x)))
        x = self.norm(x)
        return x


class Kinet_Backbone(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu', return_interm_layers=False):
        super().__init__()
        current_dim = input_dim
        initial_dim = []
        # self.num_layers = len(hidden_dims)
        self.return_interm_layers = return_interm_layers
        self.num_channels = hidden_dims
        for i, dim in enumerate(hidden_dims):
            initial_dim += [current_dim]
            current_dim = dim

        self.layers = nn.ModuleList(
            [layer_backbone(initial_dim[i], hidden_dims[i], activation) for i in range(len(hidden_dims))])

    def forward(self, tensor_list):

        x = tensor_list.tensors
        mask = tensor_list.mask
        if self.return_interm_layers:
            results_itnermediate = []
        out = []

        for i, l in enumerate(self.layers):
            x = l(x)
            # mask = F.interpolate(mask.float(), size=x.shape[-1:]).to(torch.bool)
            out += [NestedTensor(x, mask)]
            if self.return_interm_layers:
                results_itnermediate += [x]

        if self.return_interm_layers:
            return out, results_itnermediate

        return out[-1]


# class Joiner_kine(nn.Sequential):
#     def __init__(self,backbone, position_embedding):
#         super().__init__(position_embedding, backbone)
#         self.num_channels = backbone.num_channels
#
#     def forward(self, tensor_list: NestedTensor):
#         embedding_pos = self[0](tensor_list)
#         out = self[1](embedding_pos).to(embedding_pos.tensors.dtype)
#         return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if args.kinet:
        if args.use_encoding_tracklets:
            args.input_dim = args.encoding_dim_detections * 4
        else:
            args.input_dim = 4
        if args.use_class:
            args.input_dim += 2  # confidence + class
        else:
            args.input_dim += 1  # confidence
        backbone = Kinet_Backbone(args.input_dim, hidden_dims=[32, 64, args.hidden_dim], activation=args.activation,
                                  return_interm_layers=return_interm_layers)
        return backbone
    else:
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        backbone = Backbone(args.backbone,
                            train_backbone,
                            return_interm_layers,
                            args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
