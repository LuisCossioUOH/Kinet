# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import (NestedTensor, NestedTensorKinet, accuracy, dice_loss, get_world_size,
                         interpolate, is_dist_avail_and_initialized,
                         nested_tensor_from_tensor_list, sigmoid_focal_loss)


class DETR(nn.Module):
    """ This is the DETR module that performs object detection. """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, overflow_boxes=False, multi_frame_encoding=False, multi_frame_attention=False,
                 merge_frame_features=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        # self.input_proj = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        #     )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.multi_frame_encoding = multi_frame_encoding
        self.multi_frame_attention = multi_frame_attention
        self.merge_frame_features = merge_frame_features

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        # src = self.input_proj[-1](src)
        src = self.input_proj(src)
        pos = pos[-1]

        batch_size, _, _, _ = src.shape

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            # [BATCH_SIZE, NUM_PROBS, 4]
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

            num_track_queries = track_query_hs_embeds.shape[1]

            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

            for i, target in enumerate(targets):
                target['track_query_hs_embeds'] = tgt[:, i]

        assert mask is not None
        if self.multi_frame_encoding:
            pos_embed = [pos[:, i] for i in range(pos.size()[1])]
            pos_embed = [pos_embed.flatten(2).permute(0, 2, 1)]
            pos = torch.cat(pos_embed, 1)

        # else:
        #     pos = pos[:, 0]

        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, pos, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs_without_norm[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class KineT(nn.Module):
    """ This is the Kine module that performs tracking trough detections """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, overflow_boxes=False,
                 dim_tracklets=256,
                 #  multi_frame_encoding=False,multi_frame_attention=False,
                 # merge_frame_features=False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj_tracklets = MLP(dim_tracklets, self.hidden_dim, self.hidden_dim, 3)
        # self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        # self.input_proj = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        #     )])

        self.backbone = backbone
        self.aux_loss = aux_loss

        # self.multi_frame_encoding = multi_frame_encoding
        # self.multi_frame_attention = multi_frame_attention
        # self.merge_frame_features = merge_frame_features

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features = self.backbone(samples)

        src, mask = features.decompose()
        # src = self.input_proj[-1](src)
        # src = self.input_proj(features)
        # pos = pos[-1]

        batch_size = src.size()[0]

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and (len(targets[0]['track_query_hs_embeds']) > 0):
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

            num_track_queries = track_query_hs_embeds.shape[1]

            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = self.input_proj_tracklets(track_query_hs_embeds.transpose(0, 1))

            # for i, target in enumerate(targets):
            #     target['track_query_hs_embeds'] = tgt[:, i]
            # tgt += query # try no query embed
        # assert mask is not None
        # if self.multi_frame_encoding:
        #     pos_embed = [pos[:, i] for i in range(pos.size()[1])]
        #     pos_embed = [pos_embed.flatten(2).permute(0, 2, 1)]
        #     pos = torch.cat(pos_embed, 1)

        # else:
        #     pos = pos[:, 0]
        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs_without_norm[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class KinematicDetectorTransformer(nn.Module):
    def __init__(self, backbone: list, transformer, num_classes: int, num_queries: int, aux_loss=False,
                 overflow_boxes=False,
                 dim_tracklets_det=128, dim_tracklets_metadata=8,
                 #  multi_frame_encoding=False,multi_frame_attention=False,
                 # merge_frame_features=False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: list of backbones modules. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed_det = nn.Embedding(num_queries, self.hidden_dim)
        self.query_embed_metadata = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj_tracklets_det = MLP(dim_tracklets_det, self.hidden_dim, self.hidden_dim, 3)
        self.input_proj_tracklets_metadata = MLP(dim_tracklets_metadata, self.hidden_dim // 2, self.hidden_dim, 3)
        # self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        # self.input_proj = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        #     )])

        self.backbone_det = backbone[0]
        self.backbone_metadata = backbone[1]
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensorKinet, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """

        features_det, pos_det = self.backbone_det(samples.detections)
        features_metadata, pos_metadata = self.backbone_metadata(samples.metadata)

        src_det, mask = features_det[-1].decompose()
        src_metadata, _ = features_metadata[-1].decompose()

        batch_size = src_det.size()[0]

        query_embed_det = self.query_embed_det.weight
        query_embed_metadata = self.query_embed_metadata.weight
        query_embed_det = query_embed_det.unsqueeze(1).repeat(1, batch_size, 1)
        query_embed_metadata = query_embed_metadata.unsqueeze(1).repeat(1, batch_size, 1)
        tgt_det = None
        tgt_metadata = None
        if targets is not None and (len(targets[0]['track_query_hs_embeds_det']) > 0):
            track_query_hs_embeds_det = torch.stack([t['track_query_hs_embeds_det'] for t in targets])

            num_track_queries = track_query_hs_embeds_det.shape[1]

            track_query_embed_det = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed_det.device)
            query_embed_det = torch.cat([
                track_query_embed_det,
                query_embed_det], dim=0)

            tgt_det = torch.zeros_like(query_embed_det)
            tgt_det[:num_track_queries] = self.input_proj_tracklets_det(track_query_hs_embeds_det.transpose(0, 1))

            track_query_hs_embeds_meta = torch.stack([t['track_query_hs_embeds_meta'] for t in targets])

            track_query_embed_metadata = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed_metadata.device)
            query_embed_metadata = torch.cat([
                track_query_embed_metadata,
                query_embed_det], dim=0)

            tgt_metadata = torch.zeros_like(query_embed_metadata)
            tgt_metadata[:num_track_queries] = self.input_proj_tracklets_metadata(
                track_query_hs_embeds_meta.transpose(0, 1))
        hs_det, hs_metadata, hs_without_norm_det, memory_det = self.transformer(
            src_det, src_metadata, mask, query_embed_det, query_embed_metadata, tgt_det, tgt_metadata,
            pos_boxes=pos_det[0], pos_metadata=pos_metadata[0])

        # hs_metadata = src_metadata[None] # DELETE
        # hs_det = src_det[None] # DELETE

        outputs_class = self.class_embed(hs_metadata)
        outputs_coord = self.bbox_embed(hs_det).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               # 'hs_embed': hs_without_norm_det[-1]  ###
               }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out, targets, features_det, src_det, hs_det

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class KinematicDetectorEncoder(nn.Module):
    def __init__(self, backbone: list, encoder: nn.Module, num_classes: int, num_queries: int, aux_loss=False,
                 overflow_boxes=False,
                 dim_tracklets_det=128, dim_tracklets_metadata=8,
                 #  multi_frame_encoding=False,multi_frame_attention=False,
                 # merge_frame_features=False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: list of backbones modules. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image.
        """
        super().__init__()

        self.num_queries = num_queries
        self.encoder = encoder
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        # self.query_embed_det = nn.Embedding(num_queries, self.hidden_dim)
        # self.query_embed_metadata = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj_tracklets_det = MLP(dim_tracklets_det, self.hidden_dim, self.hidden_dim, 3)
        self.input_proj_tracklets_metadata = MLP(dim_tracklets_metadata, self.hidden_dim // 2, self.hidden_dim, 3)

        self.backbone_det = backbone[0]
        self.backbone_metadata = backbone[1]
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.encoder.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensorKinet, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """

        features_det, pos_det = self.backbone_det(samples.detections)
        features_metadata, pos_metadata = self.backbone_metadata(samples.metadata)

        src_det, mask = features_det[-1].decompose()
        src_metadata, _ = features_metadata[-1].decompose()

        batch_size, n_dets = src_det.size()[:2]

        # query_embed_det = self.query_embed_det.weight
        # query_embed_metadata = self.query_embed_metadata.weight
        # query_embed_det = query_embed_det.unsqueeze(1).repeat(1, batch_size, 1)
        # query_embed_metadata = query_embed_metadata.unsqueeze(1).repeat(1, batch_size, 1)
        # tgt_det = None
        # tgt_metadata = None
        if targets is not None and (len(targets[0]['track_query_hs_embeds_det']) > 0):
            track_query_src_det = torch.stack([t['track_query_hs_embeds_det'] for t in targets])

            num_track_queries = track_query_src_det.shape[1]

            # track_query_embed_det = torch.zeros(
            #     num_track_queries,
            #     batch_size,
            #     self.hidden_dim).to(query_embed_det.device)
            # query_embed_det = torch.cat([
            #     track_query_embed_det,
            #     query_embed_det], dim=0)

            # tgt_det = torch.zeros_like(query_embed_det)
            track_query_src_det = self.input_proj_tracklets_det(track_query_src_det)

            src_det = torch.cat([
                track_query_src_det,
                src_det], dim=1)
            new_pos_det = torch.zeros([batch_size, n_dets + num_track_queries, self.hidden_dim]).to(src_det.device)
            new_pos_det[:,num_track_queries:] = pos_det[0]
            pos_det = [new_pos_det]

            track_query_src_metadata = torch.stack([t['track_query_hs_embeds_meta'] for t in targets])
            track_query_src_metadata = self.input_proj_tracklets_metadata(track_query_src_metadata)
            src_metadata = torch.cat([
                track_query_src_metadata,
                src_metadata], dim=1)
            new_pos_metadata = torch.zeros([batch_size, n_dets + num_track_queries, self.hidden_dim]).to(src_det.device)
            new_pos_metadata[:, num_track_queries:] = pos_metadata[0]
            pos_metadata = [new_pos_metadata]

            new_mask = torch.zeros([batch_size,n_dets + num_track_queries],dtype=torch.bool).to(src_det.device)
            # new_mask[:, :num_track_queries] = True
            new_mask[:,num_track_queries:] = mask
            mask = new_mask

        hs_det, hs_metadata, memory_metadata, memory_det = self.encoder(
            src_det, src_metadata, mask,  pos_boxes=pos_det[0], pos_metadata=pos_metadata[0])


        outputs_class = self.class_embed(hs_metadata)
        outputs_coord = self.bbox_embed(hs_det).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               # 'hs_embed': hs_without_norm_det[-1]  ###
               }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out, targets, features_det, src_det, hs_det

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, tracking, track_query_false_positive_eos_weight):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tracking = tracking
        self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight

    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # print(src_logits[1,indices[1][0][:4],0::20])

        idx = self._get_src_permutation_idx(indices)
        # len_indices = [len(J) for I, J in indices]
        # len_targets = [t['boxes'].size()[0] for t in targets]
        # same = torch.tensor([i == t for i, t in zip(len_indices, len_targets)], dtype=torch.bool) # DELETE
        # if not same.all():
        #     print('found')
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        if self.tracking and self.track_query_false_positive_eos_weight:
            for i, target in enumerate(targets):
                if 'track_query_boxes' in target:
                    # remove no-object weighting for false track_queries
                    loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                    # assign false track_queries to some object class for the final weighting
                    target_classes = target_classes.clone()
                    target_classes[i, target['track_queries_fal_pos_mask']] = 0

        # weight = None
        # if self.tracking:
        #     weight = torch.stack([~t['track_queries_placeholder_mask'] for t in targets]).float()
        #     loss_ce *= weight

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # query_mask = None
        # if self.tracking:
        #     query_mask = torch.stack([~t['track_queries_placeholder_mask'] for t in targets])[..., None]
        #     query_mask = query_mask.repeat(1, 1, self.num_classes)

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma)
        # , query_mask=query_mask)

        # if self.tracking:
        #     mean_num_queries = torch.tensor([len(m.nonzero()) for m in query_mask]).float().mean()
        #     loss_ce *= mean_num_queries
        # else:
        #     loss_ce *= src_logits.shape[1]
        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        # compute seperate track and object query losses
        # loss_ce = sigmoid_focal_loss(
        #     src_logits, target_classes_onehot, num_boxes,
        #     alpha=self.focal_alpha, gamma=self.focal_gamma, query_mask=query_mask, reduction=False)
        # loss_ce *= src_logits.shape[1]

        # track_query_target_masks = []
        # for t, ind in zip(targets, indices):
        #     track_query_target_mask = torch.zeros_like(ind[1]).bool()

        #     for i in t['track_query_match_ids']:
        #         track_query_target_mask[ind[1].eq(i).nonzero()[0]] = True

        #     track_query_target_masks.append(track_query_target_mask)
        # track_query_target_masks = torch.cat(track_query_target_masks)

        # losses['loss_ce_track_queries'] = loss_ce[idx][track_query_target_masks].mean(1).sum() / num_boxes
        # losses['loss_ce_object_queries'] = loss_ce[idx][~track_query_target_masks].mean(1).sum() / num_boxes

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # compute seperate track and object query losses
        # track_query_target_masks = []
        # for t, ind in zip(targets, indices):
        #     track_query_target_mask = torch.zeros_like(ind[1]).bool()

        #     for i in t['track_query_match_ids']:
        #         track_query_target_mask[ind[1].eq(i).nonzero()[0]] = True

        #     track_query_target_masks.append(track_query_target_mask)
        # track_query_target_masks = torch.cat(track_query_target_masks)

        # losses['loss_bbox_track_queries'] = loss_bbox[track_query_target_masks].sum() / num_boxes
        # losses['loss_bbox_object_queries'] = loss_bbox[~track_query_target_masks].sum() / num_boxes

        # losses['loss_giou_track_queries'] = loss_giou[track_query_target_masks].sum() / num_boxes
        # losses['loss_giou_object_queries'] = loss_giou[~track_query_target_masks].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # ids = [t['image_id'] for t in targets]
        # if 100 in ids:
        #     index_check = (torch.tensor(ids) == 100).nonzero()
            # print('ids: ', ids)
            # print('ids: ', ids[index_check])
        indices = self.matcher(outputs_without_aux, targets)
         # DELETE
        # if 100 in ids:
        #     print('indices pred: ', indices[index_check])
        #     if indices[index_check][0][0] > 5:
        #         print("found: ",)
        #         indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = self.process_boxes(out_bbox, target_sizes)

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'scores_no_object': s_n_o}
            for s, l, b, s_n_o in zip(scores, labels, boxes, prob[..., -1])]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
