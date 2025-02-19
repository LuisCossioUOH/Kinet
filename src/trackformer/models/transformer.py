# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy

import torch
import torch.nn.functional as F

from typing import Optional
from torch import Tensor, nn


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 track_attention=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, encoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            track_attention=track_attention)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, tgt=None, prev_frame=None):
        # flatten BSxCxHxW to BSxCxHW
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # permute BSxCxHW to HWxBSxC

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # pos_embed = [flatten_dim,batch_size,n_frames] = []
        mask = mask.flatten(1)

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        memory_prev_frame = None
        if prev_frame is not None:
            src_prev_frame = prev_frame['src'].flatten(2).permute(2, 0, 1)
            pos_embed_prev_frame = prev_frame['pos'].flatten(2).permute(2, 0, 1)
            mask_prev_frame = prev_frame['mask'].flatten(1)

            memory_prev_frame = self.encoder(
                src_prev_frame, src_key_padding_mask=mask_prev_frame, pos=pos_embed_prev_frame)

            prev_frame['memory'] = memory_prev_frame
            prev_frame['memory_key_padding_mask'] = mask_prev_frame
            prev_frame['pos'] = pos_embed_prev_frame

        hs, hs_without_norm = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                           pos=pos_embed, query_pos=query_embed,
                                           prev_frame=prev_frame)
        # hs [n_levels, num_queries, batch_size, hidden_dim] -> [n_levels, batch_size, num_queries, hidden_dim]
        return (hs.transpose(1, 2),
                hs_without_norm.transpose(1, 2),
                memory.permute(1, 2, 0).view(bs, c, h, w)) # memory [flatten_dim, batch_size, hidden_dim] -> [batch_size, hidden_dim, height, width]


class KinematicTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,  ### TODO: REMOVE UNUSED KWARGS
                 track_attention=False):

        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, encoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            track_attention=track_attention)

        self._reset_parameters()

        # self.linear1 = nn.Linear(d_model, d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_model, d_model)
        # self.norm = nn.LayerNorm(d_model)
        # self.activation_out = _get_activation_fn(activation)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, tgt=None, pos_src=None):
        # bs, n_det, c = src.shape
        src = src.permute(1, 0, 2)  # permute BSxTxC to TxBSxC
        if not (pos_src is None):
            pos_src = pos_src.permute(1, 0, 2)
        # pos_embed = [flatten_dim,batch_size,n_frames] = []
        # mask = mask.flatten(2)

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        # pos_src = self.norm(self.linear2(self.dropout(self.activation_out(self.linear1(src)))))
        # pos = None
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_src)

        hs, hs_without_norm = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                           pos=pos_src, query_pos=query_embed)

        return (hs,
                hs_without_norm,
                memory)


class DualKinematicTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 track_attention=False):
        super().__init__()

        self.d_model = d_model

        self.transformer_det = KinematicTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                    dim_feedforward, dropout,
                                                    activation, normalize_before, return_intermediate_dec,
                                                    track_attention)
        self.transformer_metadata = KinematicTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                         dim_feedforward, dropout,
                                                         activation, normalize_before, return_intermediate_dec,
                                                         track_attention)

        self.detection_branch = IntertwinedBranch(d_model, dropout, activation)

        self.metadata_branch = IntertwinedBranch(d_model, dropout, activation)

    def forward(self, src_boxes, src_metadata, mask, query_embed_bbox, query_embed_metadata, tgt_bboxes, tgt_metadata,
                pos_boxes=None, pos_metadata=None):
        # bs, n_det, c = src_boxes.shape
        hs_det, hs_without_norm_det, memory_det = self.transformer_det(src_boxes, mask=mask,
                                                                       query_embed=query_embed_bbox,
                                                                       tgt=tgt_bboxes,
                                                                       pos_src=pos_boxes)

        hs_metadata, hs_without_norm_metadata, memory_metadata = self.transformer_metadata(src_metadata, mask=mask,
                                                                                           query_embed=query_embed_metadata,
                                                                                           tgt=tgt_metadata,
                                                                                           pos_src=pos_metadata)

        hs_det = self.detection_branch(hs_det, hs_metadata)
        hs_metadata = self.metadata_branch(hs_metadata, hs_det)
        return (hs_det.transpose(1, 2), hs_metadata.transpose(1, 2),
                hs_without_norm_det.transpose(1, 2),
                memory_det.permute(1, 0, 2))


class DualKinematicEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 ):
        super().__init__()

        self.d_model = d_model
        encoder_layer_det = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        encoder_norm_det = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_det = TransformerEncoder(encoder_layer_det, num_encoder_layers, encoder_norm_det)

        encoder_layer_meta = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, activation, normalize_before)

        encoder_norm_meta = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_meta = TransformerEncoder(encoder_layer_meta, num_encoder_layers, encoder_norm_meta)

        self.detection_branch = IntertwinedBranch(d_model, dropout, activation, dim_concat=2)

        self.metadata_branch = IntertwinedBranch(d_model, dropout, activation, dim_concat=2)

    def forward(self, src_boxes, src_metadata, mask, pos_boxes=None, pos_metadata=None):


        src_boxes = src_boxes.permute(1, 0, 2)  # permute BSxTxC to TxBSxC
        if not (pos_boxes is None):
            pos_boxes = pos_boxes.permute(1, 0, 2)

        memory_det = self.encoder_det(src_boxes, src_key_padding_mask=mask,
                                      pos=pos_boxes
                                      )


        src_metadata = src_metadata.permute(1, 0, 2)  # permute BSxTxC to TxBSxC
        if not (pos_boxes is None):
            pos_metadata = pos_metadata.permute(1, 0, 2)
        # pos_embed = [flatten_dim,batch_size,n_frames] = []
        # mask = mask.flatten(2)

        # if tgt_metadata is None:
        #     tgt_metadata = torch.zeros_like(query_embed_metadata)
        memory_metadata = self.encoder_meta(src_metadata, src_key_padding_mask=mask,
                                            pos=pos_metadata
                                            )

        hs_det = self.detection_branch(memory_det, memory_metadata)
        hs_metadata = self.metadata_branch(memory_metadata, memory_det)
        return (hs_det.transpose(0, 1)[None], hs_metadata.transpose(0, 1)[None],
                memory_metadata.permute(1, 2, 0),
                memory_det.permute(1, 2, 0))


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, encoder_layer, num_layers,
                 norm=None, return_intermediate=False, track_attention=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        self.track_attention = track_attention
        if self.track_attention:
            self.layers_track_attention = _get_clones(encoder_layer, num_layers)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                prev_frame: Optional[dict] = None):
        output = tgt

        intermediate = []

        if self.track_attention:
            track_query_pos = query_pos[:-100].clone()
            query_pos[:-100] = 0.0

        for i, layer in enumerate(self.layers):
            if self.track_attention:
                track_output = output[:-100].clone()

                track_output = self.layers_track_attention[i](
                    track_output,
                    src_mask=tgt_mask,
                    src_key_padding_mask=tgt_key_padding_mask,
                    pos=track_query_pos)

                output = torch.cat([track_output, output[-100:]])

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            output = torch.stack(intermediate)

        if self.norm is not None:
            return self.norm(output), output
        return output, output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # pos = [n_frame,flatten_dim,hidden_dim], tensor = [n_frame, flatten_dim, hidden_dim]

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class IntertwinedBranch(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, activation='relu', dim_concat=3):
        super().__init__()
        self.linear_input1 = nn.Linear(d_model, d_model // 2)
        self.linear_input2 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model // 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.dim_concat = dim_concat

    def forward(self, src1, src2):
        x1 = self.linear_input1(src1)
        x2 = self.linear_input2(src2)
        x = self.activation(torch.cat([x1, x2], dim=self.dim_concat))
        return self.norm(self.dropout(x) + src1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    if args.kine:
        if args.use_encoder_only:
            return DualKinematicEncoder(
                d_model=args.hidden_dim,
                nhead=args.nheads,
                num_encoder_layers=args.enc_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=args.activation,
                normalize_before=args.pre_norm,)
        else:
            return DualKinematicTransformer(
                d_model=args.hidden_dim,
                nhead=args.nheads,
                num_encoder_layers=args.enc_layers,
                num_decoder_layers=args.dec_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=args.activation,
                normalize_before=args.pre_norm,
                return_intermediate_dec=True,
                track_attention=args.track_attention)
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        track_attention=args.track_attention
    )
