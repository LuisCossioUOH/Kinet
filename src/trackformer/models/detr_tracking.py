import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn

from .deformable_detr import DeformableDETR
from .detr import DETR, KineT, KinematicDetector
from .matcher import HungarianMatcher
from ..util.misc import NestedTensor, NestedTensorKinet


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False, ):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        device = prev_out['pred_boxes'].device

        # for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])

        num_prev_target_ind = 0
        if min_prev_target_ind:
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

        num_prev_target_ind_for_fps = 0
        if num_prev_target_ind:
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1,
                              (1,)).item()

        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            # random subset
            if self._track_query_false_negative_prob:  # and len(prev_target_ind):
                # random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                # random_subset_mask = random_subset_mask.ge(
                #     self._track_query_false_negative_prob)

                # random_subset_mask = torch.randperm(len(prev_target_ind))[:torch.randint(0, len(prev_target_ind) + 1, (1,))]
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]

                # if not len(random_subset_mask):
                #     target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                #     target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                #     target['track_query_match_ids'] = torch.tensor([]).long().to(device)
                #     continue

                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]

            # detected prev frame tracks
            prev_track_ids = target['prev_target']['track_ids'][prev_target_ind]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # current frame track ids detected in the prev frame
            # track_ids = target['track_ids'][target_ind_matched_idx]

            # index of prev frame detection in current frame box list
            target['track_query_match_ids'] = target_ind_matched_idx

            # random false positives
            if add_false_pos:
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]

                random_false_out_ind = []

                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                # for j, prev_box_matched in enumerate(prev_boxes_matched):
                #     if j not in prev_target_ind_for_fps:
                #         continue

                for j in prev_target_ind_for_fps:
                    # if random.uniform(0, 1) < self._track_query_false_positive_prob:
                    prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                    # only cxcy
                    # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                    # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                    # box_weights = 1.0 / box_dists.add(1e-8)

                    # prev_box_ious, _ = box_ops.box_iou(
                    #     box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                    #     box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                    # box_weights = prev_box_ious[0]

                    # dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

                    if len(prev_boxes_matched) > j:
                        prev_box_matched = prev_boxes_matched[j]
                        box_weights = \
                            prev_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)

                        # if box_weights.gt(0.0).any():
                        # if box_weights.gt(0.0).any():
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])

                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()

                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # MSDeformAttn can not deal with empty inputs therefore we
            # add single false pos to have at least one track query per sample
            # not_prev_out_ind = torch.tensor([
            #     ind
            #     for ind in torch.arange(prev_out['pred_boxes'].shape[1])
            #     if ind not in prev_out_ind])
            # false_samples_inds = torch.randperm(not_prev_out_ind.size(0))[:1]
            # false_samples = not_prev_out_ind[false_samples_inds]
            # prev_out_ind = torch.cat([prev_out_ind, false_samples])
            # target_ind_matching = torch.tensor(
            #     target_ind_matching.tolist() + [False, ]).bool().to(target_ind_matching.device)

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # track_queries_match_mask = torch.ones_like(target_ind_matching).float()
            # matches indices with 1.0 and not matched -1.0
            # track_queries_mask[~target_ind_matching] = -1.0

            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

        # add placeholder track queries to allow for batch sizes > 1
        # max_track_query_hs_embeds = max([len(t['track_query_hs_embeds']) for t in targets])
        # for i, target in enumerate(targets):

        #     num_add = max_track_query_hs_embeds - len(target['track_query_hs_embeds'])

        #     if not num_add:
        #         target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #         continue

        #     raise NotImplementedError

        #     target['track_query_hs_embeds'] = torch.cat(
        #         [torch.zeros(num_add, self.hidden_dim).to(device),
        #          target['track_query_hs_embeds']
        #     ])
        #     target['track_query_boxes'] = torch.cat(
        #         [torch.zeros(num_add, 4).to(device),
        #          target['track_query_boxes']
        #     ])

        #     target['track_queries_mask'] = torch.cat([
        #         torch.tensor([True, ] * num_add).to(device),
        #         target['track_queries_mask']
        #     ]).bool()

        #     target['track_queries_fal_pos_mask'] = torch.cat([
        #         torch.tensor([False, ] * num_add).to(device),
        #         target['track_queries_fal_pos_mask']
        #     ]).bool()

        #     target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #     target['track_queries_placeholder_mask'][:num_add] = True

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
                # if True:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if 'prev_prev_image' in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target['prev_target'] = target['prev_prev_target']

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # PREV PREV
                        prev_prev_out, _, prev_prev_features, _, _ = super().forward(
                            [t['prev_prev_image'] for t in targets])

                        prev_prev_outputs_without_aux = {
                            k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                        prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                        self.add_track_queries_to_targets(
                            prev_targets, prev_prev_indices, prev_prev_out, add_false_pos=False)

                        # PREV
                        prev_out, _, prev_features, _, _ = super().forward(
                            [t['prev_image'] for t in targets],
                            prev_targets,
                            prev_prev_features)
                    else:
                        prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets])

                    # prev_out = {k: v.detach() for k, v in prev_out.items() if torch.is_tensor(v)}

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)
                    device = prev_targets[0]['labels'].device
                    new_prev_indices = []
                    for idx_output, idx_target in prev_indices:
                        new_prev_indices += [(idx_output.to(device), idx_target.to(device))]

                    self.add_track_queries_to_targets(targets, new_prev_indices, prev_out)
            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                    # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_match_ids'] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs = super().forward(samples, targets, prev_features)

        return out, targets, features, memory, hs


class SineEncodingTracklet:
    """
    Position embeding for tracklets locations (range between [0,1]) very similar to the one
    used by the Attention is all you need paper, applied to detections [N_det, 4]
    """

    def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def __call__(self, x):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        freq = (x[:, :, :, None] * torch.pi * 2) / dim_t
        embed_coord = torch.concatenate([freq[:, :, :, 0::2].cos(), freq[:, :, :, 1::2].sin()], dim=3)
        embed_coord = embed_coord.flatten(1)

        return embed_coord


class IdentityEncoding:
    def __init__(self):
        pass

    def __call__(self, x):
        x = x.flatten(1)
        return x


def generate_pseudo_tracklets(detections, n_frames):
    pseudo_tracklets = torch.tile(detections[:, None, :4], [1, n_frames, 1])
    return pseudo_tracklets


class KinetTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False,
                 ratio_add_detections=0.5,
                 frame_range=5,
                 use_encoding=True,
                 num_pos_feats=32,
                 ratio_add_tracklets=1.0
                 ):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        self._ratio_add_detections = ratio_add_detections
        self._frame_range = frame_range
        self._ratio_add_tracklets = ratio_add_tracklets
        if use_encoding:
            self._embed_tracklets = SineEncodingTracklet(num_pos_feats)
            self.dim_tracklets = 4 * num_pos_feats * frame_range
        else:
            self._embed_tracklets = IdentityEncoding()
            self.dim_tracklets = 4 * frame_range
        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_positive_detections_to_tracklets(self, target, detections, tracks_indices, det_ind, ):
        picked_indices = self._ratio_add_detections > torch.rand([len(tracks_indices)])
        replace_track_indices = tracks_indices[picked_indices]
        replace_det_indices = det_ind[picked_indices]
        pseudo_tracklets = generate_pseudo_tracklets(detections[replace_det_indices], self._frame_range)
        target['tracklets_modified'][replace_track_indices] = pseudo_tracklets

    def get_minimun_tracks(self, targets):
        indices_targets_matched = []
        indices_detections_matched = []
        for target in targets:
            detections = target['detections'][:, :4]
            tracks_indices, det_ind = self._matcher(detections, target)
            indices_targets_matched += [tracks_indices]
            indices_detections_matched += [det_ind]

        max_prev_target_ind = min([len(indices_det) for indices_det in indices_detections_matched])
        min_prev_target_ind = int(max_prev_target_ind * self._ratio_add_tracklets)
        num_min_target_ind = 0
        if min_prev_target_ind:
            num_min_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()
        return num_min_target_ind, indices_targets_matched, indices_detections_matched

    def add_track_queries_to_targets(self, targets, samples, add_false_pos=True):
        device = targets[0]['boxes'].device
        # n_frames = len(targets[0]['tracklets'])

        num_targets_ind, indices_matched_targets, indices_matched_dets = self.get_minimun_tracks(targets)

        num_prev_target_ind_for_fps = 0
        if num_targets_ind:
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_targets_ind)) + 1, (1,)).item()
            remaining_detections = min([len(target['detections']) - num_targets_ind for target in targets])
            num_prev_target_ind_for_fps = min(num_prev_target_ind_for_fps, remaining_detections)

        # if self._track_query_false_negative_prob > random.random() and num_targets_ind > 2:
        #     false_negative = True
        #     n_positives = torch.randint(1, num_targets_ind - 1, [1])[0]
        if num_targets_ind > 0:
            for i, (target) in enumerate(targets):
                # prev_det_ind, prev_tracks_ind = prev_ind
                detections = target['detections']

                tracks_indices = indices_matched_targets[i]
                det_ind = indices_matched_dets[i]
                subset_matches = torch.randperm(len(tracks_indices))[:num_targets_ind]
                tracks_indices = tracks_indices[subset_matches]
                det_ind = det_ind[subset_matches]
                target['tracklets_modified'] = target['tracklets'].clone().permute(1, 0, 2)

                # if false_negative:
                #     tracks_indices = tracks_indices[:n_positives]
                #     random_subset_mask = torch.randperm(len(tracks_indices))[:n_positives]
                #     det_ind = det_ind[random_subset_mask]

                self.add_positive_detections_to_tracklets(target, detections, tracks_indices, det_ind)

                # detected prev frame tracks
                # track_ids = target['track_ids'][tracks_indices]
                # target['track_query_match_ids'] = track_ids

                target['track_query_match_ids'] = tracks_indices
                if add_false_pos:
                    tracklets_indices, tracks_matching_mask, \
                    track_queries_fal_pos_mask = self.add_false_positives(target, detections, tracks_indices, det_ind,
                                                                          device,
                                                                          num_prev_target_ind_for_fps)
                else:
                    tracklets_indices = tracks_indices
                    track_queries_fal_pos_mask = torch.tensor([False, ] * len(tracklets_indices)).bool().to(device)
                    tracks_matching_mask = torch.tensor([True, ] * len(track_queries_fal_pos_mask)).bool().to(device)

                self.update_query_embeddings(target, tracklets_indices, device, track_queries_fal_pos_mask,
                                             tracks_matching_mask)
        else:
            self.generate_empty_tracklets(targets)

    def generate_empty_tracklets(self, targets):
        for target in targets:
            device = target['boxes'].device

            # if self._use_encoding:
            target['track_query_hs_embeds'] = torch.zeros([0, self.dim_tracklets]).to(device)
            # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
            target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
            target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
            # target['track_query_boxes'] = torch.zeros(0, 4).to(device)
            target['track_query_match_ids'] = torch.tensor([]).long().to(device)

    def update_query_embeddings(self, target, tracklets_indices, device, track_queries_fal_pos_mask,
                                track_queries_mask):
        # set tracklets embedding info
        target['track_query_hs_embeds'] = self._embed_tracklets(target['tracklets_modified'][tracklets_indices])
        # target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()
        target['track_queries_mask'] = torch.cat([
            track_queries_mask,
            torch.tensor([False, ] * self.num_queries).to(device)
        ]).bool()
        target['track_queries_fal_pos_mask'] = torch.cat([
            track_queries_fal_pos_mask,
            torch.tensor([False, ] * self.num_queries).to(device)
        ]).bool()

    def add_false_positives(self, target, detections, track_indices, det_indices, device, num_target_ind_for_fps):

        fps_det_indidces = torch.arange(detections.size()[0])
        fps_det_indidces = [
            ind.item()
            for ind in fps_det_indidces
            if ind not in det_indices]
        if len(fps_det_indidces):
            detections_ind_for_fps = torch.randint(0, len(fps_det_indidces), [num_target_ind_for_fps])

            pseudo_tracklets = generate_pseudo_tracklets(
                detections[torch.tensor(fps_det_indidces)[detections_ind_for_fps]],
                self._frame_range)

            tracklets_indices = torch.tensor(track_indices.tolist() + \
                                             [i for i in range(len(target['tracklets_modified']),
                                                               len(target['tracklets_modified']) + \
                                                               num_target_ind_for_fps)], dtype=torch.int64).to(device)

            target['tracklets_modified'] = torch.cat([target['tracklets_modified'], pseudo_tracklets], dim=0)
            fps_tracks_matching = torch.tensor(
                [False, ] * len(track_indices) + [True, ] * num_target_ind_for_fps).bool().to(device)
            tracks_matching = torch.ones_like(fps_tracks_matching).bool().to(device)
            tracks_matching[fps_tracks_matching] = False

            return tracklets_indices, tracks_matching, fps_tracks_matching
        tracks_matching = torch.ones_like(track_indices).bool().to(device)
        fps_tracks_matching = torch.zeros_like(track_indices).bool().to(device)
        return track_indices, tracks_matching, fps_tracks_matching

    def forward(self, samples: NestedTensor, targets: list = None):
        if targets is not None and not self._tracking:
            # tracklets = [target['tracklets'] for target in targets]

            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    self.add_track_queries_to_targets(targets, samples)

            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                self.generate_empty_tracklets(targets)

        out, targets, features, memory, hs = super().forward(samples, targets)
        return out, targets, features, memory, hs


class KinetTrackingBase2(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False,
                 ratio_add_detections=0.5,
                 frame_range=5,
                 use_encoding=True,
                 num_pos_feats=32,
                 ratio_add_tracklets=1.0,
                 dim_metadata=1
                 ):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        self._ratio_add_detections = ratio_add_detections
        self._frame_range = frame_range
        self._ratio_add_tracklets = ratio_add_tracklets
        if use_encoding:
            self._embed_tracklets = SineEncodingTracklet(num_pos_feats)
            self.dim_tracklets = 4 * num_pos_feats * frame_range
        else:
            self._embed_tracklets = IdentityEncoding()
            self.dim_tracklets = 4 * frame_range
        self._tracking = False
        self.dim_metadata = dim_metadata

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_positive_detections_to_tracklets(self, target: dict, detections: torch.Tensor, tracks_indices: list,
                                             det_ind: list):
        """
        Replace tracklets with detections based tracklets. A detection based tracklets is a position
        repeated N times, that simulate a newly located object.
        @param target: labels dict
        @param detections: detections tensor
        @param tracks_indices:
        @param det_ind:
        @return:
        """
        picked_indices = self._ratio_add_detections > torch.rand([len(tracks_indices)])
        replace_track_indices = tracks_indices[picked_indices]
        replace_det_indices = det_ind[picked_indices]
        pseudo_tracklets = generate_pseudo_tracklets(detections[replace_det_indices], self._frame_range)
        target['tracklets_modified'][replace_track_indices] = pseudo_tracklets
        target['tracklet_metadata'][replace_track_indices][:, :, :self.dim_metadata] = detections[replace_det_indices][
                                                                                       :, 4:]

    def get_indices_matched_targets(self, targets: dict, ):
        """
        Calculate the indices that match between targets and detections
        @param targets:
        @return:
        """
        indices_targets_matched = []
        indices_detections_matched = []
        for target in targets:
            detections = target['detections'][:, :4]
            tracks_indices, det_ind = self._matcher(detections, target)
            indices_targets_matched += [tracks_indices]
            indices_detections_matched += [det_ind]

        return indices_targets_matched, indices_detections_matched

    def get_minimum_tracks(self, indices_detections_matched: list):
        """
        Calculate a random value of possible tracks to add as tracklets
        @param targets:
        @return:
        """
        max_prev_target_ind = min([len(indices_det) for indices_det in indices_detections_matched])
        min_prev_target_ind = int(max_prev_target_ind * self._ratio_add_tracklets)
        num_min_target_ind = 0
        if min_prev_target_ind:
            num_min_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()
        return num_min_target_ind

    def generate_empty_tracklets(self, targets: list):
        for target in targets:
            device = target['boxes'].device

            # if self._use_encoding:
            target['track_query_hs_embeds_det'] = torch.zeros([0, self.dim_tracklets]).to(device)
            target['track_query_hs_embeds_meta'] = torch.zeros([0, self.dim_metadata]).to(device)
            # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
            target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
            target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
            # target['track_query_boxes'] = torch.zeros(0, 4).to(device)
            target['track_query_match_ids'] = torch.tensor([]).long().to(device)

    def update_query_embeddings(self, target: dict, tracklets_indices: torch.Tensor, device: torch.device,
                                track_queries_fal_pos_mask: torch.Tensor,
                                track_queries_mask: torch.Tensor):
        # set tracklets embedding info
        target['track_query_hs_embeds_det'] = self._embed_tracklets(target['tracklets_modified'][tracklets_indices])
        target['track_query_hs_embeds_meta'] = target['tracklets_metadata'][tracklets_indices]

        target['track_queries_mask'] = torch.cat([
            track_queries_mask,
            torch.tensor([False, ] * self.num_queries).to(device)
        ]).bool()
        target['track_queries_fal_pos_mask'] = torch.cat([
            track_queries_fal_pos_mask,
            torch.tensor([False, ] * self.num_queries).to(device)
        ]).bool()

    def add_false_positives(self, target, detections, track_indices, det_indices, device, num_target_ind_for_fps,
                            confidence_range_fp=[0.5, 0.95]):
        """
        Method to calculate indices for tracklets to add. It passes as well indices that indicate whcih are positive
        tracklets (real tracks) and false positives tracklets.
        @param target: label dict
        @param detections: detections tensor [Ndet x 5] or [Ndet x 6]
        @param track_indices: indices tracklets to add
        @param det_indices: detections matched indices
        @param device: target device
        @param num_target_ind_for_fps: number false positive to add.
        @param confidence_range_fp: confidence of newly added false positives
        @return:
        """
        fps_det_indidces = torch.arange(detections.size()[0])
        fps_det_indidces = [
            ind.item()
            for ind in fps_det_indidces
            if ind not in det_indices]
        if len(fps_det_indidces):
            detections_ind_for_fps = torch.randint(0, len(fps_det_indidces), [num_target_ind_for_fps])

            pseudo_tracklets = generate_pseudo_tracklets(
                detections[torch.tensor(fps_det_indidces)[detections_ind_for_fps]],
                self._frame_range)

            tracklets_indices = torch.tensor(track_indices.tolist() + \
                                             [i for i in range(len(target['tracklets_modified']),
                                                               len(target['tracklets_modified']) + \
                                                               num_target_ind_for_fps)], dtype=torch.int64).to(device)

            pseudo_metadata = torch.zeros([target['tracklets_metada'].size()[0],
                                           target['tracklets_metada'].size()[1],
                                           self.dim_metadata])  #### TODO: Implement random class assignment

            pseudo_metadata[:, :, 0] = (confidence_range_fp[1] - confidence_range_fp[0]) * torch.rand(
                num_target_ind_for_fps) \
                                       + confidence_range_fp[0]
            target['tracklets_modified'] = torch.cat([target['tracklets_modified'], pseudo_tracklets], dim=0)
            target['tracklets_metada'] = torch.cat([target['tracklets_metada'], pseudo_metadata], dim=0)
            fps_tracks_matching = torch.tensor(
                [False, ] * len(track_indices) + [True, ] * num_target_ind_for_fps).bool().to(device)
            tracks_matching = torch.ones_like(fps_tracks_matching).bool().to(device)
            tracks_matching[fps_tracks_matching] = False

            return tracklets_indices, tracks_matching, fps_tracks_matching
        tracks_matching = torch.ones_like(track_indices).bool().to(device)
        fps_tracks_matching = torch.zeros_like(track_indices).bool().to(device)
        return track_indices, tracks_matching, fps_tracks_matching

    def add_track_queries_to_targets(self, targets, add_false_pos=True):
        # device = targets[0]['boxes'].device
        # indices_matched_targets, indices_matched_dets = self.get_indices_matched_targets(targets)
        # num_targets_ind = self.get_minimum_tracks(indices_matched_targets, indices_matched_dets)
        #
        # num_prev_target_ind_for_fps = 0
        # if num_targets_ind:
        #     num_prev_target_ind_for_fps = \
        #         torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_targets_ind)) + 1, (1,)).item()
        #     remaining_detections = min([len(target['detections']) - num_targets_ind for target in targets])
        #     num_prev_target_ind_for_fps = min(num_prev_target_ind_for_fps, remaining_detections)  # False positives
        #
        # if num_targets_ind > 0:
        #     for i, (target) in enumerate(targets):
        #         # prev_det_ind, prev_tracks_ind = prev_ind
        #         detections = target['detections']
        #
        #         tracks_indices = indices_matched_targets[i]
        #         det_ind = indices_matched_dets[i]
        #         subset_matches = torch.randperm(len(tracks_indices))[:num_targets_ind]
        #         tracks_indices = tracks_indices[subset_matches]
        #         det_ind = det_ind[subset_matches]
        #         target['tracklets_modified'] = target['tracklets'].clone().permute(1, 0, 2)
        #         target['tracklet_metadata'] = torch.ones([target['tracklets_modified'].size()[0],
        #                                                   target['tracklets_modified'].size()[1],
        #                                                   self.dim_metadata])
        #         # if false_negative:
        #         #     tracks_indices = tracks_indices[:n_positives]
        #         #     random_subset_mask = torch.randperm(len(tracks_indices))[:n_positives]
        #         #     det_ind = det_ind[random_subset_mask]
        #
        #         self.add_positive_detections_to_tracklets(target, detections, tracks_indices, det_ind)
        #
        #         # detected prev frame tracks
        #         # track_ids = target['track_ids'][tracks_indices]
        #         # target['track_query_match_ids'] = track_ids
        #
        #         target['track_query_match_ids'] = tracks_indices
        #         if add_false_pos:
        #             tracklets_indices, tracks_matching_mask, \
        #             track_queries_fal_pos_mask = self.add_false_positives(target, detections, tracks_indices, det_ind,
        #                                                                   device,
        #                                                                   num_prev_target_ind_for_fps)
        #         else:
        #             tracklets_indices = tracks_indices
        #             track_queries_fal_pos_mask = torch.tensor([False, ] * len(tracklets_indices)).bool().to(device)
        #             tracks_matching_mask = torch.tensor([True, ] * len(track_queries_fal_pos_mask)).bool().to(device)
        #
        #         self.update_query_embeddings(target, tracklets_indices, device, track_queries_fal_pos_mask,
        #                                      tracks_matching_mask)
        # else:
        #     self.generate_empty_tracklets(targets)
        self.generate_empty_tracklets(targets)

    def forward(self, samples: NestedTensorKinet, targets: list = None):
        if targets is not None and not self._tracking:

            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    self.add_track_queries_to_targets(targets)

            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                self.generate_empty_tracklets(targets)

        out, targets, features, memory, hs = super().forward(samples, targets)

        return out, targets, features, memory, hs


class KinetTracking(KinetTrackingBase2, KinematicDetector):
    def __init__(self, tracking_kwargs, transformer_kwargs):
        KinematicDetector.__init__(self, **transformer_kwargs)
        KinetTrackingBase2.__init__(self, **tracking_kwargs)


class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
