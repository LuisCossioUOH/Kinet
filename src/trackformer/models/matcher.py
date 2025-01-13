# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class BasicBoxHungarianMatcher:
    """This class computes an assignment between the targets and detection inputs.

    In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 2, cost_giou: float = 2, use_class=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_class = use_class
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"


    def __call__(self, detections, target):
        """ Performs the matching

        Params:
            detections: This is a tensor of detections of shape [n_det, 5] or [n_det, 6], depending on wether or not
            we are using classes as condition


            target: This is a targer/dict contining:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates


        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        out_bbox = detections[:,:4]

        tgt_ids = target['labels']
        tgt_bbox = target["boxes"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        if self.use_class:
            cost_class = (tgt_ids[None] != detections[:,5,None]).to(torch.int32)
            cost_matrix += self.cost_class * cost_class
        indices_detection, indices_target = linear_sum_assignment(cost_matrix.cpu())

        return (torch.as_tensor(indices_target, dtype=torch.int64),
                torch.as_tensor(indices_detection, dtype=torch.int64))

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        #
        # [batch_size * num_queries, num_classes]
        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        #
        for i, target in enumerate(targets):
            if 'track_query_match_ids' not in target:
                continue

            prop_i = 0
            for j in range(cost_matrix.shape[1]):
                # if target['track_queries_fal_pos_mask'][j] or target['track_queries_placeholder_mask'][j]:
                if target['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost_matrix[i, j] = np.inf
                elif target['track_queries_mask'][j]:
                    track_query_id = target['track_query_match_ids'][prop_i]
                    prop_i += 1

                    cost_matrix[i, j] = np.inf
                    cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
                    cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


class OrderDetectionsMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network taking
    into account that each prediction are expected to predict ordered detections. Unmatched detections are considered
    as no-objects.
    """

    def __init__(self, n_predictions, assignment_predictions, cost_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """Creates the matcher

        Params:
            n_predictions: N predictions generated by the model
            assignment_predictions: Number of assigned predictions to each detection
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        assert (
                    n_predictions % assignment_predictions == 0), "[ERROR] Invalid number of predictions/queries and assigned predictions per detection"
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_predictions = n_predictions
        self.n_assign = assignment_predictions
        self.max_predictions = n_predictions // assignment_predictions
        # TODO: Max cost passed as parameter
        self.max_cost = - self.cost_giou * 0.1 + self.cost_bbox * 0.6
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def calculate_matching_detections(self, targets):
        """
        Perform the bipartite matching between detections generated by another detector and targets.
        Later each prediction of the training model will be assigned to their respective prediction.
        """
        indices = []

        for i, tgt in enumerate(targets):
            tgt_bbox = tgt["boxes"]
            tgt_detections = tgt["detections"]

            cost_bbox = torch.cdist(tgt_detections, tgt_bbox, p=1)
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(tgt_detections),
                box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            cost_matrix = self.cost_bbox * cost_bbox \
                          + self.cost_giou * cost_giou

            indices_matched = linear_sum_assignment(cost_matrix.cpu())

            indices_matched = np.array([(i, j) for i, j in zip(indices_matched[0],
                                                               indices_matched[1]) \
                                        if cost_matrix[i, j] < self.max_cost]).reshape([-1, 2])


            indices += [(indices_matched[:, 0], indices_matched[:, 1])]

        return indices

    def calculate_single_query_prediction(self, outputs, targets):

        indices_detections2target = self.calculate_matching_detections(targets)
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            # [batch_size * num_queries, n_classes]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # [batch_size, num_queries, 4]
        # out_prob = out_prob.view(batch_size, num_queries, -1).cpu()

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
                      + self.cost_class * cost_class \
                      + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        submatrices = cost_matrix.split(sizes, -1)
        num_track_queries = num_queries - self.n_predictions
        for i, target_batch in enumerate(targets):
            indices_det, indices_targets = indices_detections2target[i]
            n_dets = len(indices_det)
            indices_predictions = num_track_queries + (indices_det % self.max_predictions) * self.n_assign
            indices_predictions2 = (indices_predictions[:, None] + \
                                    np.tile(np.arange(self.n_assign)[None],
                                            (n_dets, 1))).reshape(-1)

            indices_targets2 = np.tile(indices_targets[:, None],
                                       self.n_assign).reshape([-1])
            cost_batch = submatrices[i][i, indices_predictions2, indices_targets2]

            cost_batch = cost_batch.view(n_dets, self.n_assign)
            best_predictions = torch.argmin(cost_batch, dim=1) + indices_predictions

            if 'track_query_match_ids' in target_batch:
                tracklets_predictions = []
                tracklets_targets = []

                for j in range(num_track_queries):
                    if target_batch['track_queries_mask'][j]:
                        # Positive track detection
                        track_query_id = target_batch['track_query_match_ids'][j]
                        matched = indices_targets == track_query_id
                        index_matched = matched.nonzero()
                        if len(index_matched):
                            best_predictions[index_matched] = j

                        else:
                            tracklets_targets += [track_query_id]
                            tracklets_predictions += [j]

                best_predictions = torch.concatenate([best_predictions,
                                                      torch.tensor(tracklets_predictions, dtype=torch.int64)])
                indices_targets = np.concatenate([indices_targets, np.array(tracklets_targets, dtype=np.int32)])

            indices += [(best_predictions, indices_targets)]
        # print('indices: \n',indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "detections": Detections passed as input

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        indices_detections2target = self.calculate_matching_detections(targets)
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            # [batch_size * num_queries, n_classes]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # if self.focal_loss:
        #     neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
        #     pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
        #     cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        # else:
        #     # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
        #     # The 1 is a constant that doesn't change the matching, it can be ommitted.
        #     cost_class = -out_prob[:, tgt_ids]

        # out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # [batch_size, num_queries, 4]
        # out_prob = out_prob.view(batch_size, num_queries, -1).cpu()

        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        #
        # # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(
        #     box_cxcywh_to_xyxy(out_bbox),
        #     box_cxcywh_to_xyxy(tgt_bbox))
        #
        # # Final cost matrix
        # cost_matrix = self.cost_bbox * cost_bbox \
        #               + self.cost_class * cost_class \
        #               + self.cost_giou * cost_giou
        # cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # sizes = [len(v["boxes"]) for v in targets]
        indices = []
        # submatrices = cost_matrix.split(sizes, -1)
        num_track_queries = num_queries - self.n_predictions
        for i, target_batch in enumerate(targets):
            indices_det, indices_targets = indices_detections2target[i]
            n_dets = len(indices_det)
            indices_predictions = num_track_queries + (indices_det % self.max_predictions) * self.n_assign
            indices_predictions2 = (indices_predictions[:, None] + \
                                    np.tile(np.arange(self.n_assign)[None],
                                            (n_dets, 1))).reshape(-1)

            indices_targets2 = np.tile(indices_targets[:, None],
                                       self.n_assign).reshape([-1])
            # cost_batch = submatrices[i][i, indices_predictions2, indices_targets2]
            #
            # cost_batch = cost_batch.view(n_dets, self.n_assign)
            # best_predictions = torch.argmin(cost_batch, dim=1) + indices_predictions

            if 'track_query_match_ids' in target_batch:
                tracklets_predictions = []
                tracklets_targets = []

                for j in range(num_track_queries):
                    if target_batch['track_queries_mask'][j]:
                        # Positive track detection
                        track_query_id = target_batch['track_query_match_ids'][j]
                        matched = indices_targets2 == track_query_id
                        index_matched = matched.nonzero()
                        if len(index_matched):
                            indices_predictions2[index_matched] = j

                        else:
                            tracklets_targets += [track_query_id]
                            tracklets_predictions += [j]

                indices_predictions2 = np.concatenate([indices_predictions2,np.array(tracklets_predictions, dtype=np.int32)])
                indices_targets2 = np.concatenate([indices_targets2, np.array(tracklets_targets, dtype=np.int32)])

            indices += [(indices_predictions2, indices_targets2)]
        # print('indices: \n',indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]




def build_matcher(args):
    if args.used_ordered_queries:
        return OrderDetectionsMatcher(
            args.num_queries,
            args.num_queries//60,
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_loss=args.focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,)
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,)
