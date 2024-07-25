import math
import torch
import random

from . import transforms as T


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def hflip_det(det, target):
    width = target["orig_size"][1]
    # n_samples = det.size()[0]
    detection_meta_data = det[:, 4:]
    flipped_det = det[:, [2, 1, 0, 3]] \
                  * torch.as_tensor([-1, 1, -1, 1]) \
                  + torch.as_tensor([width, 0, width, 0])

    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] \
                * torch.as_tensor([-1, 1, -1, 1]) \
                + torch.as_tensor([width, 0, width, 0])
        target["boxes"] = boxes

    if "tracklets" in target:
        tracklets = target["tracklets"]
        tracklets = tracklets[:, :, [2, 1, 0, 3]] \
                    * torch.as_tensor([-1, 1, -1, 1])[None] \
                    + torch.as_tensor([width, 0, width, 0])[None]
        target["tracklets"] = tracklets

    if "boxes_ignore" in target:
        boxes = target["boxes_ignore"]
        boxes = boxes[:, [2, 1, 0, 3]] \
                * torch.as_tensor([-1, 1, -1, 1]) \
                + torch.as_tensor([width, 0, width, 0])
        target["boxes_ignore"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return torch.cat([flipped_det, detection_meta_data], dim=1), target

class RandomHorizontalFlipDet:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, det, target):
        if random.random() < self.p:
            return hflip_det(det, target)
        return det, target


def add_noise_tracklets(tracklets, width, height, noise_range=[20, 30]):
    n_frames, n_dets, det_dim = tracklets.size()
    noise_values_width = torch.randint(-noise_range[0], noise_range[0], (n_frames, n_dets, det_dim // 2))
    noise_values_height = torch.randint(-noise_range[0], noise_range[1], (n_frames, n_dets, det_dim // 2))
    new_tracklets = tracklets.clone()
    new_tracklets[:, :, 0::2] += noise_values_width
    new_tracklets[:, :, 1::2] += noise_values_height
    new_tracklets[:, :, 0::2] = torch.clamp_(new_tracklets[:, :, 0::2], 0, width)
    new_tracklets[:, :, 1::2] = torch.clamp_(new_tracklets[:, :, 1::2], 0, height)
    new_tracklets[:, :, 2] = torch.clamp_(new_tracklets[:, :, 2], new_tracklets[:, :, 0] + 5, width)
    new_tracklets[:, :, 3] = torch.clamp_(new_tracklets[:, :, 3], new_tracklets[:, :, 1] + 5, height)
    # new_tracklets[:, :, :2] = torch.clamp_(new_tracklets[:, :, :2], 0)
    return new_tracklets


class RandomNoiseTracklets:
    def __init__(self, noise_range=[20, 30], prob_noise=0.1):
        self.noise_range = noise_range
        self.prob_noise = prob_noise

    def __call__(self, det, target):
        if len(target['boxes']) > 0:
            if random.random() < self.prob_noise:
                height, width = target['orig_size'][0], target['orig_size'][1]
                target['tracklets'] = add_noise_tracklets(target['tracklets'], width, height, self.noise_range)
        return det, target


class NormalizeTarget:
    def __init__(self,overflow_boxes):
        self.overflow_boxes = overflow_boxes

    def __call__(self, detections: torch.Tensor, target: dict):
        target = target.copy()
        h, w = target['orig_size'][:2]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "tracklets" in target:
            tracklets = target["tracklets"]
            tracklets[:, :, :4] = box_xyxy_to_cxcywh(tracklets[:, :, :4])
            tracklets[:, :, :4] = tracklets[:, :, :4] / torch.tensor([w, h, w, h], dtype=torch.float32)[None]
            if not self.overflow_boxes:
                tracklets = torch.clamp_(tracklets, 0, 1)
            target["tracklets"] = tracklets

        return detections, target


class NormalizeDetections:
    def __init__(self,overflow_boxes):
        self.overflow_boxes = overflow_boxes

    def __call__(self, detections: torch.Tensor, target: dict = None):
        h, w = target['orig_size'][:2]
        detections[:, :4] = box_xyxy_to_cxcywh(detections[:, :4])
        detections[:, :4] = detections[:, :4] / torch.tensor([w, h, w, h], dtype=torch.float32)
        if not self.overflow_boxes:
            detections[:, :5] = torch.clamp_(detections[:, :5], 0, 1)
        target['detections'] = detections
        return detections, target


class DetectionsEncoderSine:
    """
    Position embeding for detection locations (range between [0,1]) very similar to the one
    used by the Attention is all you need paper, applied to detections [N_det, 4]
    """

    def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        # self.normalize = normalize
        # if scale is not None and normalize is False:
        #     raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def __call__(self, x: torch.Tensor, target: dict):
        n_samples = x.size()[0]
        detections = x[:, :4]
        detection_meta_data = x[:, 4:].reshape([n_samples, -1])
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        freq = (detections[:, :, None] * torch.pi * 2) / dim_t
        embed_coord = torch.concatenate([freq[:, :, 0::2].cos(), freq[:, :, 1::2].sin()], dim=2)
        embed_coord = embed_coord.flatten(1)
        encoding_detections = torch.cat([embed_coord, detection_meta_data], dim=1)
        return encoding_detections, target

class IdentityDetectionEncoder:
    """
    No encoding for detection locations (range between [0,1])
    """

    # def __init__(self, ):
    #     pass

    def __call__(self, x: torch.Tensor, target: dict):
        return x, target


def make_kine_transforms(image_set, prob_noise_pos=0.1, overflow_boxes=False, use_sin_encoding=True,dim_encoding=32):
    if use_sin_encoding:
        norm_transforms = T.Compose([
            NormalizeTarget(overflow_boxes),
            NormalizeDetections(overflow_boxes),
            DetectionsEncoderSine(dim_encoding),
        ])
    else:
        norm_transforms = T.Compose([
            NormalizeTarget(overflow_boxes),
            NormalizeDetections(overflow_boxes),
        ])
    # max_size = 1333
    # val_width = 800
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # random_resizes = [400, 500, 600]
    # random_size_crop = (384, 600)

    # if img_transform is not None:
    #     scale = img_transform.max_size / max_size
    #     max_size = img_transform.max_size
    #     val_width = img_transform.val_width
    #
    #     # scale all with respect to custom max_size
    #     scales = [int(scale * s) for s in scales]
    #     random_resizes = [int(scale * s) for s in random_resizes]
    #     random_size_crop = [int(scale * s) for s in random_size_crop]

    if image_set == 'train':
        transforms = [
            RandomHorizontalFlipDet(),
            RandomNoiseTracklets(prob_noise=prob_noise_pos),
            #     # T.RandomResize(scales, max_size=max_size),
        ]

    elif image_set == 'val':
        # transforms = [
        #     T.RandomResize([val_width], max_size=max_size),
        # ]
        return None, norm_transforms
    else:
        ValueError(f'Unknown Image set value: {image_set}')

    return T.Compose(transforms), norm_transforms


def get_tracklet_data(target: dict, past_frames: list):
    """
    Function to generate past locations of objects (tracklets) of a tracked object

    @param target:
    @param past_frames:
    @return:
    """
    if target['boxes'].size()[0] == 0:
        return torch.empty([len(past_frames), 0, 4])

    id_track = target['track_ids']
    id_to_row = {idx.item(): i for i, idx in enumerate(id_track)}
    boxes = target['boxes']
    past_boxes = torch.zeros([len(past_frames), boxes.size()[0], 4], dtype=torch.float32)
    past_boxes += boxes[None]
    for i, frame_target in enumerate(past_frames):
        # list_ids = [i for i in range(target['boxes'].size()[0])]
        # present_ids = []
        for ann in frame_target:
            if ann['track_id'] in id_track:
                id_box = id_to_row[ann['track_id']]
                past_boxes[i, id_box, :] = torch.Tensor(ann['bbox'])
                past_boxes[i, id_box, 2:4] += past_boxes[i, id_box, :2]
                # present_ids += [id_box]

        # present_ids = set(present_ids)
        # list_ids = set(list_ids)
        # absent_list_ids = list(list_ids - present_ids)
        # for absent_id in absent_list_ids:  # Copy values of objects not yet present
        #     past_boxes[i, absent_id, :] = boxes[absent_id, :]

    # past_boxes[:, :, 2:4] += past_boxes[:, :, :2]
    return past_boxes


class ConvertCocoAnnsToTrack(object):
    def __init__(self, overflow_boxes=False):
        self.overflow_boxes = overflow_boxes

    def __call__(self, img, detections, target, prev_anns):

        assert (len(prev_anns) > 1), 'Invalid Number of targets. At least 2 frames of data are required.'
        # n_det = detections.size[0]
        w, h = img.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # x,y,w,h --> x,y,x,y
        boxes[:, 2:] += boxes[:, :2]
        detections[:, 2:4] += detections[:, :2]
        if not self.overflow_boxes:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            detections[:, 0:4:2].clamp_(min=0, max=w)
            detections[:, 1:4:2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]

        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes - 1

        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if anno and "track_id" in anno[0]:
            track_ids = torch.tensor([obj["track_id"] for obj in anno])
            target["track_ids"] = track_ids[keep]
        elif not len(boxes):
            target["track_ids"] = torch.empty(0)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        ignore = torch.tensor([obj["ignore"] if "ignore" in obj else 0 for obj in anno])

        target["area"] = area[keep]
        # target["heigth"] = h
        # target["width"] = w
        target["iscrowd"] = iscrowd[keep]
        target["ignore"] = ignore[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['tracklets'] = get_tracklet_data(target, prev_anns)

        return detections, target
