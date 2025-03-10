# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import os
import csv
import copy

import numpy as np
import torch
import bisect
import random
from pathlib import Path

from pycocotools.coco import COCO


from .coco import build as build_coco
from .crowdhuman import build_crowdhuman
from .coco import CocoDetection, make_coco_transforms
from .kinematic_utils import make_kine_transforms, ConvertCocoAnnsToTrack, DetectionsEncoderSine, NormalizeDetections

# import trackformer.datasets.transforms as T



class MOT(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(MOT, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}

        img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

        if self._prev_frame:
            frame_id = self.coco.imgs[idx]['frame_id']

            # PREV
            # first frame has no previous frame
            prev_frame_id = random.randint(
                max(0, frame_id - self._prev_frame_range),
                min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1))
            prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + prev_frame_id

            prev_img, prev_target = self._getitem_from_id(prev_image_id, random_state)
            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV frame equidistant as prev_frame
                prev_prev_frame_id = min(max(0, prev_frame_id + prev_frame_id - frame_id), self.seq_length(idx) - 1)
                prev_prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + prev_prev_frame_id

                prev_prev_img, prev_prev_target = self._getitem_from_id(prev_prev_image_id, random_state)
                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class MOT_Kine(CocoDetection):
    def __init__(self, path_images: str, path_ann_file: str, path_detections: str, transforms, norm_transforms=None,
                 overflow_boxes=False, remove_no_obj_imgs=False, min_num_objects=0, prev_frame_range=1,
                 use_classes=False):
        super(MOT_Kine, self).__init__(path_images, path_ann_file, transforms, norm_transforms=norm_transforms,
                                       overflow_boxes=overflow_boxes, remove_no_obj_imgs=remove_no_obj_imgs,
                                       min_num_objects=min_num_objects)

        # self._transforms = transforms
        # self._norm_transforms = norm_transforms
        # self.overflow_boxes = overflow_boxes
        self.prepare = ConvertCocoAnnsToTrack(overflow_boxes)

        # self.coco = COCO(path_ann_file)
        # self.path_images = path_images

        # self.ids = list(sorted(self.coco.imgs.keys()))

        # annos_image_ids = [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
        # if remove_no_obj_imgs:
        #     self.ids = sorted(list(set(annos_image_ids)))

        # if min_num_objects:
        #     counter = Counter(annos_image_ids)
        #     self.ids = [i for i in self.ids if counter[i] >= min_num_objects]

        # self.images = [self.coco.imgs[i]['file_name'] for i in range(len(self.coco.imgs))]
        # self._path_images = [os.path.join(path_img) for path_img in self.images]
        self.prev_frame_range = prev_frame_range
        self._prev_frame = prev_frame_range > 0
        self._prev_prev_frame = prev_frame_range > 1
        self.path_detections = path_detections
        self.sequence_ids = {seq: [] for seq in self.sequences}
        self.sequence_dims = []
        for id_img in self.ids:
            image = self.coco.dataset['images'][id_img]
            seq_name = image['file_name'].split('_')[0]
            self.sequence_ids[seq_name] += [image['id']]
            self.sequence_dims += [(self.coco.dataset['images'][id_img]['width'],
                                    self.coco.dataset['images'][id_img]['height'])]

        self.sequences_length = {seq: self.coco.dataset['images'][self.sequence_ids[seq][0]]['seq_length'] for seq in
                                 self.sequences}
        self.sequences_frame_ids = []
        last_value = 0
        for value in self.sequences_length.values():
            self.sequences_frame_ids += [last_value]
            last_value += value

        self.detections_coco = COCO(path_detections)
        if use_classes:
            self._load_detection = self.get_detection_with_class
        else:
            self._load_detection = self.get_detection_without_class


    def _get_samples_from_annotations(self, coco_anns, idx):
        return coco_anns.loadAnns(coco_anns.getAnnIds(idx))

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    # def __len__(self):
    #     return len(self.coco.imgs)

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def get_detection_with_class(self, idx):
        detections = self._get_samples_from_annotations(self.detections_coco, idx)
        bboxes = []
        for det in detections:
            bboxes += [det['bbox'] + [det['confidence'], det['category_id']]]
        bboxes = torch.concatenate(bboxes, dtype=torch.float32).reshape([-1, 6])
        return bboxes

    def get_detection_without_class(self, idx):
        detections = self._get_samples_from_annotations(self.detections_coco, idx)
        bboxes = []
        for det in detections:
            bboxes += [det['bbox'] + [det['confidence']]]
        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape([-1, 5])
        return bboxes

    def get_target(self, idx):
        return self._get_samples_from_annotations(self.coco, idx)

    def get_id_prev_frames(self, idx:int):
        """
        Get valid previous frames from index idx. Valid previous frames must be contained in the same video. If a video
        starts at frame 2000, the previous valid frames for idx=2001 will be [2000,2000,2000,2000,2000]
        @param idx: index of current frame
        @return: Reuturn array of increasing values that include the previous frames:
                    Ex: idx = 2112
                    return = np.array([2107 2108 2109 2110 2111])

        """
        if not (idx in self.ids):
            if idx < 0:
                return self.get_id_prev_frames(self.ids[-1])
            print('Failed index {}, trying lower value'.format(idx))
            return self.get_id_prev_frames(idx - 1)

        if idx in self.sequences_frame_ids:
            return [idx] * self.prev_frame_range

        id_check = -1
        for seq_id in self.sequences_frame_ids:
            if seq_id > idx:
                break
            id_check = seq_id

        prev_ids = np.arange(- self.prev_frame_range, 0) + idx
        return np.maximum(prev_ids, id_check)


    def __getitem__(self, idx):

        # random_state = {
        #     'random': random.getstate(),
        #     'torch': torch.random.get_rng_state()}
        #
        # img, target = self._getitem_from_id(idx,random_state)
        # if idx == 147: # DELETE
        #     print('found')
        # img = self._load_image(idx)
        target = self._load_target(idx)
        detections = self._load_detection(idx)
        dims = self.sequence_dims[idx]

        image_id = self.ids[idx]
        target = {'image_id': image_id,
                  'annotations': target}

        prev_targets = []
        idx_data = self.get_id_prev_frames(idx)

        for i in idx_data:
            prev_targets += [self._load_target(i)]

        detections, target = self.prepare(dims, detections, target, prev_targets)

        if self._transforms is not None:
            detections, target = self._transforms(detections, target)

        detections, target = self._norm_transforms(detections, target)
        return detections, target

    def write_result_files(self, results, output_dir, threshold=0.7):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= threshold:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class MOT_Kine2(CocoDetection):
    def __init__(self, path_images: str, path_ann_file: str, path_detections: str, transforms, norm_transforms=None,
                 overflow_boxes=False, remove_no_obj_imgs=False, min_num_objects=0, prev_frame_range=1,
                 use_classes=False):
        super(MOT_Kine2, self).__init__(path_images, path_ann_file, transforms, norm_transforms=norm_transforms,
                                       overflow_boxes=overflow_boxes, remove_no_obj_imgs=remove_no_obj_imgs,
                                       min_num_objects=min_num_objects)

        self.prepare = ConvertCocoAnnsToTrack(overflow_boxes)
        self.prev_frame_range = prev_frame_range
        self._prev_frame = prev_frame_range > 0
        self._prev_prev_frame = prev_frame_range > 1
        self.path_detections = path_detections
        self.sequence_ids = {seq: [] for seq in self.sequences}
        self.sequence_dims = []
        for id_img in self.ids:
            image = self.coco.dataset['images'][id_img]
            seq_name = image['file_name'].split('_')[0]
            self.sequence_ids[seq_name] += [image['id']]
            self.sequence_dims += [(self.coco.dataset['images'][id_img]['width'],
                                    self.coco.dataset['images'][id_img]['height'])]

        self.sequences_length = {seq: self.coco.dataset['images'][self.sequence_ids[seq][0]]['seq_length'] for seq in
                                 self.sequences}
        self.sequences_frame_ids = []
        last_value = 0
        for value in self.sequences_length.values():
            self.sequences_frame_ids += [last_value]
            last_value += value

        self.detections_coco = COCO(path_detections)
        if use_classes:
            self._load_detection = self.get_detection_with_class
        else:
            self._load_detection = self.get_detection_without_class


    def _get_samples_from_annotations(self, coco_anns, idx):
        return coco_anns.loadAnns(coco_anns.getAnnIds(idx))

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    # def __len__(self):
    #     return len(self.coco.imgs)

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def get_detection_with_class(self, idx):
        detections = self._get_samples_from_annotations(self.detections_coco, idx)
        bboxes = []
        metadata = []
        for det in detections:
            bboxes += [det['bbox']]
            metadata += [[det['confidence'], det['category_id']]]
        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape([-1, 4])
        metadata = torch.tensor(metadata, dtype=torch.float32).reshape([-1, 2])
        return bboxes, metadata

    def get_detection_without_class(self, idx):
        detections = self._get_samples_from_annotations(self.detections_coco, idx)
        bboxes = []
        metadata = []
        for det in detections:
            bboxes += [det['bbox']]
            metadata += [det['confidence']]
        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape([-1, 4])
        metadata = torch.tensor(metadata, dtype=torch.float32)[:,None]

        return bboxes, metadata

    def get_target(self, idx):
        return self._get_samples_from_annotations(self.coco, idx)

    def get_id_prev_frames(self, idx:int):
        """
        Get valid previous frames from index idx. Valid previous frames must be contained in the same video. If a video
        starts at frame 2000, the previous valid frames for idx=2001 will be [2000,2000,2000,2000,2000]
        @param idx: index of current frame
        @return: Return array of increasing values that include the previous frames:
                    Ex: idx = 2112
                    return = np.array([2107 2108 2109 2110 2111])

        """
        if not (idx in self.ids):
            if idx < 0:
                return self.get_id_prev_frames(self.ids[-1])
            print('Failed index {}, trying lower value'.format(idx))
            return self.get_id_prev_frames(idx - 1)

        if idx in self.sequences_frame_ids:
            return [idx] * self.prev_frame_range

        id_check = -1
        for seq_id in self.sequences_frame_ids:
            if seq_id > idx:
                break
            id_check = seq_id

        prev_ids = np.arange(- self.prev_frame_range, 0) + idx
        return np.maximum(prev_ids, id_check)


    def __getitem__(self, idx):

        # random_state = {
        #     'random': random.getstate(),
        #     'torch': torch.random.get_rng_state()}
        target = self._load_target(idx)
        positions, metadata = self._load_detection(idx)

        dims = self.sequence_dims[idx]

        image_id = self.ids[idx]
        target = {'image_id': image_id,
                  'annotations': target}

        prev_targets = []
        idx_data = self.get_id_prev_frames(idx)

        for i in idx_data:
            prev_targets += [self._load_target(i)]

        positions, target2 = self.prepare(dims, positions, target, prev_targets)

        if self._transforms is not None:
            positions, target2 = self._transforms(positions, target2)

        detections, target3 = self._norm_transforms(positions, target2)
        target3['detections'] = detections
        target3['detections_metadata'] = metadata
        return detections, metadata, target3

    def write_result_files(self, results, output_dir, threshold=0.7):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= threshold:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])


def build_mot(image_set, args):
    if image_set == 'train':
        root = Path(args.mot_path_train)
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range = args.track_prev_frame_range
    elif image_set == 'val':
        root = Path(args.mot_path_val)
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided MOT17Det path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes)

    dataset = MOT(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
    )

    return dataset


def build_mot_kine(image_set, args):
    if image_set == 'train':
        root = Path(args.mot_path_train)
        # prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range = args.track_prev_frame_range
    elif image_set == 'val':
        root = Path(args.mot_path_val)
        # prev_frame_rnd_augs = 0.0
        prev_frame_range = args.track_prev_frame_range
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided MOT17Det path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"
    detections_file = root / f"annotations/{split.replace('coco', 'det')}.json"
    # transforms, norm_transforms = make_coco_transforms(
    #     image_set, args.img_transform, args.overflow_boxes)
    # if args.use_encoding_dets:
    transforms, norm_transforms = make_kine_transforms(image_set, overflow_boxes=args.overflow_boxes,
                                                       use_sin_encoding=args.use_encoding_dets,
                                                       dim_encoding=args.encoding_dim_detections)
    # else:
    #     transforms, norm_transforms = make_kine_transforms(image_set, overflow_boxes=args.overflow_boxes,
    #                                                        use_sin_encoding=True)

    dataset = MOT_Kine2(
        img_folder, ann_file, detections_file, transforms,
        norm_transforms=norm_transforms,
        prev_frame_range=prev_frame_range,
        # return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        # prev_frame_rnd_augs=prev_frame_rnd_augs,
        # prev_prev_frame=args.track_prev_prev_frame,
    )

    return dataset


def build_mot_crowdhuman(image_set, args):
    if image_set == 'train':
        args_crowdhuman = copy.deepcopy(args)
        args_crowdhuman.train_split = args.crowdhuman_train_split

        crowdhuman_dataset = build_crowdhuman('train', args_crowdhuman)

        if getattr(args, f"{image_set}_split") is None:
            return crowdhuman_dataset

    dataset = build_mot(image_set, args)

    if image_set == 'train':
        dataset = torch.utils.data.ConcatDataset(
            [dataset, crowdhuman_dataset])

    return dataset


def build_mot_coco_person(image_set, args):
    if image_set == 'train':
        args_coco_person = copy.deepcopy(args)
        args_coco_person.train_split = args.coco_person_train_split

        coco_person_dataset = build_coco('train', args_coco_person, 'person_keypoints')

        if getattr(args, f"{image_set}_split") is None:
            return coco_person_dataset

    dataset = build_mot(image_set, args)

    if image_set == 'train':
        dataset = torch.utils.data.ConcatDataset(
            [dataset, coco_person_dataset])

    return dataset
