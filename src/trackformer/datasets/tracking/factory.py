# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union
from argparse import Namespace
from torch.utils.data import ConcatDataset

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper
from ..mot import MOT_Kine2
from ..kinematic_utils import DetectionsEncoderSine
from ..mot import build_mot_kine

DATASETS = {}

# Fill all available datasets, change here to modify / add new datasets.
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08']:
    name = f'MOT20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOT20Wrapper(split, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])


class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


def get_sub_dataset(name_seq:str ,image_set:str, detect_args:Namespace):
    """
    Create a sub dataset by filtering samples that are not from the selected sequence.
    @param name_seq: Na
    @param image_set:
    @param detect_args:
    @return:
    """
    dataset = build_mot_kine(image_set, detect_args)
    new_images = []
    for id_img, label_img in dataset.coco.imgs.items():
        if name_seq in label_img['file_name']:
            new_label = label_img.copy()
            new_label['id'] = new_label['frame_id']
            new_images += [new_label]
    new_ids = [img_ann['frame_id'] for img_ann in new_images]
    new_anns = []
    id_count = 0
    for i, ann_obj in dataset.coco.anns.items():
        if ann_obj['seq'] == name_seq:
            new_ann_obj = ann_obj.copy()
            new_ann_obj['id'] = id_count

            # new_imgToAnns[ann_obj['image_id']].append(new_ann_obj)
            new_anns += [new_ann_obj]
            id_count += 1
    new_sequences = [name_seq]

    dataset.coco.dataset['annotations'] = new_anns
    dataset.coco.dataset['images'] = new_images
    # dataset.coco.dataset['categories'] = new_categories
    dataset.coco.dataset['sequences'] = new_sequences
    dataset.coco.createIndex()
    dataset.ids = new_ids
    return dataset

def get_sub_sequences(names_seq, detect_args, image_set='val', **kwargs):

    datasets = []
    for seq in names_seq:
        datasets += [get_sub_dataset(seq, image_set, detect_args)]

    return datasets


class TrackDatasetFactoryKinet:
    """A central class to manage the individual dataset loaders for kinetic format of dataset.

       This class contains the datasets. Once initialized the individual parts (e.g. sequences)
       can be accessed.
       """
    def __init__(self,  sequence_names: Union[str, list], image_set:str ,detect_args: Namespace,  **kwargs):
        if isinstance(sequence_names, str):
            sequence_names = [sequence_names]
        self._data = self.get_datasets(sequence_names,detect_args, image_set,  **kwargs)

    def get_datasets(self, sequence_names, detect_args, image_set, **kwargs):
        # datasets_seq = []

        datasets_seq = get_sub_sequences(sequence_names,detect_args, image_set=image_set, **kwargs)
        return datasets_seq

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]