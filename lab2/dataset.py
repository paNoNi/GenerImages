import os
from typing import List
from xml.dom import minidom

import numpy as np
import torch
from PIL import Image
from mat4py import loadmat
from torchvision import transforms


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.root = root


class CarPlates(DatasetBase):
    def __init__(self, root: str):
        super().__init__(root)
        self.annos_path = os.path.join(root, 'annotations')
        self.annos = os.listdir(self.annos_path)
        self.imgs_path = os.path.join(root, 'images')

    def __getitem__(self, idx):
        anno = minidom.parse(os.path.join(self.annos_path, self.annos[idx]))
        anno = anno.getElementsByTagName('annotation')[0]
        img_name = anno.getElementsByTagName('filename')[0].firstChild.data
        img = Image.open(os.path.join(self.imgs_path, img_name)).convert('RGB')
        bndbox = anno.getElementsByTagName('object')[0].getElementsByTagName('bndbox')[0]
        boxes = list()
        labels = list()

        box = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data), \
              int(bndbox.getElementsByTagName('ymin')[0].firstChild.data), \
              int(bndbox.getElementsByTagName('xmax')[0].firstChild.data), \
              int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        boxes.append(box)
        labels.append(1)

        target = dict()
        target['boxes'] = torch.tensor(data=boxes, dtype=torch.float64)
        target['labels'] = torch.tensor(data=labels, dtype=torch.int64)
        img = transforms.PILToTensor()(img) / 255.
        return img, target

    def __len__(self):
        return len(self.annos)


class PlatesDataset(DatasetBase):
    def __init__(self, root):
        super().__init__(root)
        # load all image files, sorting them to ensure that they are aligned
        self.names = []
        self.anno = loadmat(os.path.join(self.root, 'digitStruct2.mat'))['digitStruct']

    def __getitem__(self, idx):

        img_name = self.anno['name'][idx]
        bounding_boxes = self.anno['bbox'][idx]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        labels = list()
        boxes = list()

        if not isinstance(bounding_boxes['label'], list):
            bounding_boxes['label'] = [bounding_boxes['label']]
            bounding_boxes['left'] = [bounding_boxes['left']]
            bounding_boxes['top'] = [bounding_boxes['top']]
            bounding_boxes['width'] = [bounding_boxes['width']]
            bounding_boxes['height'] = [bounding_boxes['height']]

        count_boxes = len(bounding_boxes['label'])
        for i in range(count_boxes):
            box = bounding_boxes['left'][i], \
                  bounding_boxes['top'][i], \
                  bounding_boxes['left'][i] + bounding_boxes['width'][i], \
                  (bounding_boxes['top'][i] + bounding_boxes['height'][i])
            boxes.append(box)

        boxes = np.array(boxes)
        one_box = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
        one_box = np.array(one_box).reshape((1, -1))
        labels.append(1)

        target = dict()
        target['boxes'] = torch.tensor(data=one_box, dtype=torch.float64)
        target['labels'] = torch.tensor(data=labels, dtype=torch.int64)
        img = transforms.PILToTensor()(img) / 255.

        return img, target

    def __len__(self):
        return len(self.anno['name'])


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[DatasetBase]):
        self.__datasets = datasets
        self.__lens = [len(dataset) for dataset in self.__datasets]

    def __getitem__(self, idx):
        current_len = 0
        dataset = None
        dataset_idx = idx
        for i in range(self.__len__()):
            current_len += self.__lens[i]
            if idx < current_len:
                dataset = self.__datasets[i]
                break
            dataset_idx -= self.__lens[i]

        return dataset.__getitem__(dataset_idx)

    @property
    def dataset_count(self):
        return len(self.__datasets)

    def dataset_path(self, idx):
        return self.__datasets[idx].data_path

    def __len__(self):
        return sum(self.__lens)
