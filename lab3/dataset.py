import json
import os

import cv2
import numpy as np
import torch
from PIL import Image


class SignsDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms):
        self.__root = root
        self.__anno = self.__get_anno(root)
        self.__keys = list(self.__anno.keys())
        self.transforms = transforms
        self.skipped = 0

    def __get_anno(self, root: str) -> dict:
        with open(os.path.join(root, 'via_region_data.json'), 'r') as fp:
            anno = json.load(fp)
        return anno

    def __getitem__(self, idx):
        while True:
            cur_idx = int((idx + self.skipped) % self.__len__())
            # load images and masks
            solo_anno = self.__anno[self.__keys[cur_idx]]
            img_path = os.path.join(self.__root, solo_anno['filename'])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background

            boxes = list()
            regions = solo_anno['regions']
            masks = list()
            for i_region in list(regions.keys()):
                # if 'name' not in list(regions[i_region]['region_attributes'].keys()):
                #     continue
                shape_attrs = regions[i_region]['shape_attributes']
                new_mask = np.zeros((img.size[1], img.size[0], 3))
                if shape_attrs['name'] == 'ellipse':
                    points = shape_attrs['cx'], shape_attrs['cy'], shape_attrs['rx'], shape_attrs['ry']
                    min_x = points[0] - points[2]
                    min_y = points[1] - points[3]
                    max_x = points[0] + points[2]
                    max_y = points[1] + points[3]
                    box = np.array([int(min_x), int(min_y), int(max_x), int(max_y)])
                    new_mask = self.__ellipse_mask(new_mask, points=points)
                elif shape_attrs['name'] == 'rect':
                    points = int(shape_attrs['x']), int(shape_attrs['y']), int(shape_attrs['width']), int(
                        shape_attrs['height'])
                    min_xy = points[:2]
                    max_xy = points[0] + points[2], points[1] + points[3]
                    box = np.array([*min_xy, *max_xy])
                    new_mask = self.__rect_mask(new_mask, points)
                elif shape_attrs['name'] == 'circle':
                    center = (int(shape_attrs['cx']), int(shape_attrs['cy']))
                    radius = int(shape_attrs['r'])
                    min_x = center[0] - radius
                    min_y = center[1] - radius
                    max_x = center[0] + radius
                    max_y = center[1] + radius
                    box = np.array([min_x, min_y, max_x, max_y])
                    new_mask = cv2.circle(new_mask, center=center, radius=radius, color=(255, 255, 255), thickness=-1)
                else:
                    all_points_x = shape_attrs['all_points_x']
                    all_points_x = [int(point) for point in all_points_x]
                    min_x = np.min(np.array(all_points_x))
                    max_x = np.max(np.array(all_points_x))
                    all_points_y = shape_attrs['all_points_y']
                    all_points_y = [int(point) for point in all_points_y]
                    min_y = np.min(np.array(all_points_y))
                    max_y = np.max(np.array(all_points_y))
                    pts = list(zip(all_points_x, all_points_y))
                    box = np.array([min_x, min_y, max_x, max_y])
                    new_mask = cv2.fillPoly(new_mask, pts=np.array([pts]), color=(255, 255, 255))

                if (box[3] - box[1]) > 5 * (box[2] - box[0]):
                    continue
                masks.append(new_mask[:, :, 0])
                boxes.append(box)

            # masks = np.array(masks)
            # masks = torch.as_tensor(masks)
            # masks = torch.sum(masks, dim=0)
            # print(masks.shape)
            # print(masks)

            image_id = torch.tensor([cur_idx])

            if len(boxes) == 0:
                self.skipped += 1
                continue

            boxes = np.array(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = np.array(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            masks[masks > 0] = torch.tensor(1, dtype=torch.uint8)
            labels = torch.ones((masks.shape[0],), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

    def __ellipse_mask(self, obj_mask: np.ndarray, points: tuple):
        mask = np.zeros_like(obj_mask)
        points = [int(point) for point in points]
        mask = cv2.ellipse(mask, points[:2], points[2:], 0.0, 0.0, 360.0, (255, 255, 255), -1)
        return mask

    def __rect_mask(self, obj_mask: np.ndarray, points: tuple):
        pts = [[points[0], points[1]],
               [points[0] + points[2], points[1]],
               [points[0] + points[2], points[1] + points[3]],
               [points[0], points[1] + points[3]]]
        pts = np.array([pts])
        return cv2.fillPoly(obj_mask, pts, (255, 255, 255))

    def __len__(self):
        return len(self.__keys)

# class SignsDataset(torch.utils.data.Dataset):
#     def __init__(self, root: str, transforms):
#         self.__root = root
#         self.__anno = self.__get_anno(root)
#         self.__keys = list(self.__anno.keys())
#         self.transforms = transforms
#
#     def __get_anno(self, root: str) -> dict:
#         with open(os.path.join(root, 'via_region_data.json'), 'r') as fp:
#             anno = json.load(fp)
#         return anno
#
#     def __getitem__(self, idx):
#         # load images and masks
#         solo_anno = self.__anno[self.__keys[idx]]
#         img_path = os.path.join(self.__root, solo_anno['filename'])
#         img = Image.open(img_path).convert("RGB")
#         # note that we haven't converted the mask to RGB,
#         # because each color corresponds to a different instance
#         # with 0 being background
#
#         obj_mask = np.zeros((img.size[1], img.size[0], 3))
#         regions = solo_anno['regions']
#         print(regions)
#         for i_region in list(regions.keys()):
#             shape_attrs = regions[i_region]['shape_attributes']
#             if shape_attrs['name'] == 'ellipse':
#                 points = shape_attrs['cx'], shape_attrs['cy'], shape_attrs['rx'], shape_attrs['ry']
#                 obj_mask = self.__ellipse_mask(obj_mask, points=points)
#             elif shape_attrs['name'] == 'rect':
#                 points = int(shape_attrs['x']), int(shape_attrs['y']), int(shape_attrs['width']), int(shape_attrs['height'])
#                 obj_mask = self.__rect_mask(obj_mask, points)
#             else:
#                 all_points_x = shape_attrs['all_points_x']
#                 all_points_x = [int(point) for point in all_points_x]
#                 all_points_y = shape_attrs['all_points_y']
#                 all_points_y = [int(point) for point in all_points_y]
#                 pts = list(zip(all_points_x, all_points_y))
#                 obj_mask = cv2.fillPoly(obj_mask, pts=np.array([pts]), color=(255, 255, 255))
#
#         cv2.imshow('Ellipse', obj_mask)
#         cv2.waitKey(10_000)
#         cv2.destroyAllWindows()
#
#         obj_mask = obj_mask == 255
#         obj_mask = obj_mask[:, :, 0]
#         zero_mask = np.zeros_like(obj_mask)
#         zero_mask[obj_mask] = 1
#         obj_mask = zero_mask
#
#         # convert everything into a torch.Tensor
#         # there is only one class
#         num_classes = 2
#         labels = torch.tensor([0, 1], dtype=torch.int64)
#         masks = np.zeros(shape=(num_classes, img.size[1], img.size[0]))
#         masks[0] = 1
#         masks[0, obj_mask] = 0
#         masks[1] = obj_mask
#
#         masks = torch.as_tensor(masks, dtype=torch.uint8)
#
#         image_id = torch.tensor([idx])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_classes,), dtype=torch.int64)
#
#         target = {}
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["iscrowd"] = iscrowd
#         target["boxes"] = torch.tensor(dtype=torch.uint8)
#
#         if self.transforms is not None:
#             img = self.transforms(img)
#
#         return img, target
#
#     def __ellipse_mask(self, obj_mask: np.ndarray, points: tuple):
#         mask = np.zeros_like(obj_mask)
#         points = [int(point) for point in points]
#         print(points)
#         mask = cv2.ellipse(mask, (points[:2], points[2:], 0), (255, 255, 255), -1)
#         return mask
#
#     def __rect_mask(self, obj_mask: np.ndarray, points: tuple):
#         pts = [[points[0], points[1]],
#                [points[0] + points[2], points[1]],
#                [points[0] + points[2], points[1] + points[3]],
#                [points[0], points[1]] + points[3]]
#         return cv2.fillPoly(obj_mask, pts, (255, 255, 255))
#
#     def __len__(self):
#         return len(self.__keys)
