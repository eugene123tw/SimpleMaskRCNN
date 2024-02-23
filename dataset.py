"""Base class for OTXDataset."""
from datumaro import Image, Polygon

import numpy as np
from torch.utils.data import Dataset
import torch
import pycocotools.mask as mask_utils
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def polygon_to_bitmap(
    polygons: list[Polygon],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert a list of polygons to a bitmap mask.

    Args:
        polygons (list[Polygon]): List of Datumaro Polygon objects.
        height (int): bitmap height
        width (int): bitmap width

    Returns:
        np.ndarray: bitmap masks
    """
    polygons = [polygon.points for polygon in polygons]
    rles = mask_utils.frPyObjects(polygons, height, width)
    return mask_utils.decode(rles).astype(bool).transpose((2, 0, 1))


class DatumaroDataset(Dataset):
    """Base DatumaroDataset.

    Args:
        dm_subset: Datumaro subset of a dataset
        transforms: Transforms to apply on images
    """

    def __init__(
        self,
        dm_dataset,
        transforms,
    ) -> None:
        self.dm_subset = dm_dataset
        self.ids = [item.id for item in dm_dataset]
        self.num_classes = len([i for i in self.dm_subset.categories().values()][-1])
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def _iterable_transforms(self, item):
        if not isinstance(self.transforms, list):
            raise TypeError(item)

        results = item
        for transform in self.transforms:
            results = transform(results)

        return results

    def __getitem__(self, index: int):
        results = self._get_item_impl(index)

        if results is not None:
            return results

    def _get_img_data_and_shape(self, img: Image):
        img_data = img.data

        if img_data is None:
            msg = "Cannot get image data"
            raise RuntimeError(msg)
        img_data = np.transpose(img_data, (2, 0, 1))
        return img_data, img_data.shape[:2]

    def _get_item_impl(self, index: int):
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_labels, gt_masks = [], [], []

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                bbox = np.array(annotation.get_bbox(), dtype=np.float32)
                gt_bboxes.append(bbox)
                gt_labels.append(annotation.label)
                gt_masks.append(polygon_to_bitmap([annotation], *img_shape)[0])

        # convert xywh to xyxy format
        bboxes = np.array(gt_bboxes, dtype=np.float32)
        bboxes[:, 2:] += bboxes[:, :2]

        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        masks = torch.tensor(masks, dtype=torch.uint8)
        labels = torch.tensor(gt_labels, dtype=torch.long)
        img_data = torch.tensor(img_data, dtype=torch.float32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        img_data = tv_tensors.Image(img_data)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=F.get_size(img_data))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels

        if self.transforms is not None:
            img_data, target = self.transforms(img_data, target)

        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = index

        return img_data, target
