import os
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import jittor as jt
from jittor.dataset import Dataset
import random

def convert_coco_poly_mask(segmentations, h, w):
    masks = []
    for seg in segmentations:
        if isinstance(seg, list):
            rles = maskUtils.frPyObjects(seg, h, w)
            if isinstance(rles, list):
                rle = maskUtils.merge(rles)
            else:
                rle = rles
            m = maskUtils.decode(rle)
        elif isinstance(seg, dict):
            m = maskUtils.decode(seg)
        else:
            try:
                rles = maskUtils.frPyObjects(seg, h, w)
                m = maskUtils.decode(rles)
            except Exception:
                m = np.zeros((h, w), dtype=np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        masks.append(m.astype(np.uint8))
    if masks:
        return np.stack(masks, axis=0)  # (N, h, w)
    else:
        return np.zeros((0, h, w), dtype=np.uint8)

def coco_remove_images_without_annotations(coco, ids):
    valid_ids = []
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue
        anns = coco.loadAnns(ann_ids)
        has_valid = False
        for obj in anns:
            if obj.get("iscrowd", 0) == 0 and obj.get("area", 0) > 0:
                has_valid = True
                break
        if has_valid:
            valid_ids.append(img_id)
    return valid_ids

class COCODataset(Dataset):

    def __init__(self, img_dir, ann_file,is_train, transforms=None, 
                 remove_images_without_annotations=False, 
                 label_offset=1, 
                 sample_fraction=0.1,  # 采样比例
                 min_objects=2):       
        super().__init__()
        assert label_offset in (0,1)
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.label_offset = label_offset
        self.min_objects = min_objects  
        self.sample_fraction = sample_fraction  
        self.is_train=is_train
        
        if remove_images_without_annotations:
            self.ids = coco_remove_images_without_annotations(self.coco, self.ids)
        
        
        self.filter_images_by_object_count()
        
        
        self.subsample_dataset()

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        self.cat2label = {c['id']: idx + label_offset for idx, c in enumerate(cats)}
        self.label2cat = {idx + label_offset: c['id'] for idx, c in enumerate(cats)}
        self.num_categories = len(cats)
        print(f"最终数据集大小: {len(self.ids)} 张图像")
        
        

    def filter_images_by_object_count(self):
        """过滤掉目标数不足的图像"""
        filtered_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 统计有效目标数
            valid_objects = 0
            for obj in anns:
                if obj.get("iscrowd", 0) == 0 and obj.get("area", 0) > 0:
                    valid_objects += 1
            
            # 保留有足够目标的图像
            if valid_objects >= self.min_objects:
                filtered_ids.append(img_id)
        
        self.ids = filtered_ids
        

    def subsample_dataset(self):
        """随机采样指定比例的数据"""
        if self.sample_fraction < 1.0:
            # 固定随机种子以确保可重复性
            random.seed(42)
            num_samples = max(1, int(len(self.ids) * self.sample_fraction))
            self.ids = random.sample(self.ids, num_samples)
            

    def __len__(self):
        return len(self.ids)

    def parse_targets(self, img_id, anns, w, h):
        # 仅保留非crowd且面积>0的目标
        anno = [obj for obj in anns if obj.get('iscrowd', 0) == 0 and obj.get('area', 0) > 0]

        boxes = [obj["bbox"] for obj in anno]  # [x, y, w, h]
        if boxes:
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]  # 转换为 x1,y1,x2,y2
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # 映射类别ID到连续标签
        classes = []
        for obj in anno:
            cid = obj["category_id"]
            if cid in self.cat2label:
                classes.append(self.cat2label[cid])
            else:
                # 回退：映射到第一个类别
                classes.append(self.label_offset)  
        if classes:
            classes = np.array(classes, dtype=np.int64)
        else:
            classes = np.zeros((0,), dtype=np.int64)

        area = np.array([obj["area"] for obj in anno], dtype=np.float32) if anno else np.zeros((0,), dtype=np.float32)
        iscrowd = np.array([obj.get("iscrowd", 0) for obj in anno], dtype=np.int64) if anno else np.zeros((0,), dtype=np.int64)

        segmentations = [obj.get("segmentation", None) for obj in anno]
        if len(segmentations) > 0:
            masks = convert_coco_poly_mask(segmentations, h, w)  # uint8 (N,h,w)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)

        # 过滤无效框
        if boxes.shape[0] > 0:
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not np.all(keep):
                boxes = boxes[keep]
                classes = classes[keep]
                masks = masks[keep]
                area = area[keep]
                iscrowd = iscrowd[keep]

        target = {
            "boxes": boxes,               # (N,4) float32
            "labels": classes,            # (N,) int64 (连续标签)
            "masks": masks,               # (N,H,W) uint8
            "image_id": np.array([img_id], dtype=np.int64),
            "area": area,                 # (N,) float32
            "iscrowd": iscrowd,           # (N,) int64
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        return target

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.img_dir, path)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target = self.parse_targets(img_id, anns, w, h)

        if self.transforms is not None:
            image, target = self.transforms(image, target, self.is_train)

        # 转换目标字段为jt.array
        boxes = jt.array(target["boxes"].astype(np.float32)) if target["boxes"].size else jt.zeros((0,4), dtype=jt.float32)
        labels = jt.array(target["labels"].astype(np.int64)) if target["labels"].size else jt.zeros((0,), dtype=jt.int64)
        masks = jt.array(target["masks"].astype(np.float32)) if target["masks"].size else jt.zeros((0, h, w), dtype=jt.float32)
        area = jt.array(target["area"].astype(np.float32)) if target["area"].size else jt.zeros((0,), dtype=jt.float32)
        iscrowd = jt.array(target["iscrowd"].astype(np.int64)) if target["iscrowd"].size else jt.zeros((0,), dtype=jt.int64)
        image_id = jt.array(target["image_id"].astype(np.int64))
        orig_size = jt.array(target["orig_size"].astype(np.int64))

        target_jt = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": orig_size,
        }

        return image, target_jt