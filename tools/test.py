#!/usr/bin/env python3
"""
test.py - 对没有 annotation 的测试集做推理并可视化

用法示例：
python test.py --path /path/to/images_dir --model outputs/model_11.pkl --out test_results --num 10

说明：
- --path: 图片目录或单张图片路径（若为目录，会从中随机抽取 --num 张）
- --model: 模型文件路径（支持 jt.save 的 state_dict 或包含 "model" 字段的 checkpoint）
- 结果保存在 --out 目录，每张图片生成两个文件：<stem>_det.png（bbox）和 <stem>_seg.png（mask overlay）
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jittor as jt

from models import MaskRCNN
from models.backbone import resnet50_fpn_backbone

# -------------------- 辅助函数 --------------------
def list_images_in_dir(dir_path: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    p = Path(dir_path)
    return [str(x) for x in p.iterdir() if x.suffix.lower() in exts]

def pick_images(path: str, num: int, seed: int = 42) -> List[str]:
    p = Path(path)
    if p.is_dir():
        imgs = list_images_in_dir(path)
        random.seed(seed)
        return random.sample(imgs, min(num, len(imgs)))
    elif p.is_file():
        return [str(p)]
    else:
        raise ValueError(f"Path not found: {path}")

def load_state_to_model(model, model_path: str):
    ck = jt.load(model_path)
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
    else:
        state = ck
    try:
        model.load_parameters(state)
    except Exception:
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"无法将权重加载到模型：{e}")

def pil_to_jt_tensor(img: Image.Image) -> jt.Var:
    arr = np.array(img, dtype=np.float32) / 255.0  # H,W,C
    arr = arr.transpose(2, 0, 1).astype(np.float32)  # C,H,W
    return jt.array(arr)

def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, jt.Var):
        try:
            return x.numpy()
        except Exception:
            try:
                return np.array(x.data.tolist())
            except Exception:
                return np.array(x)
    try:
        return np.array(x)
    except Exception:
        return x

# ------------ visualization helpers --------------
def draw_boxes_on_pil(img: Image.Image, boxes: np.ndarray, labels: List[str], scores: np.ndarray,
                      color=(255, 0, 0), thickness=2, font=None) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    N = 0 if boxes is None else boxes.shape[0]
    H_img, W_img = out.size[1], out.size[0]  
    for i in range(N):
        # 安全取值与检查
        try:
            vals = boxes[i].tolist()
            if len(vals) < 4:
                continue
            x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        except Exception:

            continue

        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # 忽略极小的 box（宽/高 < 1 像素）
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue


        x1c = max(0, min(W_img, x1))
        x2c = max(0, min(W_img, x2))
        y1c = max(0, min(H_img, y1))
        y2c = max(0, min(H_img, y2))
        if x2c <= x1c or y2c <= y1c:
            continue

        x1i, y1i, x2i, y2i = int(round(x1c)), int(round(y1c)), int(round(x2c)), int(round(y2c))

        # 画框与文本
        draw.rectangle([x1i, y1i, x2i, y2i], outline=color, width=thickness)
        label = labels[i] if i < len(labels) else ""
        score = None
        try:
            if scores is not None:
                score = float(scores[i])
        except Exception:
            score = None
        text = f"{label} {score:.2f}" if score is not None else f"{label}"

        # 计算文字背景大小并绘制
        if font:
            ts = font.getsize(text)
        else:
            ts = (len(text) * 6, 10)
        tx2 = x1i + 3 + ts[0] + 2
        ty2 = y1i + 3 + ts[1] + 2
        draw.rectangle([x1i + 3, y1i + 3, tx2, ty2], fill=color)
        draw.text((x1i + 5, y1i + 4), text, fill=(255, 255, 255), font=font)

    return out


def overlay_masks_on_pil(img: Image.Image, masks: np.ndarray, labels: List[str], scores: np.ndarray,
                         alpha: float = 0.45, thickness: int = 2, font=None) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    H, W = base.size[1], base.size[0]
    N = 0 if masks is None else masks.shape[0]
    rng = np.random.RandomState(12345)
    colors = [tuple(map(int, rng.randint(0, 255, size=3).tolist())) for _ in range(max(1, N))]
    for i in range(N):
        m = masks[i]
        # ensure binary 0/1
        if m.dtype != np.uint8 and m.dtype != np.bool_:
            m = (m >= 0.5).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            continue
        mask_pil = Image.fromarray((m * 255).astype(np.uint8))
        color = colors[i]
        overlay.paste(Image.new("RGBA", base.size, color + (int(255 * alpha),)), (0, 0), mask_pil)
        ys, xs = np.where(m)
        y1, y2 = int(np.min(ys)), int(np.max(ys))
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        draw_overlay.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        label = labels[i] if i < len(labels) else ""
        score = scores[i] if scores is not None else None
        text = f"{label} {score:.2f}" if score is not None else f"{label}"
        if font:
            ts = font.getsize(text)
        else:
            ts = (len(text) * 6, 10)
        draw_overlay.rectangle([x1 + 3, y1 + 3, x1 + 6 + ts[0], y1 + 6 + ts[1]], fill=color)
        draw_overlay.text((x1 + 5, y1 + 4), text, fill=(255, 255, 255), font=font)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

# ------------- Resize/Preprocess (与训练一致) ---------------
from PIL import ImageOps

class TestResize:
    """
    Resize image to meet min_size/max_size constraints, keep aspect ratio,
    return resized PIL image and scale (sx, sy) and new size (new_w, new_h)
    """
    def __init__(self, min_size=800, max_size=1333, interpolation=Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, pil_img: Image.Image):
        orig_w, orig_h = pil_img.size
        short_side = min(orig_h, orig_w)
        long_side = max(orig_h, orig_w)
        scale = float(self.min_size) / float(short_side)
        if int(round(long_side * scale)) > self.max_size:
            scale = float(self.max_size) / float(long_side)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        if new_h == orig_h and new_w == orig_w:
            return pil_img, (1.0, 1.0), (new_w, new_h)
        img_resized = pil_img.resize((new_w, new_h), resample=self.interpolation)
        return img_resized, (new_w / orig_w, new_h / orig_h), (new_w, new_h)

# -------------------- 推理主流程 --------------------
@jt.no_grad()
def run_test(image_paths: List[str],
             model_path: str,
             out_dir: str,
             num_classes: int = 80,
             min_size: int = 800,
             max_size: int = 1333,
             score_thresh: float = 0.05):
    os.makedirs(out_dir, exist_ok=True)
    import jittor as jt
    print("jt.has_cuda:", jt.has_cuda)
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print("jt.flags.use_cuda:", jt.flags.use_cuda)
    # 创建模型（与训练时保持一致）
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth")
    model = MaskRCNN(backbone, num_classes=num_classes+1)
    model.eval()

    print("Loading model weights from", model_path)
    load_state_to_model(model, model_path)
    print("Model loaded.")

    resize_fn = TestResize(min_size=min_size, max_size=max_size)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for p in image_paths:
        print("Processing", p)
        orig = Image.open(p).convert("RGB")
        orig_w, orig_h = orig.size

        img_resized, (sx, sy), (new_w, new_h) = resize_fn(orig)

        # preprocessing
        img_jt = pil_to_jt_tensor(img_resized)  # C,H,W
        mean = jt.array(np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1))
        std = jt.array(np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1))
        img_jt = (img_jt - mean) / std

        images_input = img_jt.unsqueeze(0)
 
        jt.gc()
        jt.clean()
            
        outputs = model(images_input, ori_size=[(orig_h, orig_w)])

        if isinstance(outputs, dict):
            out0 = outputs
        else:
            try:
                out0 = outputs[0]
            except Exception:
                out0 = outputs

        boxes = to_numpy(out0.get("boxes", None))
        labels = to_numpy(out0.get("labels", None))
        scores = to_numpy(out0.get("scores", None))
        masks = to_numpy(out0.get("masks", None))

        # if no detections
        if boxes is None or boxes.size == 0:
            print(f"No detections for {p}")
            det_img = orig.copy()
            seg_img = orig.copy()
            det_img.save(os.path.join(out_dir, Path(p).stem + "_det.png"))
            seg_img.save(os.path.join(out_dir, Path(p).stem + "_seg.png"))
            continue

        
        boxes = boxes.reshape(-1, 4)


        '''max_x2 = boxes[:,2].max()
        max_y2 = boxes[:,3].max()'''
        '''if max_x2 <= new_w + 1e-3 and max_y2 <= new_h + 1e-3:
            # scale to original
            scale_x = orig_w / float(new_w)
            scale_y = orig_h / float(new_h)
            boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
            boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y'''

        # filter by score
        if scores is not None:
            keep = np.where(scores >= score_thresh)[0]
        else:
            keep = np.arange(boxes.shape[0])
        if keep.size == 0:
            print(f"No detections above score {score_thresh} for {p}")
            det_img = orig.copy()
            seg_img = orig.copy()
            det_img.save(os.path.join(out_dir, Path(p).stem + "_det.png"))
            seg_img.save(os.path.join(out_dir, Path(p).stem + "_seg.png"))
            continue

        boxes = boxes[keep]
        labels = labels[keep] if labels is not None else np.zeros((len(keep),), dtype=np.int32)
        scores = scores[keep] if scores is not None else np.zeros((len(keep),), dtype=np.float32)
        if masks is not None:
            # masks could be (N,1,H,W) or (N,H,W)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:,0,...]
            if masks.ndim == 4 and masks.shape[1] in (3,):  # improbable, average
                masks = masks.mean(axis=1)
            # If mask size equals resized image, resize to original
            if masks.ndim == 3:
                mh, mw = masks.shape[1], masks.shape[2]
                if (mh, mw) != (orig_h, orig_w):
                    resized_masks = []
                    for m in masks:
                        pil_m = Image.fromarray((m * 255).astype(np.uint8)) if m.dtype != np.uint8 else Image.fromarray(m)
                        pil_m = pil_m.resize((orig_w, orig_h), resample=Image.NEAREST)
                        arr = np.array(pil_m).astype(np.uint8)
                        resized_masks.append((arr >= 128).astype(np.uint8))
                    masks = np.stack(resized_masks, axis=0)
            else:
                masks = None

        # prepare label text (simple id -> string). If you want names, pass a mapping.
        labels_text = [str(int(x)) for x in labels]

        # visualization
        try:
            det_viz = draw_boxes_on_pil(orig.copy(), boxes, labels_text, scores, color=(255,0,0), thickness=2, font=ImageFont.load_default())
        except Exception:
            det_viz = draw_boxes_on_pil(orig.copy(), boxes, labels_text, scores, color=(255,0,0), thickness=2, font=None)
        det_name = os.path.join(out_dir, Path(p).stem + "_det.png")
        det_viz.save(det_name)

        if masks is not None and masks.size > 0:
            try:
                seg_viz = overlay_masks_on_pil(orig.copy(), masks, labels_text, scores, alpha=0.45, thickness=2, font=ImageFont.load_default())
            except Exception:
                seg_viz = overlay_masks_on_pil(orig.copy(), masks, labels_text, scores, alpha=0.45, thickness=2, font=None)
        else:
            seg_viz = det_viz

        seg_name = os.path.join(out_dir, Path(p).stem + "_seg.png")
        seg_viz.save(seg_name)
        print(f"Saved: {det_name}, {seg_name}")

# ----------------------- CLI ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="图片目录或单张图片路径")
    p.add_argument("--model", required=True, help="模型文件路径")
    p.add_argument("--out", default="test_results", help="输出目录")
    p.add_argument("--num", type=int, default=10, help="随机选择图片数量（若 path 是单张图会忽略）")
    p.add_argument("--seed", type=int, default=2025, help="随机种子")
    p.add_argument("--min-size", type=int, default=800, help="resize min_size")
    p.add_argument("--max-size", type=int, default=1333, help="resize max_size")
    p.add_argument("--score-thresh", type=float, default=0.3, help="检测分数阈值")
    p.add_argument("--num-classes", type=int, default=80, help="训练时的类别数")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_list = pick_images(args.path, args.num, seed=args.seed)
    print("Picked images:", img_list)
    run_test(img_list, args.model, args.out, num_classes=args.num_classes,
             min_size=args.min_size, max_size=args.max_size, score_thresh=args.score_thresh)
