import os
import time
import math
import sys
import copy
import json
import yaml
import datetime
import numpy as np
import jittor as jt
from jittor import optim

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

# ----------------- helpers -----------------
def _to_numpy(x):
    """Robust conversion from jt.Var / list / numpy to numpy.ndarray."""
    if isinstance(x, jt.Var):
        try:
            return x.numpy()
        except Exception:
            try:
                return np.array(x.data.tolist())
            except Exception:
                return np.array(x.tolist())
    try:
        return np.array(x)
    except Exception:
        return np.array(x, dtype=object)


# ----------------- training utilities -----------------
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """学习率预热调度器（返回一个 jt.optim.LambdaLR）"""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / float(warmup_iters)
        return warmup_factor * (1 - alpha) + alpha
    return jt.optim.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch(model, optimizer, data_loader, epoch, log_file=None,
                    print_freq=50, warmup=False, global_scheduler=None, start_global_step=0):
    """
    训练一个 epoch：支持传入 global_scheduler（per-iteration），
    如果提供则每次 iter 后会调用 global_scheduler.step() 并更新 global_step。
    返回 (running_loss, now_lr, updated_global_step)
    """
    model.train()

    running_loss = 0.0
    num_iter = 0
    start_time = time.time()

    # 局部 warmup（仅当没有外部 global_scheduler 时作为后备）
    lr_scheduler = None
    if warmup and global_scheduler is None:
        warmup_factor = 1.0 / 250
        warmup_iters = min(250, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 日志表头控制（若文件不存在或空则写表头）
    need_header = False
    if log_file:
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            need_header = True

    global_step = int(start_global_step)

    now_lr = None

    for i, (images, targets) in enumerate(data_loader):
        # 显存回收尝试（若可用）
        try:
            jt.gc()
            jt.clean()
        except Exception:
            pass

        original_image_sizes = []
        for t in targets["orig_size"]:
            original_image_sizes.append((t[0], t[1]))

        loss_dict = model(images, targets=targets, ori_size=original_image_sizes)

        # losses 用于反向 (jt.Var)
        losses = sum(loss_dict.values())

        # 把每个 loss 转成 float，用于记录与平均
        loss_scalars = {}
        for k, v in loss_dict.items():
            try:
                val = float(v.item()) if hasattr(v, 'item') else float(v)
            except Exception:
                try:
                    val = float(_to_numpy(v))
                except Exception:
                    val = float('nan')
            loss_scalars[k] = val

        loss_value = sum(loss_scalars.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print("loss_dict:", loss_dict)
            sys.exit(1)

        # 反向 + 优化（Jittor: optimizer.step(losses)）
        optimizer.step(losses)

        # 如果提供 global_scheduler（推荐），每个迭代都 step 它
        if global_scheduler is not None:
            try:
                global_scheduler.step()
            except Exception:
                pass
            global_step += 1
        else:
            # fallback: 使用函数内部创建的 lr_scheduler（用于兼容旧逻辑）
            if lr_scheduler is not None:
                try:
                    lr_scheduler.step()
                except Exception:
                    pass

        num_iter += 1
        running_loss = (running_loss * (num_iter - 1) + loss_value) / num_iter

        # 获取当前学习率
        try:
            now_lr = optimizer.param_groups[0]["lr"]
        except Exception:
            now_lr = getattr(optimizer, "lr", None)

        # 日志写入（每100 iter 或最后一 iter）
        if log_file and (i % 100 == 0 or i == len(data_loader) - 1):
            if need_header:
                header_fields = ["epoch", "iter", "running_loss", "lr"] + list(loss_scalars.keys())
                with open(log_file, "a") as f:
                    f.write("\t".join(header_fields) + "\n")
                need_header = False

            loss_values_str = [f"{loss_scalars[k]:.6f}" for k in loss_scalars.keys()]
            line_fields = [str(epoch), str(i), f"{running_loss:.6f}", f"{now_lr:.6f}" if now_lr is not None else "None"]
            line_fields += loss_values_str
            with open(log_file, "a") as f:
                f.write("\t".join(line_fields) + "\n")

        # 控制台输出
        if i % print_freq == 0 or i == len(data_loader) - 1:
            if now_lr is not None:
                print(f"Epoch: [{epoch}] Iter: [{i}/{len(data_loader)}] Loss: {running_loss:.6f} LR: {now_lr:.6f}")
            else:
                print(f"Epoch: [{epoch}] Iter: [{i}/{len(data_loader)}] Loss: {running_loss:.6f} LR: None")

        # 释放引用帮助回收
        images = None
        targets = None
        losses = None
        loss_dict = None
        loss_scalars = None

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

    return running_loss, now_lr, global_step


# ----------------- evaluation & metric -----------------
@jt.no_grad()
def evaluate(model, data_loader):
    """
    评估函数：model 在 eval 模式下 inference，使用 data_loader.dataset.coco 做映射。
    返回：det_info, seg_info（COCOeval.stats 列表或 None）
    """
    model.eval()

    
    classes_mapping = getattr(data_loader.dataset, "label2cat", None)

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox",
                                results_file_name="det_results.json", classes_mapping=classes_mapping)
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm",
                                results_file_name="seg_results.json", classes_mapping=classes_mapping)

    for i, (images, targets) in enumerate(data_loader):

        try:
            jt.gc()
            jt.clean()
        except Exception:
            pass
        original_image_sizes = []
        for t in targets["orig_size"]:
            original_image_sizes.append((t[0], t[1]))

        outputs = model(images, ori_size=original_image_sizes)

        # ensure outputs is list-like per-image
        if isinstance(outputs, dict):
            outputs_list = [outputs]
        else:
            try:
                outputs_list = list(outputs)
            except Exception:
                outputs_list = [outputs]

        if i % 20 == 0:
            print(f"Eval Iter [{i}/{len(data_loader)}]")

     
        targets_list = [targets]
        
        
        # 更新 evaluator（targets_list 与 outputs_list 应该一一对应）
        det_metric.update(targets_list, outputs_list)
        seg_metric.update(targets_list, outputs_list)

        images = None
        targets = None
        outputs = None

    det_info = det_metric.evaluate()
    seg_info = seg_metric.evaluate()

    if det_info:
        print(f"Detection mAP: {det_info[0]:.4f}")
    if seg_info:
        print(f"Segmentation mAP: {seg_info[0]:.4f}")

    return det_info, seg_info


class EvalCOCOMetric:
    """
    A robust COCO evaluator that accepts:
      - targets: list of per-image dicts (each has 'image_id', etc.)
      - outputs: list of per-image dicts (each has 'boxes','scores','labels','masks' optional)
    It will map dataset-internal labels back to COCO category ids when classes_mapping is provided.
    """

    def __init__(self, coco: COCO, iou_type: str = "bbox", results_file_name: str = "predict_results.json", classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.iou_type = iou_type
        assert iou_type in ["bbox", "segm", "keypoints"]
        self.results = []
        self.results_file_name = results_file_name
        # classes_mapping: dataset.label2cat mapping (label -> original_coco_cat_id)
        self.classes_mapping = classes_mapping

    def _to_numpy(self, x):
        return _to_numpy(x)

    def _resolve_category_id(self, label):
        try:
            lab = int(label)
        except Exception:
            try:
                lab = int(np.array(label).tolist())
            except Exception:
                lab = label

        if self.classes_mapping is None:
            return lab

        if lab in self.classes_mapping:
            return int(self.classes_mapping[lab])
        if str(lab) in self.classes_mapping:
            return int(self.classes_mapping[str(lab)])
        return lab

    def _get_image_id(self, target):
        if isinstance(target, dict) and "image_id" in target:
            v = target["image_id"]
        else:
            v = target
        if isinstance(v, jt.Var):
            try:
                return int(v.item())
            except Exception:
                try:
                    return int(np.array(v.data.tolist()).item())
                except Exception:
                    return int(np.array(v).item())
        try:
            return int(np.array(v).tolist())
        except Exception:
            return int(v)

    def prepare_for_coco_detection(self, targets, outputs):
        for target, output in zip(targets, outputs):
            if output is None:
                continue
            boxes = output.get("boxes", None) if isinstance(output, dict) else None
            labels = output.get("labels", None) if isinstance(output, dict) else None
            scores = output.get("scores", None) if isinstance(output, dict) else None

            if boxes is None or labels is None or scores is None:
                continue

            try:
                img_id = self._get_image_id(target)
            except Exception:
                continue

            per_image_boxes = self._to_numpy(boxes).astype(np.float32)
            if per_image_boxes.size == 0:
                continue
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

            per_image_labels = self._to_numpy(labels).astype(np.int64)
            per_image_scores = self._to_numpy(scores).astype(np.float32)

            for box, cls, score in zip(per_image_boxes, per_image_labels, per_image_scores):
                coco_cat = self._resolve_category_id(int(cls))
                box_list = [round(float(b), 2) for b in box.tolist()]
                res = {
                    "image_id": int(img_id),
                    "category_id": int(coco_cat),
                    "bbox": box_list,
                    "score": round(float(score), 3)
                }
                self.results.append(res)

    def prepare_for_coco_segmentation(self, targets, outputs):
        for target, output in zip(targets, outputs):
            if output is None:
                continue

            masks = output.get("masks", None) if isinstance(output, dict) else None
            labels = output.get("labels", None) if isinstance(output, dict) else None
            scores = output.get("scores", None) if isinstance(output, dict) else None

            if masks is None or labels is None or scores is None:
                continue

            try:
                img_id = self._get_image_id(target)
            except Exception:
                continue

            masks_np = self._to_numpy(masks)
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0, :, :]

            if masks_np.ndim != 3:
                continue

            per_image_labels = self._to_numpy(labels).astype(np.int64)
            per_image_scores = self._to_numpy(scores).astype(np.float32)

            masks_bin = (masks_np > 0.5).astype(np.uint8)
            for m, cls, score in zip(masks_bin, per_image_labels, per_image_scores):
                rle = mask_util.encode(np.asfortranarray(m[:, :, np.newaxis]))[0]
                if isinstance(rle.get("counts", None), bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                coco_cat = self._resolve_category_id(int(cls))
                res = {
                    "image_id": int(img_id),
                    "category_id": int(coco_cat),
                    "segmentation": rle,
                    "score": round(float(score), 3)
                }
                self.results.append(res)

    def update(self, targets, outputs):
        if self.iou_type == "bbox":
            self.prepare_for_coco_detection(targets, outputs)
        elif self.iou_type == "segm":
            self.prepare_for_coco_segmentation(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def evaluate(self):
        if len(self.results) == 0:
            print("No predictions to evaluate.")
            return None

        with open(self.results_file_name, "w") as f:
            json.dump(self.results, f)

        coco_true = self.coco
        coco_pre = coco_true.loadRes(self.results_file_name)
        coco_eval = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist() if hasattr(coco_eval, 'stats') else list(coco_eval.stats)
        return stats
