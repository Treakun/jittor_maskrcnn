import jittor as jt
from typing import Tuple

import jittor as jt
import numpy as np
import math
from typing import List
import os
import jittor as jt
import numpy as np
import os
import jittor as jt
import numpy as np

def _numpy_nms(boxes_np: np.ndarray, scores_np: np.ndarray, iou_thresh: float):
    
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores_np.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def _check_and_fix_boxes(boxes: jt.Var):
    # boxes: [N,4]
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.float().reshape(-1,4)
    # 防止极端值：clamp 到合理范围（可调整）
    boxes = boxes.clamp(-1e6, 1e6)
    # 保证 x2>=x1, y2>=y1
    x1 = jt.minimum(boxes[:,0], boxes[:,2])
    y1 = jt.minimum(boxes[:,1], boxes[:,3])
    x2 = jt.maximum(boxes[:,0], boxes[:,2])
    y2 = jt.maximum(boxes[:,1], boxes[:,3])
    boxes = jt.stack([x1, y1, x2, y2], dim=1)
    return boxes

def nms_jt(boxes: jt.Var, scores: jt.Var, iou_threshold: float):
    """
    Pure Jittor implementation of NMS (per-class usage expected).
    Returns jt.Var of kept indices (int32) relative to input boxes order.
    """
    boxes = boxes.float().reshape(-1,4)
    scores = scores.float().reshape(-1)
    N = int(boxes.shape[0])
    if N == 0:
        return jt.array([], dtype=jt.int32)

    # safety fix
    boxes = _check_and_fix_boxes(boxes)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

    # order: indices sorted by score desc
    order = jt.argsort(scores, descending=True)
    keep = []

   
    while order.numel() > 0:
        i = int(order[0].item())   # single index (force to python int)
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
       
        xx1 = jt.maximum(x1[i], x1[rest])
        yy1 = jt.maximum(y1[i], y1[rest])
        xx2 = jt.minimum(x2[i], x2[rest])
        yy2 = jt.minimum(y2[i], y2[rest])
        w = jt.maximum(0.0, xx2 - xx1 + 1.0)
        h = jt.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        # guard union>0
        iou = inter / union
        inds = jt.where(iou <= iou_threshold)[0]
        order = rest[inds]

    return jt.array(keep, dtype=jt.int32)

def batched_nms(boxes: jt.Var, scores: jt.Var, idxs: jt.Var, iou_threshold: float):
    """
    Stable batched NMS: per-class processing in pure Jittor,
    with a small-scale numpy fallback per-class if jt.nms fails.
    """
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)

    boxes = boxes.float().reshape(-1,4)
    scores = scores.float().reshape(-1)
    idxs = idxs.reshape(-1).int()

    # sanitize boxes globally (cheap)
    boxes = _check_and_fix_boxes(boxes)

    # unique classes (jt op)
    try:
        unique_classes = jt.unique(idxs)
    except Exception:
        # 如果 jt.unique 异常，构造范围
        try:
            unique_classes = jt.arange(0, int(idxs.max().item())+1, dtype=jt.int32)
        except Exception:
            unique_classes = jt.array([], dtype=jt.int32)

    keep_indices = []

    # iterate per class (类别通常 << N)
    for c in unique_classes:
        c_int = int(c.item()) if isinstance(c, jt.Var) else int(c)
        mask = (idxs == c_int)
        inds = jt.where(mask)[0]   
        if inds.numel() == 0:
            continue

        cls_boxes = boxes[inds]
        cls_scores = scores[inds]

        # 
        if os.environ.get("USE_CPU_NMS", "0") == "1":
            try:
                b_np = cls_boxes.numpy()
                s_np = cls_scores.numpy()
                local_keep = _numpy_nms(b_np, s_np, iou_threshold)
                keep_idx = [int(inds[int(k)]) for k in local_keep]
            except Exception:
                # 回退：按 score topk 保留候选，避免崩溃
                topk = jt.argsort(cls_scores, descending=True)[:min(100, int(inds.numel()))]
                try:
                    topk_np = topk.numpy()
                    keep_idx = [int(inds[int(k)]) for k in topk_np]
                except Exception:
                    keep_idx = [int(inds[int(k)].item()) for k in range(int(topk.numel()))]
        else:
           
            try:
                local_keep = nms_jt(cls_boxes, cls_scores, iou_threshold)
                # local_keep 相对于 cls_boxes，需要映射回原始索引
                try:
                    local_keep_np = local_keep.numpy()   # per-class 通常很小，可安全同步
                    keep_idx = [int(inds[int(k)]) for k in local_keep_np]
                except Exception:
                    keep_idx = [int(inds[int(i)].item()) for i in range(int(local_keep.numel()))]
            except Exception:
                # 如果 jt 路径异常，回退到 per-class numpy nms
                try:
                    b_np = cls_boxes.numpy()
                    s_np = cls_scores.numpy()
                    local_keep = _numpy_nms(b_np, s_np, iou_threshold)
                    keep_idx = [int(inds[int(k)]) for k in local_keep]
                except Exception:
                    # 最后退化：按 score topk
                    topk = jt.argsort(cls_scores, descending=True)[:min(100, int(inds.numel()))]
                    try:
                        topk_np = topk.numpy()
                        keep_idx = [int(inds[int(k)]) for k in topk_np]
                    except Exception:
                        keep_idx = [int(inds[int(k)].item()) for k in range(int(topk.numel()))]

        keep_indices.extend(keep_idx)

    if len(keep_indices) == 0:
        return jt.empty((0,), dtype=jt.int64)

    # 返回 Jittor 数组（注意类型）
    return jt.array(np.array(keep_indices, dtype=np.int64))

def remove_small_boxes(boxes, min_size):
    """
    移除小于指定尺寸的边界框
    输入:
        boxes: [N,4] (x1,y1,x2,y2)
        min_size: 最小尺寸阈值
    返回:
        keep: 保留框的索引
    """
    ws = boxes[:, 2] - boxes[:, 0]  # 宽度
    hs = boxes[:, 3] - boxes[:, 1]  # 高度
    keep = (ws >= min_size) & (hs >= min_size)  # 同时满足宽高阈值
    return jt.where(keep)[0]  # 返回满足条件的索引
def clip_boxes_to_image(boxes, size):
    """
    将边界框裁剪到图像范围内
    输入:
        boxes: [N,4] (x1,y1,x2,y2)
        size: (height, width) 图像尺寸
    返回:
        clipped_boxes: 裁剪后的边界框
    """
    height, width = size
    # 分离坐标
    boxes_x = boxes[:, 0::2]  # x1,x2
    boxes_y = boxes[:, 1::2]  # y1,y2
    
    # 裁剪到[0, width]和[0, height]
    boxes_x = boxes_x.clamp(0, width)
    boxes_y = boxes_y.clamp(0, height)
    
    # 重新组合坐标
    clipped_boxes = jt.stack([boxes_x[:, 0], boxes_y[:, 0], 
                              boxes_x[:, 1], boxes_y[:, 1]], dim=1)
    return clipped_boxes
def box_area(boxes):
    """
    计算边界框面积，支持多种维度输入
    """
    if boxes.ndim == 1:
        # 单个边界框 [x1, y1, x2, y2]
        return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    elif boxes.ndim == 2:
        # 多个边界框 [N, 4]
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    else:
        raise ValueError(f"Unsupported boxes dimension: {boxes.ndim}")


def box_iou(boxes1, boxes2):
    """
    计算两组边界框之间的IoU
    """
    # 确保输入是二维的 [N, 4]
    if boxes1.ndim == 1:
        boxes1 = boxes1.unsqueeze(0)  # 变为 [1, 4]
    if boxes2.ndim == 1:
        boxes2 = boxes2.unsqueeze(0)  # 变为 [1, 4]
    
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]
    
    # 计算交集的左上角和右下角坐标
    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    # 计算交集区域的宽高
    wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # 计算并集
    union = area1[:, None] + area2 - inter
    
    # 计算IoU
    iou = inter / union
    return iou