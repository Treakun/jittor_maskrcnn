import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import jittor as jt
from jittor import nn
from .roi_align import MultiScaleRoIAlign
from .roi_head import RoIHeads

from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class MaskRCNN(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=100,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=100,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
    ):
        super().__init__()
        
        if not hasattr(backbone, "out_channels"):
            raise ValueError("backbone should contain an attribute out_channels")

        out_channels = backbone.out_channels

        # 1. 创建RPN组件
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(train=rpn_pre_nms_top_n_train, test=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(train=rpn_post_nms_top_n_train, test=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        # 2. 创建RoI Heads组件
        # 边界框分支
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * 7 * 7, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        # 掩码分支
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=14,
                sampling_ratio=2
            )

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(
                mask_predictor_in_channels, mask_dim_reduced, num_classes
            )
        
        # 创建完整的RoI Heads
        self.roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            # 添加掩码组件
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor
        )

        
        
        # 4. 保存backbone
        self.backbone = backbone
        self._has_warned = False
    
    def postprocess(self,
                    result,               
                    image_shapes,          
                    original_image_sizes   
                    ):
        # If training -> return unchanged
        if self.is_training():
            return result

        # 遍历每张图片，恢复 boxes 到原始尺度，若包含 masks 则把 mask 粘回原图
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred.get("boxes", None)
            if boxes is None:
                continue

            boxes = resize_boxes(boxes, im_s, o_im_s)  # 把 boxes 缩放回原图尺度
            result[i]["boxes"] = boxes

            if "masks" in pred:
                masks = pred["masks"]
                # paste_masks_in_image 返回 jt.Var，形状 [num_obj,1,H_orig,W_orig]
                masks_in_image = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks_in_image

        return result

    def execute(self, images, targets=None,ori_size=None):
        if self.is_training() and targets is None:
            raise ValueError("In training mode, targets should be passed")
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.shape[-2:], targets)
        
        detections = self.postprocess(detections, [images.shape[-2:]], ori_size)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.is_training():
            return losses
        return detections


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def execute(self, x):
        x = x.flatten(1)
        x = nn.relu(self.fc6(x))
        x = nn.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def execute(self, x):
        if x.ndim == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        mods = []
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            mods.append(nn.Conv2d(
                next_feature, layer_features,
                kernel_size=3, stride=1,
                padding=dilation, dilation=dilation
            ))
            mods.append(nn.ReLU())
            next_feature = layer_features
        super().__init__(*mods)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            nn.ConvTranspose2d(in_channels, dim_reduced, kernel_size=2, stride=2),
            nn.ReLU(),
            # 只预测前景类的掩码（不包括背景）
            nn.Conv2d(dim_reduced, num_classes - 1, kernel_size=1, stride=1)
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


import numpy as np
from PIL import Image



def resize_boxes(boxes,  original_size, new_size ):
    """
    boxes: jt.Var or numpy array, shape (N,4), format [x1,y1,x2,y2]
    original_size: size where boxes are currently referenced to (h_orig, w_orig)
    new_size: desired output size (h_new, w_new)
    返回: jt.Var shape (N,4) scaled boxes
    """
    # ensure jt.Var
    if not isinstance(boxes, jt.Var):
        boxes = jt.array(boxes.astype(np.float32))

    
    h_orig, w_orig = int(original_size[0]), int(original_size[1])
    h_new, w_new = int(new_size[0]), int(new_size[1])

    # ratio = new / orig
    # 注意顺序：x 对应宽度，y 对应高度
    ratio_h = float(h_new) / float(h_orig)
    ratio_w = float(w_new) / float(w_orig)

    # boxes assumed as (N,4): x1,y1,x2,y2
    x1 = boxes[:, 0] * ratio_w
    y1 = boxes[:, 1] * ratio_h
    x2 = boxes[:, 2] * ratio_w
    y2 = boxes[:, 3] * ratio_h

    return jt.stack([x1, y1, x2, y2], dim=1)

def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # 显式同步并复制数据到 CPU
    masks_sync = masks.detach().cpu()
    boxes_sync = boxes.detach().cpu()
    
    if masks_sync.ndim == 4 and masks_sync.shape[1] == 1:
        masks_sync = masks_sync[:, 0]
    
    N = masks_sync.shape[0]
    im_h, im_w = int(img_shape[0]), int(img_shape[1])
    outs = []
    
    for i in range(N):
        m = masks_sync[i].numpy()  # 现在安全转换为 numpy
        b = boxes_sync[i].numpy()
        mask_i = paste_mask_in_image(m, b, im_h, im_w, padding)
        outs.append(mask_i)
    
    if outs:
        outs_np = np.stack(outs, axis=0).astype(np.float32)[:, None, :, :]
        return jt.array(outs_np)
    else:
        return jt.array(np.zeros((0, 1, im_h, im_w), dtype=np.float32))

def paste_mask_in_image(mask, box, im_h, im_w, padding=1):
    """
    mask: 2D numpy array (H_mask, W_mask) or jt.Var
    box: [x1,y1,x2,y2] (floats)
    returns 2D numpy array (H_image, W_image) with values 0/1 (float32)
    """
    # ensure numpy arrays
    if isinstance(mask, jt.Var):
        mask = mask.numpy()
    mask = np.asarray(mask, dtype=np.float32)

    # box to integers (we'll clamp to image)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or x1 >= im_w or y1 >= im_h:
        return np.zeros((im_h, im_w), dtype=np.uint8)
    # expand box by padding pixels (optional)
    x1 = int(np.floor(x1)) - padding
    y1 = int(np.floor(y1)) - padding
    x2 = int(np.ceil(x2)) + padding
    y2 = int(np.ceil(y2)) + padding

    # clip to image
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(im_w, x2)
    y2_clamped = min(im_h, y2)

    out_h = y2_clamped - y1_clamped
    out_w = x2_clamped - x1_clamped

    canvas = np.zeros((im_h, im_w), dtype=np.uint8)

    if out_h <= 0 or out_w <= 0:
        return canvas  # all zeros

    # resize mask to (out_h, out_w) using PIL for bilinear sampling
    # mask may be any scale; convert to PIL and resize
    try:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        # PIL resize expects size (width, height)
        resized = mask_img.resize((out_w, out_h), resample=Image.BILINEAR)
        resized_np = np.array(resized).astype(np.float32) / 255.0
    except Exception:
        # fallback: use numpy simple resize (nearest)
        resized_np = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((out_w, out_h))).astype(np.float32) / 255.0

    # threshold to binary mask (0/1)
    bin_mask = (resized_np >= 0.5).astype(np.uint8)

    # paste into canvas
    canvas[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = bin_mask

    return canvas
