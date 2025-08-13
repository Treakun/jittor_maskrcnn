from typing import List, Optional, Dict, Tuple

import jittor as jt
from jittor import nn, Function
import jittor.nn as F
from jittor import init

from . import det_utils
from . import boxes as box_ops


class AnchorsGenerator(nn.Module):
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=jt.float32):
        scales = jt.array(scales, dtype=dtype)
        aspect_ratios = jt.array(aspect_ratios, dtype=dtype)
        h_ratios = jt.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios.unsqueeze(1) * scales.unsqueeze(0)).view(-1)
        hs = (h_ratios.unsqueeze(1) * scales.unsqueeze(0)).view(-1)

        base_anchors = jt.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype):
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            shifts_x = jt.arange(0, grid_width, dtype=jt.float32) * stride_width
            shifts_y = jt.arange(0, grid_height, dtype=jt.float32) * stride_height

            shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.view(-1)
            shift_y = shift_y.view(-1)

            shifts = jt.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.view(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def execute(self, image_list, feature_maps):
        
        grid_sizes = [ [int(s) for s in feature_map.shape[-2:]] for feature_map in feature_maps ]
        
        image_size = [int(s) for s in image_list.shape[-2:]]

        dtype = feature_maps[0].dtype

        
        strides = [
            [image_size[0] // g[0], image_size[1] // g[1]]
            for g in grid_sizes
        ]

        # 建立缓存 key：使用不可变的 tuple-of-tuples（不会触发任何张量计算）
        key = (tuple(tuple(g) for g in grid_sizes), tuple(tuple(s) for s in strides))
        if key in self._cache:
            anchors_over_all_feature_maps = self._cache[key]
        else:
           
            self.set_cell_anchors(dtype)
            anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
            self._cache[key] = anchors_over_all_feature_maps

        anchors = []
        
        for i in range(len(image_list)):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(jt.concat(anchors_in_image))

        
        self._cache.clear()

        return anchors



class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                init.gauss_(layer.weight, std=0.01)
                init.constant_(layer.bias, 0)

    def execute(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.view(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = jt.concat(box_cls_flattened, dim=1).view(-1, C)
    box_regression = jt.concat(box_regression_flattened, dim=1).view(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.4):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.is_training():
            return self._pre_nms_top_n['train']
        return self._pre_nms_top_n['test']

    def post_nms_top_n(self):
        if self.is_training():
            return self._post_nms_top_n['train']
        return self._post_nms_top_n['test']

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        targets=[targets]
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"].squeeze()
            if gt_boxes.numel() == 0:
                matched_gt_boxes_per_image = jt.zeros(anchors_per_image.shape, dtype=jt.float32)
                labels_per_image = jt.zeros((anchors_per_image.shape[0],), dtype=jt.float32)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min_v=0)]
                
                labels_per_image = (matched_idxs >= 0).float32()
                bg_inds = (matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD)
                labels_per_image[bg_inds] = 0.0
                discard_inds = (matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS)
                labels_per_image[discard_inds] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, dim=1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return jt.concat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        

        objectness = objectness.detach()
        objectness = objectness.view(num_images, -1)

        levels = []
        for idx, n in enumerate(num_anchors_per_level):
            levels.append(jt.full((n,), idx, dtype=jt.int64))
        levels = jt.concat(levels, 0).view(1, -1).expand_as(objectness)

        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = jt.arange(num_images)
        batch_idx = image_range.unsqueeze(1)

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = jt.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            
            keep = jt.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores


    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = jt.where(jt.concat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = jt.where(jt.concat(sampled_neg_inds, dim=0))[0]
        sampled_inds = jt.concat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = jt.concat(labels, dim=0)
        regression_targets = jt.concat(regression_targets, dim=0)
        num_pos = sampled_pos_inds.numel()
        if num_pos == 0:
            box_loss = jt.zeros((), dtype=pred_bbox_deltas.dtype)
        else:
            box_loss = det_utils.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1/9,
                size_average=False,
            ) / float(num_pos)

       

        objectness_loss = nn.binary_cross_entropy_with_logits(
            objectness[sampled_inds], 
            labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def execute(self, images, features, targets=None):
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head.execute(features)
        anchors = self.anchor_generator.execute(images, features)
        num_images = len(anchors)

        num_anchors_per_level = [
            o.shape[1] * o.shape[2] * o.shape[3] for o in objectness
        ]

        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(
            proposals, objectness, [images.shape[2:]], num_anchors_per_level
        )

        losses = {}
        if self.is_training():
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses