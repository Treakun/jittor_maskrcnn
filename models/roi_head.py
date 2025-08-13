from typing import Optional, List, Dict, Tuple

import jittor as jt
from jittor import nn
from jittor import init
from . import det_utils
from . import boxes as box_ops
from .roi_align import roi_align
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN (Jittor version).

    参数:
      class_logits: (N, num_classes) 预测的分类 logits
      box_regression: (N, num_classes*4) 或 (N, num_classes, 4) 的回归预测
      labels: list of jt.Var, 每张图采样后的标签 -> 将被 concat 成 (N,)
      regression_targets: list of jt.Var, 每张图采样后的回归目标 -> 将被 concat 成 (N,4)

    返回:
      classification_loss, box_loss
    """
    # 拼接每张图的数据
    labels = jt.concat(labels, dim=0).astype(jt.int64)               # (N,)
    regression_targets = jt.concat(regression_targets, dim=0)       # (N, 4)

    # 分类损失（直接对 concat 后的 logits 与 labels 计算）
    classification_loss = nn.cross_entropy_loss(class_logits, labels)

    # 找到正样本索引（label > 0）
    sampled_pos_inds_subset = jt.where(labels > 0)[0]
    num_pos = int(sampled_pos_inds_subset.numel())

    # box_regression 形状调整为 (N, num_classes, 4)
    N = class_logits.shape[0]
    num_classes = class_logits.shape[1]
    box_regression = box_regression.reshape(N, num_classes, 4)

    if num_pos == 0:
        # 没有正样本时，回归损失为 0（保持梯度图稳定）
        box_loss = jt.zeros((), dtype=box_regression.dtype)
    else:
        # 正确地选择对应 class 的回归预测：使用 sampled_pos_inds_subset 作为行索引，
        # labels_pos 作为列（类别）索引
        labels_pos = labels[sampled_pos_inds_subset].long()  # (num_pos,)
        box_regression_pos = box_regression[sampled_pos_inds_subset, labels_pos]  # (num_pos, 4)

        # 平滑 L1 损失（不平均）
        box_loss = det_utils.smooth_l1_loss(
            box_regression_pos,
            regression_targets[sampled_pos_inds_subset],
            beta=1/9,
            size_average=False,
        )
        # 按正样本数归一化（避免被所有 proposal 数稀释）
        box_loss = box_loss / float(max(1, num_pos))

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    """
    From the results of the CNN, post process the masks (Jittor version).
    """
    mask_prob = jt.sigmoid(x)
    boxes_per_image = [label.shape[0] for label in labels]
    if len(labels) > 0:
        labels_cat = jt.concat(labels, dim=0).long()
    else:
        # 没有任何 box 的情况，直接返回对应数量的空列表
        return [jt.zeros((0, 1, x.shape[2], x.shape[3])) for _ in boxes_per_image]
    num_masks = x.shape[0]
    
    # 创建索引张量

    idx = jt.arange(num_masks).long()
    selected = mask_prob[idx, labels_cat][:, None]
    masks_per_image = selected.split(boxes_per_image, dim=0) if len(boxes_per_image) > 0 else []
    
    return masks_per_image


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, discretization_size=None, sampling_ratio=2):

    if discretization_size is None:
        discretization_size = mask_logits.shape[-1]

    # For each image produce: per-positive labels and mask targets (cropped/resized)
    all_labels = []
    all_mask_targets = []

    
    # If inputs are provided concatenated, user should pass them as lists per image.
    for img_idx, (proposals_per_image, matched_idxs_per_image, gt_masks_per_image, gt_labels_per_image) in enumerate(
            zip(proposals, mask_matched_idxs, gt_masks, gt_labels)):
        # ensure tensors
        matched = matched_idxs_per_image
        # indices of positive proposals (matched >= 0)
        pos_inds = jt.where(matched >= 0)[0]
        if pos_inds.shape[0] == 0:
            continue

        # matched gt indices for those positive proposals
        matched_gt_inds = matched[pos_inds].astype(jt.int32)

        # gather labels for these matched gts
        # gt_labels_per_image is (G,) (int class ids)
        labels_pos = gt_labels_per_image.squeeze()[matched_gt_inds]            # shape (Np,)
        all_labels.append(labels_pos)

        # gather mask targets: project ground-truth masks of the matched gts into box crops
        # proposals_per_image[pos_inds] shape (Np,4)
        boxes_pos = proposals_per_image[pos_inds].astype(jt.float32)
        mask_targets_pos = project_masks_on_boxes(
            gt_masks_per_image.squeeze(), boxes_pos, matched_gt_inds, discretization_size, sampling_ratio=sampling_ratio
        )  # (Np, M, M), float32 in {0,1}
        all_mask_targets.append(mask_targets_pos)

    if len(all_labels) == 0:
        # no positive samples -> zero loss
        return mask_logits.sum() * 0.0

    labels = jt.concat(all_labels, dim=0).astype(jt.int64)           # (N_total_pos,)
    mask_targets = jt.concat(all_mask_targets, dim=0).astype(jt.float32)  # (N_total_pos, M, M)

    # mask_logits shape: (N_total_pos, num_classes, M, M)
    # pick the predicted mask for the ground-truth class for each instance
    idx = jt.arange(labels.shape[0], dtype=jt.int64)
    # select per-instance predicted logits: shape (N_pos, M, M)
    pred_logits = mask_logits[idx, labels]   # (N_pos, M, M)

    # binary cross entropy with logits
    loss = nn.binary_cross_entropy_with_logits(pred_logits, mask_targets)
    return loss

def project_masks_on_boxes(gt_masks, boxes, matched_gt_inds, M, sampling_ratio=2):
    """
    参数：
      boxes: jt.Var (Np, 4)  (x1,y1,x2,y2) in image coords
      matched_gt_inds: jt.Var (Np,) int indices into gt_masks (每个 proposal 对应的 gt idx)
      M: 输出 mask 大小 (int)
    返回:
      jt.Var (Np, M, M) 二值化后的 mask targets (float32)
    """

    

    # matched_gt_inds -> 1D integer python list to avoid fancy indexing issues
    # ensure shape (Np,)
    if isinstance(matched_gt_inds, jt.Var):
        matched_gt_inds = matched_gt_inds.flatten().astype(jt.int32)
        idx_list = [int(x) for x in matched_gt_inds.numpy().tolist()]
    else:
        idx_list = list(matched_gt_inds)

    Np = boxes.shape[0]
    if Np == 0:
        return jt.zeros((0, M, M), dtype=jt.float32)

    # 2) 安全地按 idx_list 索引，生成 (Np, H, W)
    # 使用 python 索引并堆叠，避免 jittor 的高级索引在某些版本出现 shape 异常
    selected = []
    for i in idx_list:
        # defensive: clamp i in [0,G-1]
        i_clamped = max(0, min(i, gt_masks.shape[0]-1))
        selected.append(gt_masks[i_clamped])
    # stack -> (Np, H, W)
    selected_masks = jt.stack(selected, dim=0).astype(jt.float32)

    # 3) 作为单通道图像 (Np,1,H,W)
    selected_masks = selected_masks.unsqueeze(1)  # (Np,1,H,W)

    # 4) 构造 rois，使每个 roi 对应选中的 mask（batch idx = 0..Np-1）
    batch_ids = jt.arange(Np, dtype=jt.float32).reshape(-1,1)
    rois = jt.concat([batch_ids, boxes.astype(jt.float32)], dim=1)  # (Np,5)

    # 5) 用 roi_align 把每个 mask crop 到 (M,M)
    resized = roi_align(
        selected_masks,       # (Np,1,H,W)
        rois,                 # (Np,5) with batch idx mapping
        output_size=(M, M),
        spatial_scale=1.0,
        sampling_ratio=sampling_ratio,
        align_corners=True
    )  # (Np, 1, M, M)

    resized = resized[:, 0, :, :]  # (Np, M, M)
    # 6) 二值化
    resized = (resized >= 0.5).astype(jt.float32)

    return resized

class RoIHeads(nn.Module):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 fg_iou_thresh=0.5,
                 bg_iou_thresh=0.5,
                 batch_size_per_image=256,
                 positive_fraction=0.25,
                 bbox_reg_weights=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detection_per_img=100,
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    def has_mask(self):
        return all([
            self.mask_roi_pool is not None,
            self.mask_head is not None,
            self.mask_predictor is not None
        ])

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # 处理没有GT框的情况
                clamped_matched_idxs_in_image = jt.zeros(
                    (proposals_in_image.shape[0],), dtype=jt.int64
                )
                labels_in_image = jt.zeros(
                    (proposals_in_image.shape[0],), dtype=jt.int64
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, 
                    proposals_in_image
                )
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min_v=0)
                
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.int64()

                # 设置背景标签
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                
                # 设置忽略标签
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        
        for pos_inds_img, neg_inds_img in zip(sampled_pos_inds, sampled_neg_inds):
            img_sampled_inds = jt.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
            
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        return [
            jt.concat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

    

    def select_training_samples(self, proposals, targets):
        
        if targets is None:
            raise ValueError("target should not be None.")
            
        dtype = proposals[0].dtype
        
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]
        
        # 添加GT框到proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        
        # 分配目标
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        
        # 采样
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        
        for img_id in range(len(proposals)):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = jt.zeros((1, 4), dtype=dtype)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        
        # 计算回归目标
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        
        pred_scores = nn.softmax(class_logits, dim=-1)
        
        pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes=boxes.squeeze(1)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            
            # 创建标签
            labels = jt.arange(num_classes)
            labels = labels.view(1, -1).expand(scores.shape)
            
            # 移除背景类
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            # 展平所有预测
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            
            # 移除低分框
            inds = jt.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            

            # NMS处理
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            
            # 保留前k个检测结果
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            
        return all_boxes, all_scores, all_labels

    def execute(self, features, proposals, image_shapes, targets=None):
        image_shapes=[image_shapes]
        if self.is_training():
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # ROI特征提取
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head.execute(box_features)
        class_logits, box_regression = self.box_predictor.execute(box_features)
        
        result = []
        losses = {}
        
        if self.is_training():
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
           

            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            for i in range(len(boxes)):
                result.append({
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                })
        
        # 掩码分支处理
        if self.has_mask():
            if self.is_training():
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(len(proposals)):
                    pos = jt.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                mask_proposals = [p["boxes"] for p in result]
                pos_matched_idxs = None
                
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head.execute(mask_features)
            mask_logits = self.mask_predictor.execute(mask_features)
            targets=[targets]
            if self.is_training():
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                losses["loss_mask"] = rcnn_loss_mask
            else:
                labels = [r["labels"] for r in result]
                mask_probs = maskrcnn_inference(mask_logits, labels)
                for i, r in enumerate(result):
                    r["masks"] = mask_probs[i]
        
        return result, losses