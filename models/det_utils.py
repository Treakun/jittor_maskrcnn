import jittor as jt
import math
from typing import List, Tuple
from jittor import Var
import jittor as jt
import numpy as np
from typing import List, Tuple

class BalancedPositiveNegativeSampler:
    """
    平衡正负样本采样器
    返回每张图的二值掩码（jt.Var），用于从预测中采样正负样本。
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[jt.Var]) -> Tuple[List[jt.Var], List[jt.Var]]:
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image in matched_idxs:
            # matched_idxs_per_image: jt.Var, shape (M,)
            # positive/negative 索引（jt.Var）
            positive = jt.where(matched_idxs_per_image >= 1)[0]
            negative = jt.where(matched_idxs_per_image == 0)[0]

            # 计算要采样的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(int(positive.numel()), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(int(negative.numel()), num_neg)

            # 随机采样正负索引（如果要采样数量为0则返回空 jt.Var）
            if num_pos > 0:
                perm1 = jt.randperm(int(positive.numel()))[:num_pos]
                pos_idx_per_image = positive[perm1]
            else:
                pos_idx_per_image = jt.array([], dtype=jt.int64)

            if num_neg > 0:
                perm2 = jt.randperm(int(negative.numel()))[:num_neg]
                neg_idx_per_image = negative[perm2]
            else:
                neg_idx_per_image = jt.array([], dtype=jt.int64)

            # 构造二进制掩码：在 numpy 中设置 True，然后转换为 jt.Var
            M = int(matched_idxs_per_image.numel())
            pos_mask_np = np.zeros((M,), dtype=np.bool_)
            neg_mask_np = np.zeros((M,), dtype=np.bool_)

            if pos_idx_per_image.numel() > 0:
                # 转成 numpy 索引（注意：这些索引不会参与反向传播）
                pos_np = pos_idx_per_image.numpy().astype(np.int64)
                pos_mask_np[pos_np] = True

            if neg_idx_per_image.numel() > 0:
                neg_np = neg_idx_per_image.numpy().astype(np.int64)
                neg_mask_np[neg_np] = True

            pos_mask = jt.array(pos_mask_np)
            neg_mask = jt.array(neg_mask_np)

            pos_idx.append(pos_mask)
            neg_idx.append(neg_mask)

        return pos_idx, neg_idx


def encode_boxes(reference_boxes: Var, proposals: Var, weights: Var) -> Var:
    """
    将边界框编码为回归参数

    参数:
        reference_boxes (Var): 参考框(真实框)
        proposals (Var): 待编码的框(锚框)
        weights (Var): 编码权重

    返回:
        targets (Var): 编码后的回归参数
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]

    # 解包提案框坐标
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # 解包参考框坐标
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # 计算提案框的宽度、高度和中心点
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    # 计算参考框的宽度、高度和中心点
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # 计算回归目标
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * jt.log(gt_widths / ex_widths)
    targets_dh = wh * jt.log(gt_heights / ex_heights)

    # 连接所有目标值
    targets = jt.concat([targets_dx, targets_dy, targets_dw, targets_dh], dim=1)
    return targets


class BoxCoder:
    """
    边界框编码器
    将边界框编码为回归参数，以及将回归参数解码为边界框
    """

    def __init__(self, weights: Tuple[float, float, float, float], 
                 bbox_xform_clip: float = math.log(1000. / 16)):
        """
        参数:
            weights (4元组): 编码权重
            bbox_xform_clip (float): 边界框变换的截断值
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Var], proposals: List[Var]) -> List[Var]:
        """
        将参考框编码为相对于提案框的回归参数
        
        参数:
            reference_boxes: 每张图像的参考框列表
            proposals: 每张图像的提案框列表
            
        返回:
            编码后的回归参数列表
        """
        # 拼接所有图像的数据
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = jt.concat(reference_boxes, dim=0)
        proposals = jt.concat(proposals, dim=0)
        
        # 编码并分割回各图像
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Var, proposals: Var) -> Var:
        """
        单批次编码
        
        参数:
            reference_boxes: 参考框
            proposals: 提案框
            
        返回:
            编码后的回归参数
        """
        dtype = reference_boxes.dtype
        weights = jt.array(self.weights, dtype=dtype)
        return encode_boxes(reference_boxes, proposals, weights)

    def decode(self, rel_codes: Var, boxes: List[Var]) -> Var:
        """
        将回归参数解码为边界框
        
        参数:
            rel_codes: 回归参数
            boxes: 基础框(锚框/提案框)列表
            
        返回:
            解码后的边界框
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, Var)
        
        # 拼接所有框
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = jt.concat(boxes, dim=0)
        total_boxes = sum(boxes_per_image)

        # 解码
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        
        # 重塑为每张图像的边界框
        if total_boxes > 0:
            pred_boxes = pred_boxes.reshape(total_boxes, -1, 4)
        
        return pred_boxes

    def decode_single(self, rel_codes: Var, boxes: Var) -> Var:
        # rel_codes: (N, 4)
        # boxes: (N, 4)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0] / wx
        dy = rel_codes[:, 1] / wy
        dw = rel_codes[:, 2] / ww
        dh = rel_codes[:, 3] / wh

        dw = jt.clamp(dw, max_v=self.bbox_xform_clip)
        dh = jt.clamp(dh, max_v=self.bbox_xform_clip)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = jt.exp(dw) * widths
        pred_h = jt.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        return jt.stack([x1, y1, x2, y2], dim=1)  # (N,4)



class Matcher:
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold: float, low_threshold: float, 
                 allow_low_quality_matches: bool = False):
        """
        匹配器
        
        参数:
            high_threshold (float): 高阈值，大于等于此值的匹配为候选
            low_threshold (float): 低阈值，用于分层匹配
            allow_low_quality_matches (bool): 是否允许低质量匹配
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Var) -> Var:
        """
        计算匹配
        
        参数:
            match_quality_matrix (Var): M(gt) x N(预测)的匹配质量矩阵
            
        返回:
            matches (Var): 长度为N的张量，表示每个预测的匹配结果
        """
        # 处理空输入
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("训练时缺少真实框")
            else:
                raise ValueError("训练时缺少提案框")
        
        # 寻找每个预测框的最佳匹配
        matches, matched_vals = match_quality_matrix.argmax(dim=0)
        


        # 如果需要低质量匹配，保存原始匹配
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        
        # 标记低于低阈值的匹配
        below_low_threshold = matched_vals < self.low_threshold
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        
        # 标记在阈值之间的匹配
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        
        # 处理低质量匹配
        if self.allow_low_quality_matches:
            self.set_low_quality_matches(matches, all_matches, match_quality_matrix)
        
        return matches

    def set_low_quality_matches(self, matches: Var, all_matches: Var, 
                                match_quality_matrix: Var):
        """
        设置低质量匹配
        
        参数:
            matches: 当前匹配结果
            all_matches: 原始匹配结果
            match_quality_matrix: 匹配质量矩阵
        """
        # 寻找每个真实框的最佳匹配
        _, highest_quality_foreach_gt = match_quality_matrix.argmax(dim=1)
        
        # 寻找所有最高质量匹配
        gt_pred_pairs_of_highest_quality = jt.where(
            match_quality_matrix == highest_quality_foreach_gt.unsqueeze(1)
        )
        
        # 更新匹配结果
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def smooth_l1_loss(input: Var, target: Var, beta: float = 1./9, 
                   size_average: bool = True) -> Var:
    """
    平滑L1损失函数
    
    参数:
        input: 预测值
        target: 目标值
        beta: 平滑参数
        size_average: 是否平均损失
        
    返回:
        损失值
    """
    n = jt.abs(input - target)
    cond = n < beta
    loss = jt.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    
    if size_average:
        return loss.mean()
    return loss.sum()