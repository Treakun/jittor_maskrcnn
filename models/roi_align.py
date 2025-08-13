import jittor as jt
from jittor import nn
import math


def roi_align(input: jt.Var,
              rois: jt.Var,
              output_size,
              spatial_scale: float = 1.0,
              sampling_ratio: int = 2,
              align_corners: bool = True,
              chunk_size: int = 64) -> jt.Var:
    """
    chunked roi_align: avoid allocating feat_repeat of size (K,C,H,W) at once.
    chunk_size: process rois in chunks of this size.
    """
    out_h, out_w = output_size
    N, C, H, W = input.shape

    K = int(rois.shape[0])
    if K == 0:
        return jt.zeros((0, C, out_h, out_w), dtype=input.dtype)

    s = int(sampling_ratio) if int(sampling_ratio) > 0 else 1

    batch_idx = rois[:, 0].int()
    fx1 = (rois[:, 1] * spatial_scale).astype(jt.float32)
    fy1 = (rois[:, 2] * spatial_scale).astype(jt.float32)
    fx2 = (rois[:, 3] * spatial_scale).astype(jt.float32)
    fy2 = (rois[:, 4] * spatial_scale).astype(jt.float32)

    roi_w = (fx2 - fx1).clamp(min_v=1e-6)
    roi_h = (fy2 - fy1).clamp(min_v=1e-6)

    bin_w = roi_w / float(out_w)
    bin_h = roi_h / float(out_h)

    # sampling relative positions
    i_idx = jt.arange(out_h, dtype=jt.float32).reshape((out_h,1))
    iy_offset = (jt.arange(s, dtype=jt.float32) + 0.5) / float(s)
    y_rel = (i_idx + iy_offset).reshape((-1,))  # (out_h * s,)

    j_idx = jt.arange(out_w, dtype=jt.float32).reshape((out_w,1))
    ix_offset = (jt.arange(s, dtype=jt.float32) + 0.5) / float(s)
    x_rel = (j_idx + ix_offset).reshape((-1,))  # (out_w * s,)

    # centers per roi (K, out_h*s) and (K, out_w*s)
    centers_y = fy1.reshape((-1,1)) + (y_rel.reshape((1,-1)) * bin_h.reshape((-1,1)))
    centers_x = fx1.reshape((-1,1)) + (x_rel.reshape((1,-1)) * bin_w.reshape((-1,1)))

    # prepare output
    out = jt.zeros((K, C, out_h, out_w), dtype=input.dtype)

    # process in chunks
    total_sample_h = out_h * s
    total_sample_w = out_w * s

    # helpers to normalize to [-1,1]
    def norm_x(x):
        return (x / (W - 1.0)) * 2.0 - 1.0 if W > 1 else jt.zeros_like(x)
    def norm_y(y):
        return (y / (H - 1.0)) * 2.0 - 1.0 if H > 1 else jt.zeros_like(y)

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        idx = slice(start, end)

        # take centers for this chunk
        c_x = centers_x[idx]  # (chunk, out_w*s)
        c_y = centers_y[idx]  # (chunk, out_h*s)

        # build per-chunk grids
        # grid_x: (chunk, out_h*s, out_w*s)
        grid_x = c_x[:, None, :].repeat(1, total_sample_h, 1)
        grid_y = c_y[:, :, None].repeat(1, 1, total_sample_w)

        grid_x_norm = norm_x(grid_x)
        grid_y_norm = norm_y(grid_y)
        grid = jt.stack([grid_x_norm, grid_y_norm], dim=3)  # (chunk, Hout, Wout, 2)

        # select corresponding batch indices for this chunk
        bidx_chunk = batch_idx[idx]
        # select feat maps per roi in chunk; this still creates (chunk, C, H, W) but chunk_size small
        feat_repeat = input[bidx_chunk]  # shape (chunk, C, H, W)

        sampled = nn.grid_sample(feat_repeat, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
        # sampled -> (chunk, C, out_h*s, out_w*s)
        sampled = sampled.reshape((end - start, C, out_h, s, out_w, s))
        pooled = sampled.mean(3).mean(4)  # (chunk, C, out_h, out_w)

        out[idx] = pooled

    return out

class MultiScaleRoIAlign(nn.Module):
    def __init__(self, featmap_names, output_size,
                 sampling_ratio: int, canonical_scale: int = 224, canonical_level: int = 4):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.output_size = tuple(output_size)
        self.sampling_ratio = sampling_ratio
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def execute(self, x, boxes, image_shapes):
        """
        x: dict of feature maps, keys in featmap_names, values jt.Var (B,C,H,W)
        boxes: list length B, each jt.Var (Ni,4) in image coords [x1,y1,x2,y2]
        image_shapes: list of (H, W)
        """
        # select feature maps
        x_filtered = [x[name] for name in self.featmap_names]

        # compute scales
        if not image_shapes:
            raise ValueError("image_shapes must not be empty")
        max_h = max([s[0] for s in image_shapes])
        max_w = max([s[1] for s in image_shapes])
        original_size = (max_h, max_w)

        scales = []
        for feat in x_filtered:
            fh, fw = feat.shape[-2], feat.shape[-1]
            scale_h = float(fh) / float(original_size[0])
            scale_w = float(fw) / float(original_size[1])
            # do not quantize to power of two
            scales.append((scale_h + scale_w) / 2.0)

        # compute level range
        lvl_min = -math.log2(scales[0]) if scales[0] > 0 else -100
        lvl_max = -math.log2(scales[-1]) if scales[-1] > 0 else 100

        # convert boxes list -> rois (K,5)
        rois = self._convert_to_roi_format(boxes)  # shape (K,5) or (0,5)

        if rois.shape[0] == 0:
            # empty
            C = x_filtered[0].shape[1]
            return jt.zeros((0, C, self.output_size[0], self.output_size[1]))

        # level mapping (per-roi)
        all_boxes = jt.concat(boxes, dim=0)
        widths = all_boxes[:,2] - all_boxes[:,0]
        heights = all_boxes[:,3] - all_boxes[:,1]
        scales_box = jt.sqrt(jt.maximum(widths * heights, 1e-6))
        target_lv = self.canonical_level + jt.log2(scales_box / float(self.canonical_scale))
        target_lv = jt.floor(target_lv)
        target_lv = jt.clamp(target_lv, min_v=int(lvl_min), max_v=int(lvl_max))
        levels = (target_lv - int(lvl_min)).int()  # 0-based indices

        # result buffer
        K = rois.shape[0]
        C = x_filtered[0].shape[1]
        out = jt.zeros((K, C, self.output_size[0], self.output_size[1]), dtype=x_filtered[0].dtype)

        # pool per level
        num_levels = len(x_filtered)
        for lvl in range(num_levels):
            idx = jt.where(levels == lvl)[0]
            if idx.shape[0] == 0:
                continue
            rois_per_level = rois[idx]  # (r,5)
            feat = x_filtered[lvl]
            spatial_scale = float(scales[lvl])
            pooled = roi_align(feat, rois_per_level, self.output_size, spatial_scale=spatial_scale, sampling_ratio=self.sampling_ratio)
            out[idx] = pooled

        return out

    def _convert_to_roi_format(self, boxes):
        """
        boxes: list length B of jt.Var (Ni,4)
        returns: jt.Var (K,5) float32
        """
        rois_list = []
        for b_idx, b in enumerate(boxes):
            if b is None or b.shape[0] == 0:
                continue
            # create batch index column
            idx_col = jt.full((b.shape[0],1), float(b_idx), dtype=b.dtype)
            rois_list.append(jt.concat([idx_col, b.astype(jt.float32)], dim=1))
        if not rois_list:
            return jt.zeros((0,5), dtype=jt.float32)
        return jt.concat(rois_list, dim=0)
