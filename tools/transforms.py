import random
import numpy as np
from PIL import Image
import jittor as jt

class Resize:
    """
    Resize image (PIL) so that its short side == min_size and long side <= max_size,
    while keeping aspect ratio. Also resize boxes, masks, area accordingly.

    target is expected to be a dict with keys:
      - 'boxes' : numpy array (N,4) in [x1,y1,x2,y2] (float)
      - 'masks' : numpy array (N,H,W) uint8 or float (optional)
      - 'area'  : numpy array (N,) float (optional)
      - (others left unchanged)
    """
    def __init__(self, min_size=800, max_size=1333, interpolation=Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def get_size(self, image):
        # PIL.Image.size -> (width, height)
        w, h = image.size
        return h, w  # return (height, width) for consistency with other code

    def __call__(self, image, target, is_train):
        """
        image: PIL.Image
        target: dict with numpy arrays (boxes, masks, area)
        """
        if not isinstance(image, Image.Image):
            # support numpy array input too
            image = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)

        orig_w, orig_h = image.size  # PIL: (width, height)
        orig_h = int(orig_h); orig_w = int(orig_w)

        short_side = min(orig_h, orig_w)
        long_side = max(orig_h, orig_w)

        # compute scale
        scale = float(self.min_size) / float(short_side)
        # if scaling would make long side exceed max_size, adjust scale
        if int(round(long_side * scale)) > self.max_size:
            scale = float(self.max_size) / float(long_side)

        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        # If no scale change, return original
        if new_h == orig_h and new_w == orig_w:
            # ensure orig_size stored
            if 'orig_size' not in target:
                target['orig_size'] = np.array([orig_h, orig_w], dtype=np.int64)
            target['size'] = np.array([new_h, new_w], dtype=np.int64)
            return image, target

        # resize image (PIL expects (width, height))
        image_resized = image.resize((new_w, new_h), resample=self.interpolation)
        if is_train:
            # adjust boxes
            if 'boxes' in target and target['boxes'] is not None:
                boxes = target['boxes']
                if isinstance(boxes, np.ndarray) and boxes.size:
                    boxes = boxes.astype(np.float32) * scale
                    target['boxes'] = boxes
                else:
                    # handle empty
                    target['boxes'] = np.zeros((0,4), dtype=np.float32)

            # adjust masks: expected (N, H, W)
            if 'masks' in target and target['masks'] is not None:
                masks = target['masks']
                if isinstance(masks, np.ndarray) and masks.size:
                    N = masks.shape[0]
                    resized_masks = []
                    for i in range(N):
                        m = masks[i]
                        # m: H_orig x W_orig
                        # convert to PIL and resize using NEAREST to preserve binary
                        m_img = Image.fromarray((m * 255).astype(np.uint8))
                        m_resized = m_img.resize((new_w, new_h), resample=Image.NEAREST)
                        m_np = np.array(m_resized).astype(np.uint8)
                        # binarize (in case antialiasing created shades)
                        m_bin = (m_np >= 128).astype(np.uint8)
                        resized_masks.append(m_bin)
                    target['masks'] = np.stack(resized_masks, axis=0)
                else:
                    # empty masks
                    target['masks'] = np.zeros((0, new_h, new_w), dtype=np.uint8)

            # adjust area if present (scale^2)
            if 'area' in target and target['area'] is not None:
                area = target['area']
                if isinstance(area, np.ndarray) and area.size:
                    target['area'] = area.astype(np.float32) * (scale ** 2)
                else:
                    target['area'] = np.zeros((0,), dtype=np.float32)

            # store original size (before transform) if not present
            if 'orig_size' not in target:
                target['orig_size'] = np.array([orig_h, orig_w], dtype=np.int64)
            # store new size
            target['size'] = np.array([new_h, new_w], dtype=np.int64)

        return image_resized, target


class ToTensor:
    """PIL Image -> jt.Var (C,H,W), float32, in [0,1]. Keep target as numpy dict."""
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            arr = np.array(image, dtype=np.float32) / 255.0  # H,W,C
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32) / 255.0
        else:
            # assume jt.Var already in some form
            if isinstance(image, jt.Var):
                return image, target
            else:
                raise TypeError("Unsupported image type in ToTensor: %s" % type(image))

        # HWC -> CHW
        arr = arr.transpose(2, 0, 1).astype(np.float32)  # (3,H,W)
        img_jt = jt.array(arr)
        return img_jt, target


class Normalize:
    """
    Normalize jt.Var image with mean/std. image expected as jt.Var (3,H,W), dtype float32.
    mean/std stored as numpy arrays to avoid early CUDA init, converted to jt.Var at call time.
    """
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
        self.std  = np.array(std  or [0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

    def __call__(self, image, target):
        if not isinstance(image, jt.Var):
            raise TypeError("Normalize expects image as jt.Var (C,H,W)")
        mean = jt.array(self.mean)
        std  = jt.array(self.std)
        image = (image - mean) / std
        return image, target


class RandomHorizontalFlip:
    """Flip image (jt.Var C,H,W) and adjust boxes/masks in target (numpy)."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if not isinstance(image, jt.Var):
                raise TypeError("RandomHorizontalFlip expects image as jt.Var (C,H,W)")

            _, h, w = image.shape
            image = image.flip(2)  # flip width axis

            if 'boxes' in target and target['boxes'] is not None and getattr(target['boxes'], "size", None) != 0:
                boxes = target['boxes']
                if isinstance(boxes, np.ndarray):
                    flipped = boxes.copy()
                    flipped[:, 0] = w - boxes[:, 2]
                    flipped[:, 2] = w - boxes[:, 0]
                    target['boxes'] = flipped
                else:
                    try:
                        b_np = np.array(boxes)
                        flipped = b_np.copy()
                        flipped[:, 0] = w - b_np[:, 2]
                        flipped[:, 2] = w - b_np[:, 0]
                        target['boxes'] = flipped
                    except Exception:
                        pass

            if 'masks' in target and target['masks'] is not None and getattr(target['masks'], "size", None) != 0:
                masks = target['masks']
                if isinstance(masks, np.ndarray):
                    target['masks'] = np.ascontiguousarray(np.flip(masks, axis=2))
                else:
                    try:
                        target['masks'] = masks.flip(2)
                    except Exception:
                        pass

        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,is_train):
        for t in self.transforms:
            if isinstance(t, Resize):
                image, target = t(image, target,is_train)
            else:
                image, target = t(image, target)
        return image, target
