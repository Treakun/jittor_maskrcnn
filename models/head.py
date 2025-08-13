import jittor as jt
from jittor import nn
from jittor import init

class BoxHead(nn.Module):
    def __init__(self, in_channels, num_classes, pooler_resolution):
        super().__init__()
        self.pooler_resolution = pooler_resolution
        flattened_size = in_channels * pooler_resolution * pooler_resolution
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        for l in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            init.gauss_(l.weight, 0, 0.01)
            init.constant_(l.bias, 0)

    def execute(self, x):
        # x: [N, C, pooler_resolution, pooler_resolution]
        x = x.flatten(1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, 2)
        self.mask_fcn_logits = nn.Conv2d(256, num_classes, 1)
        for l in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv, self.mask_fcn_logits]:
            init.gauss_(l.weight, 0, 0.01)
            init.constant_(l.bias, 0)

    def execute(self, x):
        # x: [N, C, M, M]
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            x = nn.relu(conv(x))
        x = nn.relu(self.deconv(x))
        return self.mask_fcn_logits(x)