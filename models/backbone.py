import os
import jittor as jt
from jittor import nn, Module
from typing import List, Optional, Type

class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        # 使用标准的BatchNorm2d
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channel, out_channel, 
                              kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = norm_layer(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                              kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet主干网络"""
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))
        
        layers = []
        layers.append(block(self.in_channel, channel, stride=stride,
                          downsample=downsample))
        self.in_channel = channel * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = jt.flatten(x, 1)
            x = self.fc(x)
            
        return x


def resnet50_fpn_backbone(
    pretrain_path: str = "",
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[nn.Module] = None
) :
    """
    创建ResNet50-FPN骨干网络（无冻结层）
    
    参数:
        pretrain_path: 预训练权重路径（不再使用）
        returned_layers: 返回的特征图层 (默认为[1,2,3,4])
        extra_blocks: 额外的FPN块
        
    返回:
        BackboneWithFPN实例
    """
    # 创建ResNet50主干
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False)
    
    # 设置默认额外块
    if extra_blocks is None:
        from .feature_pyramid_network import LastLevelMaxPool
        extra_blocks = LastLevelMaxPool()
    
    # 设置返回层 (默认为所有层)
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5, \
        "returned_layers must be between 1 and 4"
    
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256  # FPN输出通道数
    
    # 创建FPN骨干
    from .feature_pyramid_network import BackboneWithFPN
    return BackboneWithFPN(
        backbone=resnet_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks
    )