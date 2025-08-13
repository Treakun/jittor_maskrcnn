from collections import OrderedDict
import jittor as jt
from jittor import nn



class IntermediateLayerGetter(nn.Module):

    def __init__(self, model, return_layers):
        super().__init__()
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.return_layers = {str(k): str(v) for k, v in return_layers.items()}
        remaining_layers = self.return_layers.copy()
        
        # 注册需要的子模块
        for name, module in model.named_children():
            self.add_module(name, module)
            if name in remaining_layers:
                del remaining_layers[name]
            if not remaining_layers:
                break

    def execute(self, x):
        out = OrderedDict()
        for name, module in self._modules.items():
            
            
            x = module(x)  
           
            
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    """
    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if re_getter:
            assert return_layers is not None
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def execute(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for object detection.
    """
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # 初始化参数
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x, idx):
        return self.inner_blocks[idx](x)

    def get_result_from_layer_blocks(self, x, idx):
        return self.layer_blocks[idx](x)

    def execute(self, x):
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = [self.get_result_from_layer_blocks(last_inner, -1)]

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            
            inner_top_down = nn.interpolate(
                last_inner, 
                size=feat_shape, 
                mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        return OrderedDict([(k, v) for k, v in zip(names, results)])


class LastLevelMaxPool(nn.Module):
    """
    Applies max_pool2d on top of the last feature map.
    """
    def execute(self, x, y, names):
        names.append("pool")
        # 使用正确的池化参数
        pooled = nn.pool(x[-1], kernel_size=1, stride=2, padding=0, op="maximum")
        x.append(pooled)
        return x, names