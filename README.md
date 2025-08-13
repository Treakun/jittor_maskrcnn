# Mask R-CNN Jittor 实现

本仓库使用 [Jittor](https://github.com/Jittor/jittor) 深度学习框架实现了 Mask R-CNN 模型，并在 COCO 数据集上进行了训练和验证。

## 环境配置

环境配置遵循 [Jittor 官方安装指南](https://github.com/Jittor/jittor#install)。请参考官方文档完成 Jittor 框架的安装。

## 数据集

使用 COCO 2017 数据集，仅使用 **2%** 的训练/验证数据：
```
data/coco/
├── annotations
└── images
    ├── train2017  
    └── val2017      
```

## 训练脚本

```bash
python tools/train.py
```

## 测试脚本

使用训练好的模型进行测试：
```bash
python tools/test.py \
    --path data/coco/images/test2017 \
    --model outputs/model_10.pkl \
    --out test_results
```

## 日志文件

所有训练和验证日志以及训练loss保存在 `logs/` 目录

## 实验配置

| 参数          | 值               |
|---------------|------------------|
| 数据集        | COCO 2017 (2%子集) |
| 训练周期      | 10 epochs        |
| Batch Size    | 1                |
| 初始学习率    | 0.0004           |
| 优化器        | SGD with Momentum|

## 实验结果

代码可能仍然存在一些bug，导致验证和测试结果很差。

## 与 PyTorch 实现对齐

本实现与 PyTorch 版本在以下方面保持一致：
1. 数据预处理与增强策略
2. 网络架构 (ResNet-50-FPN)
3. 损失函数设计
4. 优化器配置 (SGD with Momentum)
