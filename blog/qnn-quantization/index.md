---
title: "QNN 模型量化"
date: 2026-04-01T16:00:00+08:00
lastmod: 2026-04-01T16:00:00+08:00
draft: false
description: "QNN 模型量化流程详解"
slug: "qnn-quantization"
tags: ["qnn"]
categories: ["qnn"]

comments: true
math: true
---

# QNN 模型量化

本文介绍如何使用 QNN 工具将 PyTorch/ONNX 模型量化为 INT8 格式。

## 1. 量化概述

### 1.1 什么是量化

量化是将浮点模型转换为低精度整数模型：

$$
x_{int} = \text{round}\left(\frac{x_{float}}{scale}\right) + \text{offset}
$$

### 1.2 量化类型

| 类型 | 说明 |
|------|------|
| 训练后量化 (PTQ) | 训练后直接量化，无需重训练 |
| 量化感知训练 (QAT) | 训练时考虑量化误差，精度更高 |

## 2. 准备模型

### 2.1 导出 ONNX

```python
import torch
import torch.onnx

# 加载模型
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1,  3,  224,  224)
torch.onnx.export(
    model, 
    dummy_input, 
    'model.onnx', 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch'},  'output': {0: 'batch'}}
)
```

## 3. 使用 QNN 量化

### 3.1 命令行工具

```bash
# 模型转换
qnn-onnx-converter --input_model model.onnx --output_path model.cpp

# 模型量化
qnn-model-lib-generator \
    --model model.cpp \
    --backend libQnnHtp.so \
    --output_dir output
```

### 3.2 Python API

```python
import qnn
from qnn.quantization import PostTrainingQuantizer

# 加载模型
model = qnn.load_model('model.onnx')

# 准备校准数据
calib_data = load_calibration_data()  # 100-500 张图片

# 创建量化器
quantizer = PostTrainingQuantizer(
    model=model, 
    calibration_data=calib_data, 
    quant_scheme='int8'
)

# 执行量化
quantized_model = quantizer.quantize()

# 保存模型
quantized_model.save('model_quantized.bin')
```

## 4. 校准数据

### 4.1 准备校准集

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485,  0.456,  0.406], 
                        std=[0.229,  0.224,  0.225])
])

calib_dataset = datasets.ImageFolder('calibration_images',  transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_dataset,  batch_size=1)

# 校准数据要求
# - 数量：100-500 张
# - 分布：覆盖典型输入场景
# - 预处理：与推理时一致
```

### 4.2 校准策略

```python
# Min-Max 校准
quantizer = PostTrainingQuantizer(
    model=model, 
    calibration_data=calib_data, 
    quant_scheme='min_max'
)

# 熵校准（推荐）
quantizer = PostTrainingQuantizer(
    model=model, 
    calibration_data=calib_data, 
    quant_scheme='entropy'
)

# 百分位校准
quantizer = PostTrainingQuantizer(
    model=model, 
    calibration_data=calib_data, 
    quant_scheme='percentile', 
    percentile=99.9
)
```

## 5. 量化精度验证

```python
import numpy as np

def evaluate_quantization(model_fp32,  model_int8,  test_loader):
    """比较量化前后精度"""
    
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    
    for images,  labels in test_loader:
        # FP32 推理
        output_fp32 = model_fp32(images)
        pred_fp32 = output_fp32.argmax(dim=1)
        
        # INT8 推理
        output_int8 = model_int8(images)
        pred_int8 = output_int8.argmax(dim=1)
        
        correct_fp32 += (pred_fp32 == labels).sum().item()
        correct_int8 += (pred_int8 == labels).sum().item()
        total += labels.size(0)
    
    print(f"FP32 Accuracy: {100 * correct_fp32 / total:.2f}%")
    print(f"INT8 Accuracy: {100 * correct_int8 / total:.2f}%")
    print(f"Accuracy Drop: {100 * (correct_fp32 - correct_int8) / total:.2f}%")
```

## 6. 常见问题

### 6.1 精度下降过多

- 增加校准数据量
- 尝试熵校准
- 对敏感层使用 INT16 或 FP16

### 6.2 某些算子不支持量化

- 检查 QNN 支持的算子列表
- 对不支持的算子保持 FP32

### 6.3 动态 shape 问题

- 使用固定 input shape
- 或配置 dynamic shape 支持

---

## 参考链接

- [QNN 量化文档](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk/getting-started/quantization)
- [AIMET 量化工具](https://quic.github.io/aimet-pages/)

