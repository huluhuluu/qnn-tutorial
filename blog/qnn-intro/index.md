---
title: "QNN 简介"
date: 2026-04-01T15:00:00+08:00
lastmod: 2026-04-01T15:00:00+08:00
draft: false
description: "Qualcomm QNN 量化推理框架介绍"
slug: "qnn-intro"
tags: ["qnn"]
categories: ["qnn"]

comments: true
math: true
---

# QNN 简介

QNN (Qualcomm AI Engine Direct SDK) 是高通提供的 AI 推理框架，专为 Snapdragon 处理器优化。

## 1. 什么是 QNN

QNN 提供：
- 模型量化工具（INT8/INT16）
- 高效推理引擎
- 支持 CPU/GPU/DSP 异构计算
- Hexagon DSP 加速

### 1.1 QNN 架构

```
┌─────────────────────────────────────────┐
│             Application                  │
├─────────────────────────────────────────┤
│             QNN Runtime                  │
├──────────┬──────────┬──────────────────┤
│   CPU    │   GPU    │   Hexagon DSP     │
└──────────┴──────────┴──────────────────┘
```

### 1.2 核心组件

| 组件 | 说明 |
|------|------|
| QNN SDK | 开发工具包，包含量化工具和推理库 |
| QNN Runtime | 运行时库，执行推理 |
| QNN Ops | 算子库，支持常见神经网络算子 |
| Hexagon NN | DSP 推理引擎 |

## 2. 支持的模型

### 2.1 常见模型

| 模型类型 | 示例 |
|---------|------|
| 图像分类 | ResNet,  MobileNet,  EfficientNet |
| 目标检测 | YOLO,  SSD,  MobileDet |
| 语义分割 | DeepLab,  U-Net |
| NLP | BERT,  GPT (部分支持) |

### 2.2 量化支持

| 量化类型 | 说明 |
|---------|------|
| INT8 | 8位整数量化（推荐） |
| INT16 | 16位整数量化 |
| FP16 | 16位浮点（部分平台） |

## 3. QNN vs 其他框架

| 特性 | QNN | MNN | TFLite | NCNN |
|------|-----|-----|--------|------|
| 平台支持 | Snapdragon | 全平台 | 全平台 | 全平台 |
| DSP 加速 | ✅ | ✅ (部分) | ❌ | ❌ |
| 量化工具 | ✅ | ✅ | ✅ | ✅ |
| 自定义算子 | ✅ | ✅ | ✅ | ✅ |
| 开源程度 | 部分开源 | 完全开源 | 完全开源 | 完全开源 |

## 4. 开发流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  训练模型    │ -> │  模型量化    │ -> │  端侧部署    │
│ (PyTorch等)  │    │  (QNN工具)   │    │  (QNN推理)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 5. 适用场景

- **移动端 AI 应用**：手机端智能应用
- **IoT 设备**：边缘计算设备
- **汽车电子**：车载智能系统
- **机器人**：嵌入式智能控制

---

## 参考链接

- [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [QNN GitHub](https://github.com/quic/aimet-model-zoo)
- [Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)

