---
title: "QNN 核心概念与调用流程"
date: 2026-04-01T15:00:00+08:00
lastmod: 2026-04-15T21:20:00+08:00
draft: false
description: "QNN SDK 架构、Backend 体系、Graph API 与 Android 端完整调用流程"
slug: "qnn-intro"
tags: ["qnn"]
categories: ["qnn"]
comments: true
math: true
---

# QNN 核心概念与调用流程

如果你以前主要接触 `ONNX Runtime / TFLite / MNN`，第一次看 `QNN` 往往会觉得对象很多、名字很散。其实可以先把它理解成两层：

- 上层是统一的 C API。
- 下层是不同硬件后端的实现，最常见的是 `CPU / GPU / HTP`。

## 1. QNN 到底在抽象什么

QNN 想做的是把“模型描述”和“硬件执行”拆开。

```text
Application
    |
QNN Interface
    |
+----------------+----------------+----------------+
| CPU Backend    | GPU Backend    | HTP Backend    |
+----------------+----------------+----------------+
| Kryo CPU       | Adreno GPU     | Hexagon / HTP  |
+----------------+----------------+----------------+
```

这意味着上层逻辑可以尽量保持一致，而底层换到不同后端时，主要变化落在：

- 加载哪个 backend 动态库。
- 图能否被编译到目标硬件。
- Tensor 类型和量化参数是否符合该后端要求。

## 2. 五个最重要的对象

### 2.1 Backend

Backend 表示目标执行后端。

- `CPU` 适合调试和对齐结果。
- `GPU` 适合浮点路径。
- `HTP` 适合量化推理，也是手机端真正值得投入的方向。

### 2.2 Device

Device 表示具体的硬件设备和设备配置。很多时候你会把它看成“和后端配套的执行环境”。

### 2.3 Context

Context 可以理解成一次模型部署会话对应的资源容器。

- 它负责持有编译后的图和运行期资源。
- 你也可以把它理解成“模型实例”。

### 2.4 Graph

Graph 是实际执行的计算图。

- 可以通过 API 在线构图。
- 也可以提前离线生成，再运行时直接加载。

### 2.5 Tensor

Tensor 不只是数据地址，还包含：

- 数据类型。
- 维度信息。
- 量化参数。
- 内存类型和缓冲区描述。

真正容易出错的地方往往不是“图没建起来”，而是输入输出 Tensor 的数据类型和量化参数不匹配。

## 3. 推荐的心智模型

我建议把 QNN 的工作流记成下面这条链：

```text
加载 Backend
-> 创建设备 Device
-> 创建/加载 Context
-> 拿到 Graph
-> 准备输入输出 Tensor
-> 执行 Graph
-> 回收资源
```

如果是在线构图，会多出：

```text
Graph create -> add node -> finalize
```

如果是离线准备，则运行时会变成：

```text
contextCreateFromBinary -> graph execute
```

## 4. 在线构图和离线准备怎么选

### 4.1 在线构图

优点：

- 灵活，适合验证小图或自定义流程。
- 方便理解 Tensor 和 Node 的组织方式。

缺点：

- 启动慢。
- 图编译开销重。
- 更像开发态工作流，不像生产态工作流。

### 4.2 离线准备

优点：

- 冷启动更快。
- 部署流程更稳定。
- 更接近真实 Android 应用落地方式。

缺点：

- 调试不如在线构图直观。
- 需要先把转换工具链走通。

我的建议是：

- 学习阶段用在线构图理解对象关系。
- 真正部署阶段优先走离线准备。

## 5. 你真正会经历的完整链路

从模型到手机端执行，通常是下面这条链：

```text
PyTorch / ONNX
-> QNN converter
-> model library
-> context binary
-> Android app / native demo
```

每一步分别负责：

- `converter`：把模型描述转成 QNN 能理解的形式。
- `model library`：生成目标平台可加载的模型库。
- `context binary`：把图进一步编译成更接近目标硬件的运行时产物。

## 6. Android 端实际调用顺序

下面这条顺序比记单个 API 名字更重要：

```text
dlopen backend
-> 获取 QNN 接口
-> backendCreate
-> deviceCreate
-> contextCreate / contextCreateFromBinary
-> graphExecute
-> 读取输出
-> release
```

如果是 `HTP` 路线，还要把设备端动态库、stub/skel 和模型产物一起考虑，不然程序可能根本起不来。

## 7. 初学者最容易混淆的三件事

### 7.1 QNN 不是只做量化

QNN 不是“量化工具”，而是一整套推理执行框架。量化只是其中最常见、最有价值的一条路径。

### 7.2 Context Binary 不是普通模型文件

它更接近“已经为目标后端准备好的执行产物”，和单纯的 `onnx`、`tflite` 不是一个层次。

### 7.3 HTP 不是自动就会快

只有图能被正确量化、算子支持良好、内存和部署方式配套时，HTP 才真的有明显优势。

## 8. 一句话总结

看懂 QNN 的关键不是背 API，而是先把对象关系理顺：

- `Backend` 决定你跑在哪。
- `Context` 决定你加载了什么。
- `Graph` 决定你执行什么。
- `Tensor` 决定你喂进去和拿出来的到底是什么。

后面再去看环境配置、量化和 Graph API，思路会顺很多。
