---
title: "QNN Graph API 与 MatMul 示例"
date: 2026-04-01T16:30:00+08:00
lastmod: 2026-04-15T21:20:00+08:00
draft: false
description: "用一个最小 MatMul 图理解 QNN Graph API、Tensor 描述和 HTP 路线"
slug: "qnn-gemm"
tags: ["qnn"]
categories: ["qnn"]
comments: true
math: true
---

# QNN Graph API 与 MatMul 示例

真正理解 QNN 的一个好办法，不是直接上完整模型，而是手写一个最小图。`MatMul` 刚好足够简单，又能把最关键的几个对象都串起来。

## 1. 为什么用 MatMul 做第一例子

因为它能让你只关注这些最核心的问题：

- Tensor 怎么描述。
- Node 怎么挂进 Graph。
- Graph finalize 到底意味着什么。
- 从 `CPU` 切到 `HTP` 时，哪些信息必须跟着变。

## 2. 最小图长什么样

逻辑非常简单：

```text
A (M x K)
   \
    MatMul -> C (M x N)
   /
B (K x N)
```

但真正要写进 QNN 的东西远不止数学公式，还包括：

- 每个 Tensor 的名字和 id。
- rank 和 dimensions。
- 数据类型。
- 输入输出的归属类型。
- 如果走量化，还要补 scale 和 zero-point。

## 3. 你实际会写的步骤

如果把 Graph API 的最小例子压缩成流程，大概就是：

1. 加载接口。
2. 创建 backend。
3. 创建 context。
4. 创建 graph。
5. 定义输入输出 tensor。
6. 定义 MatMul node。
7. finalize graph。
8. 准备输入输出 buffer。
9. execute。

这里最值得记住的是：`finalize` 是一个分界点。

- 在它之前，你在“描述图”。
- 在它之后，你在“执行图”。

## 4. Tensor 描述是这类例子的重点

很多 QNN 初学错误都发生在 Tensor 描述上，而不是 MatMul 公式本身。

至少要明确：

- 数据类型是不是对的。
- shape 顺序是不是对的。
- buffer 大小是不是和 shape 对齐。
- 如果是量化 Tensor，量化参数有没有带上。

## 5. 从 CPU 验证到 HTP 部署

我建议第一轮一定分两步：

### 5.1 先在 CPU backend 对齐

目标是确认：

- 图本身能建起来。
- 输入输出逻辑没错。
- 张量描述没错。

### 5.2 再切到 HTP backend

这时再开始考虑：

- 量化数据类型。
- 量化参数。
- 设备侧依赖库。
- HTP 通信和执行环境。

这样排错范围会小很多。

## 6. 如果切到 HTP，需要跟着改什么

最常见的变化有三类：

### 6.1 Backend 变了

加载的动态库从 `CPU` 相关实现切到 `HTP` 相关实现。

### 6.2 Tensor 类型变了

如果走量化路径，输入输出 Tensor 的数据类型和量化参数都要跟着改。

### 6.3 运行时依赖更多了

`HTP` 路线通常不只是一个主库，还会涉及额外的设备端依赖。部署时一定要把“程序、模型、运行时依赖”三类文件分开检查。

## 7. 这个例子真正想教会什么

不是“怎么把矩阵乘法跑到 HTP 上”，而是：

- Graph API 的心智模型是什么。
- 哪些问题属于图描述。
- 哪些问题属于后端部署。
- 哪些问题属于量化。

只要这个边界清楚，后面从 `MatMul` 换成卷积、激活、完整模型，难度不会一下子失控。

## 8. 一份务实的实验建议

等后面你要做真机实测时，可以按这个顺序补：

1. CPU 浮点正确性。
2. HTP 量化正确性。
3. 不同矩阵大小下的耗时。
4. finalize 开销与 execute 开销分开统计。

这样写出来的教程会比“只贴最终加速比”更有用。
