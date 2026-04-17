---
title: "QNN 教程"
date: 2026-04-01T15:00:00+08:00
lastmod: 2026-04-15T21:20:00+08:00
draft: false
description: "围绕 Qualcomm AI Engine Direct 的架构、环境、量化流程与 Graph API 实战的初版教程"
slug: "qnn-tutorial"
tags: ["qnn"]
categories: ["qnn"]
build:
  list: never
comments: true
math: true
---

# QNN 教程

这组内容先按“能把事情串起来”的初版来写，不依赖当前环境真的跑通代码。重点不是穷举 API，而是把 QNN 在 Android 端落地时最容易卡住的几个环节讲清楚：对象模型、离线准备、量化、部署与排错。

## 适合谁看

- 已经有 `ONNX / TFLite / PyTorch` 模型，想部署到高通平台。
- 想理解 `QNN`、`HTP`、`Context Binary`、`Graph API` 分别是什么。
- 还没开始做性能优化，但希望一开始就站在正确的工程抽象上。

## 系列目录

| 文章 | 说明 |
|------|------|
| [核心概念与调用流程](/p/qnn-intro/) | QNN 的对象模型、Backend 体系，以及 Android 端从加载到执行的完整链路 |
| [环境配置](/p/qnn-setup/) | SDK、NDK、交叉编译、设备侧库文件与常见环境坑 |
| [模型量化与离线准备](/p/qnn-quantization/) | 从 ONNX 到量化模型库，再到 Context Binary 的推荐流程 |
| [Graph API 与 MatMul 示例](/p/qnn-gemm/) | 用一个可控的小图理解 Tensor、Node、量化参数与 HTP 部署思路 |

## 我建议的学习顺序

1. 先看 `qnn-intro`，把 `Backend / Device / Context / Graph / Tensor` 这几个对象关系理顺。
2. 再看 `qnn-setup`，明确宿主机、手机端、动态库和工具链分别放哪。
3. 然后看 `qnn-quantization`，把“转换 -> 量化 -> 生成模型库 -> 生成 context binary”这条链打通。
4. 最后看 `qnn-gemm`，把抽象概念落到最简单的手写图示例里。

## 系列定位

- 这不是 QNN API 参考手册。
- 这也不是一篇“复制命令就必定跑通”的实操记录。
- 它更像后续做真实项目时的一份工程地图，先把概念和路径搭起来，后面你再把设备细节、脚本和截图补进去。
