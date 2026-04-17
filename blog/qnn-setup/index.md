---
title: "QNN 环境配置"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-15T21:20:00+08:00
draft: false
description: "QNN SDK、Android NDK 与设备侧运行环境的初版配置指南"
slug: "qnn-setup"
tags: ["qnn"]
categories: ["qnn"]
comments: true
math: true
---

# QNN 环境配置

这一篇不追求“一个命令复制就跑通”，而是先把环境拆成三段来理解：

1. 宿主机工具链。
2. 交叉编译环境。
3. 手机端运行时环境。

真正卡人的地方通常不是第 1 步，而是第 2 步和第 3 步混在一起。

## 1. 先准备哪些东西

### 1.1 宿主机

- `QNN SDK`
- `Python`
- `CMake`
- `Android NDK`
- 一个能正常工作的 `adb`

### 1.2 目标设备

- 推荐较新的 Snapdragon 平台。
- 如果目标是 `HTP`，设备代际越新越省心。

## 2. 推荐的目录组织

我建议一开始就把目录分开，不要把所有产物塞在一个文件夹里：

```text
workspace/
├── qnn-sdk/
├── models/
├── build/
├── output/
└── device_push/
```

分别放：

- `qnn-sdk`：SDK 本体。
- `models`：原始模型和校准数据。
- `build`：交叉编译中间产物。
- `output`：模型库、context binary。
- `device_push`：准备推到手机上的文件。

## 3. 必要环境变量

最少要把 SDK 根目录和 NDK 根目录固定下来：

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
export ANDROID_NDK=/path/to/android-ndk
```

如果你习惯脚本化，也可以额外准备：

```bash
export PATH=$QNN_SDK_ROOT/bin:$PATH
```

## 4. 宿主机侧最先检查什么

先确认下面几件事：

- QNN 的转换工具是否能调用。
- NDK 是否能正常被 `cmake` 识别。
- `adb devices` 是否能看到真机。

如果这三件事还没稳定，不建议马上开始研究 HTP 细节。

## 5. Android 端你最终要推什么

很多初学者只推了可执行文件和模型，结果程序一运行就报动态库错误。更稳妥的思路是把设备侧文件分成三类：

- 你的可执行程序或 app 产物。
- QNN 运行时相关动态库。
- 模型产物，比如 model library、context binary、配置文件。

一个典型的设备侧临时目录可能长这样：

```text
/data/local/tmp/qnn_demo/
├── demo_binary
├── libQnnSystem.so
├── libQnnHtp.so
├── 其他后端依赖
├── model.so
└── model.bin
```

## 6. 交叉编译时我会先验证什么

在碰真实模型前，先验证下面两件事：

1. 能否用 NDK 编出一个最小 native 可执行程序。
2. 这个程序能否在手机端被正确加载，并找到需要的 `.so`。

原因很简单：如果连动态库加载链路都没打通，后面模型转换和量化结果根本没法验证。

## 7. 设备侧最常见的三个坑

### 7.1 找不到动态库

表象：

- 启动直接报 `dlopen failed`
- 找不到 `libQnn*.so`

先检查：

- 文件是不是都推到了同一目录。
- `LD_LIBRARY_PATH` 是否包含该目录。

### 7.2 后端版本不匹配

表象：

- 能加载主库，但执行时通信失败或初始化失败。

这通常不是代码逻辑问题，而是 SDK、设备和具体后端依赖不匹配。

### 7.3 你以为是 QNN 问题，其实是 NDK 问题

比如：

- ABI 不一致。
- STL 选择不一致。
- Android 平台版本过低。

所以一开始就把 `CMAKE_TOOLCHAIN_FILE`、`ANDROID_ABI`、`ANDROID_PLATFORM` 写死，会省很多时间。

## 8. 一个稳妥的初版流程

我建议按下面顺序推进：

1. 安装 SDK 和 NDK。
2. 确认 `adb`、交叉编译和动态库加载没问题。
3. 再去做模型转换。
4. 最后再碰 `HTP` 和量化。

原因是：前两步更像系统工程，后两步更像模型工程。混在一起排错非常痛苦。

## 9. 这一篇的结论

QNN 环境配置真正要解决的不是“把 SDK 解压了没有”，而是：

- 宿主机工具链是否稳定。
- 手机端运行时链路是否稳定。
- 你有没有把模型产物和运行时依赖分清楚。

只要这三件事理顺，后面的量化和 Graph API 才有意义。
