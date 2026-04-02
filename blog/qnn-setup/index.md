---
title: "QNN 环境配置"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-01T15:30:00+08:00
draft: false
description: "QNN SDK 安装与配置指南"
slug: "qnn-setup"
tags: ["qnn"]
categories: ["qnn"]

comments: true
math: true
---

# QNN 环境配置

本文介绍如何安装和配置 QNN SDK 开发环境。

## 1. 系统要求

### 1.1 开发环境

- Ubuntu 20.04/22.04 或 Windows 10/11
- Python 3.8+
- CMake 3.13+
- Android NDK (Android 开发)

### 1.2 目标设备

- Snapdragon 8 Gen 1 及以上（推荐）
- Snapdragon 888 及以上
- 其他 Snapdragon 处理器（部分功能支持）

## 2. 下载 QNN SDK

从 Qualcomm 开发者网站下载：

```
https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk
```

需要注册 Qualcomm 开发者账号。

## 3. 安装 QNN SDK

### 3.1 Linux

```bash
# 解压 SDK
unzip qnn-sdk-linux.zip -d ~/qnn

# 设置环境变量
export QNN_SDK_ROOT=~/qnn/QNN
export PATH=$PATH:$QNN_SDK_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$QNN_SDK_ROOT/lib

# 添加到 ~/.bashrc
echo 'export QNN_SDK_ROOT=~/qnn/QNN' >> ~/.bashrc
echo 'export PATH=$PATH:$QNN_SDK_ROOT/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$QNN_SDK_ROOT/lib' >> ~/.bashrc
```

### 3.2 Windows

```powershell
# 解压到目标目录
Expand-Archive qnn-sdk-windows.zip -DestinationPath C:\qnn

# 设置环境变量
$env:QNN_SDK_ROOT = "C:\qnn\QNN"
$env:Path += ";C:\qnn\QNN\bin"

# 永久设置
[Environment]::SetEnvironmentVariable("QNN_SDK_ROOT",  "C:\qnn\QNN",  "User")
```

## 4. 验证安装

```bash
# 检查版本
qnn-version

# 查看支持的算子
qnn-op-list

# 运行示例
cd $QNN_SDK_ROOT/examples
python run_example.py
```

## 5. Python 环境

```bash
# 创建虚拟环境
python -m venv qnn-env
source qnn-env/bin/activate

# 安装依赖
pip install torch onnx onnxruntime
pip install $QNN_SDK_ROOT/python/qnn-*.whl
```

## 6. Android 交叉编译

```bash
# 设置 NDK 路径
export ANDROID_NDK=/path/to/ndk

# 交叉编译示例
cd $QNN_SDK_ROOT/examples
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-30
make -j4
```

## 7. 常见问题

### 7.1 找不到 QNN 库

确保 `LD_LIBRARY_PATH` 包含 QNN lib 目录。

### 7.2 Python 绑定错误

确保 Python 版本兼容，并正确安装 wheel 包。

### 7.3 设备不支持

部分功能需要较新的 Snapdragon 处理器。

---

## 参考链接

- [QNN SDK 下载](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [QNN 示例代码](https://github.com/quic/aimet-model-zoo)

