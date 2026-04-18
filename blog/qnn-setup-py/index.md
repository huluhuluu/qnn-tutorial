---
title: "QNN 环境准备与 python实例"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-17T14:10:00+08:00
draft: false
description: "从宿主机开始，以最小 MatMul 模型串起 QNN 环境、模型转换、模型库生成、Android 真机推送与端侧执行"
slug: "qnn-setup"
tags: ["qnn"]
categories: ["qnn"]
comments: true
math: true
---

# QNN 环境准备与 MatMul 端侧实例

这篇把 [QNN 介绍](../qnn-intro/index.md)和[QNN实例](../qnn-setup/index.md) 里的在线构图执行方式通过`QNN` 提供的`python`工具链实现

- 执行环境如下
  - 宿主机：`Ubuntu 22.04 x86_64`
  - QNN SDK：`2.40.0.251030`, `python 3.10 ABI`
  - Android NDK：`android-ndk-r29`
  - SoC：`SnapDragon 8 elite`, `Chip: SM8750`, `Hexagon Arch: v79`
  - Backend：`CPU/GPU/HTP`

## 1. 环境配置

使用 `QNN` 的 `python` 工具链，执行
```bash
cd /root/qairt/2.40.0.251030/bin # 根据实际路径修改
source envsetup.sh 

# 这会设置python环境变量，确保后续命令能找到正确的python依赖
➜  ✗ echo $PYTHONPATH    
/root/qairt/2.40.0.251030/lib/python/:/root/qairt/2.40.0.251030/benchmarks/QNN/
```

接下来的操作依赖`pytorch`包，通过脚本检查环境：
```bash
# 查看环境检查结果
➜  ✗ ${QAIRT_SDK_ROOT}/bin/envcheck -a
Checking Android NDK Environment
--------------------------------------------------------------
[INFO] Found ndk-build at /root/android-ndk/android-ndk-r29/ndk-build and ANDROID_NDK_ROOT is also set.
--------------------------------------------------------------

Checking Clang Environment
--------------------------------------------------------------
[INFO] Found clang++ at /usr/bin/clang++
--------------------------------------------------------------

Checking TensorFlow Environment
--------------------------------------------------------------
[ERROR] Unable to import tensorflow using python3.
--------------------------------------------------------------

Checking TFLite Environment
--------------------------------------------------------------
[ERROR] Unable to import tflite using python3.
--------------------------------------------------------------

Checking ONNX Environment
--------------------------------------------------------------
[ERROR] Unable to import onnx using python3.
--------------------------------------------------------------

Checking PyTorch Environment
--------------------------------------------------------------
PyTorch is set-up successfully
--------------------------------------------------------------
```

可以通过`python`虚拟环境管理包，依赖版本在[官方文档](https://docs.qualcomm.com/nav/home/linux_setup.html?product=1601111740010412#step-3-install-model-frameworks)标出
```bash
conda create -n qnn python=3.10 -y  # 创建python3.10的虚拟环境
conda activate qnn  # 激活虚拟环境
export UV_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple/ # 设置国内镜像源
uv pip install torch==1.13.1 
```

## 2. 执行流程

`QNN` 提供了从模型转换到设备执行的`python`全链路工具，流程如下：
```text
宿主机环境
-> 生成 MatMul 计算图
-> 转成 QNN model.cpp/.bin
-> 生成 aarch64-android 模型库
-> 推到 Android 设备
-> 用 qnn-net-run 执行
-> 回拉输出并和 NumPy 基线对比
```


### 2.1 生成 MatMul 计算图

目前计算图最常用的是 `ONNX` 格式，下面的脚本使用`pytorch`生成一个最小的 `MatMul` 计算图并且导出。

```python
from pathlib import Path

import numpy as np
import torch

root = Path("matmul-demo/model")
root.mkdir(parents=True, exist_ok=True)

# 设置随机数种子，确保每次生成的输入和权重都一样
rng = np.random.default_rng(0)
A = rng.normal(size=(1, 128)).astype(np.float32)
B = rng.normal(size=(128, 64)).astype(np.float32)
bias = rng.normal(size=(64,)).astype(np.float32)

# 定义 MatMul 模型
class MatMulModel(torch.nn.Module):
    def __init__(self, weight: np.ndarray, bias_value: np.ndarray):
        super().__init__()
        self.register_buffer("weight", torch.from_numpy(weight))
        self.register_buffer("bias", torch.from_numpy(bias_value))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

model = MatMulModel(B, bias).eval()
input_tensor = torch.from_numpy(A)

with torch.no_grad():
    Y = model(input_tensor).cpu().numpy()
# 导出 ONNX 模型
torch.onnx.export(
    model,
    input_tensor,
    str(root / "matmul.onnx"),
    input_names=["A"],
    output_names=["Y"],
    opset_version=13,
    do_constant_folding=False,
)
# 保存输入和参考输出
A.tofile(root / "input.raw")
np.save(root / "reference.npy", Y)
(root / "input_list.txt").write_text(str((root / "input.raw").resolve()) + "\n")
```

### 2.2 转成 QNN 计算图格式
```bash
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network matmul-demo/model/matmul.onnx \
  --output_path matmul-demo/build/fp32/matmul_fp32.cpp
```


###  2.3 编译

```bash
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c matmul-demo/build/fp32/matmul_fp32.cpp \
  -b matmul-demo/build/fp32/matmul_fp32.bin \
  -t aarch64-android \
  -o matmul-demo/build/fp32/model_libs
```

### 2.4 执行

下面统一测试 `fp32` 精度，并分别在 `CPU/GPU/NPU` 三个后端执行。

首先推送需要的动态库、执行文件、模型库和输入数据到设备：

```bash
export DEVICE_ROOT=/data/local/tmp/qnn_matmul
adb -s 127.0.0.1:40404 shell "mkdir -p $DEVICE_ROOT/cpu $DEVICE_ROOT/gpu $DEVICE_ROOT/npu"

adb -s 127.0.0.1:40404 push \
  $QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run \
  $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnCpu.so \
  matmul-demo/build/fp32/model_libs/aarch64-android/libmatmul_fp32.so \
  matmul-demo/model/input.raw \
  $DEVICE_ROOT/cpu/

adb -s 127.0.0.1:40404 push \
  $QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run \
  $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnGpu.so \
  matmul-demo/build/fp32/model_libs/aarch64-android/libmatmul_fp32.so \
  matmul-demo/model/input.raw \
  $DEVICE_ROOT/gpu/

adb -s 127.0.0.1:40404 push \
  $QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run \
  $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so \
  $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so \
  matmul-demo/build/fp32/model_libs/aarch64-android/libmatmul_fp32.so \
  matmul-demo/model/input.raw \
  $DEVICE_ROOT/npu/

# 准备输入列表
adb -s 127.0.0.1:40404 shell "cd $DEVICE_ROOT/cpu && printf '%s\n' input.raw > input_list.txt"
adb -s 127.0.0.1:40404 shell "cd $DEVICE_ROOT/gpu && printf '%s\n' input.raw > input_list.txt"
adb -s 127.0.0.1:40404 shell "cd $DEVICE_ROOT/npu && printf '%s\n' input.raw > input_list.txt"
```

然后分别执行三组测试。

`CPU`：

```bash
adb -s 127.0.0.1:40404 shell "
  cd $DEVICE_ROOT/cpu && \
  export LD_LIBRARY_PATH=$DEVICE_ROOT/cpu && \
  ./qnn-net-run \
    --model libmatmul_fp32.so \
    --backend libQnnCpu.so \
    --input_list input_list.txt \
    --output_dir output
"
```

`GPU`：

```bash
adb -s 127.0.0.1:40404 shell "
  cd $DEVICE_ROOT/gpu && \
  export LD_LIBRARY_PATH=$DEVICE_ROOT/gpu && \
  ./qnn-net-run \
    --model libmatmul_fp32.so \
    --backend libQnnGpu.so \
    --input_list input_list.txt \
    --output_dir output
"
```

`NPU`：

```bash
adb -s 127.0.0.1:40404 shell "
  cd $DEVICE_ROOT/npu && \
  export LD_LIBRARY_PATH=$DEVICE_ROOT/npu && \
  export ADSP_LIBRARY_PATH=$DEVICE_ROOT/npu && \
  ./qnn-net-run \
    --model libmatmul_fp32.so \
    --backend libQnnHtp.so \
    --input_list input_list.txt \
    --output_dir output
"
```

### 2.5. 验证输出

分别回拉 `CPU/GPU/NPU` 三组输出：

```bash
mkdir -p matmul-demo/device/{cpu,gpu,npu}
adb -s 127.0.0.1:40404 pull $DEVICE_ROOT/cpu/output matmul-demo/device/cpu
adb -s 127.0.0.1:40404 pull $DEVICE_ROOT/gpu/output matmul-demo/device/gpu
adb -s 127.0.0.1:40404 pull $DEVICE_ROOT/npu/output matmul-demo/device/npu
```

然后做一个简单对比：

```python
from pathlib import Path
import numpy as np

ref = np.load("matmul-demo/model/reference.npy").reshape(-1)

for name, atol in [("cpu", 1e-6), ("gpu", 1e-5), ("npu", 1e-4)]:
    out_path = next(Path(f"matmul-demo/device/{name}/output").rglob("*.raw"))
    out = np.fromfile(out_path, dtype=np.float32).reshape(-1)
    max_abs_err = np.max(np.abs(out - ref))
    mean_abs_err = np.mean(np.abs(out - ref))
    is_close = np.allclose(out, ref, atol=atol, rtol=0.0)
    print(
        name,
        "pass=",
        is_close,
        "atol=",
        atol,
        "max_abs_err=",
        max_abs_err,
        "mean_abs_err=",
        mean_abs_err,
    )
```
测试结果如下，误差npu最大：
```text
cpu pass= False atol= 1e-06 max_abs_err= 4.7683716e-06 mean_abs_err= 1.2996607e-06
gpu pass= True atol= 1e-05 max_abs_err= 5.722046e-06 mean_abs_err= 1.6018748e-06
npu pass= False atol= 0.0001 max_abs_err= 0.009534836 mean_abs_err= 0.0027581714
```

**通过该工具链可以把torch模型导出onnx算子图，转换编译并使用qnn-net-run在端侧npu上使用，缺点是灵活度不够**