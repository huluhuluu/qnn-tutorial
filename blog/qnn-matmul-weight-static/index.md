---
title: "QNN MatMul 权重静态化与输入化"
date: 2026-04-27T10:00:00+08:00
lastmod: 2026-04-27T10:00:00+08:00
draft: false
description: "分析 MatMul 中把权重做成静态张量或输入张量时，对图体积、构图开销和 NPU 执行时间的影响。"
slug: "qnn-matmul-weight-static"
tags: ["qnn", "matmul"]
categories: ["qnn"]
comments: true
math: true
---

# QNN MatMul 权重静态化与输入化

`MatMul` 里，把矩阵权重做成 `QNN_TENSOR_TYPE_STATIC`，和把权重做成 `QNN_TENSOR_TYPE_APP_WRITE`的区别：

- 纯图拓扑几乎不变，变的是图里携带的数据和运行时绑定方式。
- 如果看上下文缓存、模型库或打包后的 `so` 体积，静态权重通常更大。
- 如果看 NPU 上多次重复推理的执行时间，静态权重通常更容易快。

## 1. 两种写法

在 [QNN 环境](/p/qnn-setup/) 里，`MatMul` 可以写成两种方式：

```cpp
// 静态权重
Qnn_Tensor_t weightTensor = makeTensor("weight",
                                      QNN_TENSOR_TYPE_STATIC,
                                      QNN_DATATYPE_FLOAT_32,
                                      weightDims.data(),
                                      kTensorRank,
                                      weightData.data(),
                                      static_cast<uint32_t>(weightData.size() * sizeof(float)));

// 权重输入
Qnn_Tensor_t weightTensor = makeTensor("weight",
                                      QNN_TENSOR_TYPE_APP_WRITE,
                                      QNN_DATATYPE_FLOAT_32,
                                      weightDims.data(),
                                      kTensorRank,
                                      nullptr,
                                      0);
```
两种方式的异同主要体现在：

- `MatMul` 的节点数和边数不变。
- 生成出来的模型库、上下文二进制、或者打包后的 `so` 体积，静态权重会明显更大。
- 运行时输入列表里，输入化权重会多一份外部输入。

**计算时间：** 静态权重可以让后端提前看到常量值，很多后端会在 `finalize` 或编译阶段做常量相关处理，比如预打包、布局转换、融合和缓存。

- `build/finalize` 可能更慢
- 上下文缓存可能更大
- 这部分成本通常只付一次


**图大小：** 输入权重每次都要作为 graph input 绑定和传输，静态权重则直接复用后端已经接收和准备好的常量。所以在重复推理场景里，静态权重速度更占优：

- 少一次大块权重输入
- 少一次权重 buffer 绑定/校验
- 少一次 host 到 NPU 侧的数据准备机会
- 更容易命中后端内部的权重布局缓存

如果权重特别大，这个差距会更明显。

## 2. 在线构图bench

源码放在：

```text
blog/qnn-matmul-weight-static/weight_static_benchmark.cpp
```

它在同一个进程里构建两张 `MatMul` 图，分别是：

- `static`：`weight` 是 `QNN_TENSOR_TYPE_STATIC`，构图时权重数据直接挂在张量里。
- `app_write`：`weight` 是 `QNN_TENSOR_TYPE_APP_WRITE`，构图时权重张量不带数据；每轮执行时把权重数据作为输入传进去。

两张图使用同样的 `M/K/N`、同样的输入数据、同样的权重数据和同一个 HTP backend。输出表里重点看这些列：

- `precision`：本轮测试的张量精度，支持 `fp32/fp16/int8/int16`。
- `ctx_bin(MB)`：`QnnContext_getBinarySize()` 返回的上下文缓存大小，按 `1024 * 1024` 转成 MB 显示。
- `static(MB)`：注册到图里的静态权重 payload 大小，按 MB 显示。
- `runtime(MB)`：每次执行需要从应用侧传入的输入大小，按 MB 显示。
- `final(ms)`：`QnnGraph_finalize()` 耗时。
- `first(ms)`：第一次 `QnnGraph_execute()` 耗时。
- `exec(ms)`：多轮 `QnnGraph_execute()` 的均值和标准差。
- `precision_error_summary`：按精度汇总所有 `OK` case 的误差；`max_err` 是该精度下最大的最大误差，`avg_mean_err` 是该精度下 `mean_err` 的平均值。

每个 `precision + weight_mode` 会单独跑一行。benchmark 默认测试：

```text
fp32,fp16,int8,int16
```

如果只想跑某些精度，用：

```bash
--precisions fp16,int8,int16
```

### 2.1 编译

这个 benchmark 由仓库根目录的 `CMakeLists.txt` 统一管理, Android 交叉编译:

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28

cmake --build build --target qnn_weight_static_benchmark -j4
```

### 2.2 推送到设备

把 benchmark、HTP 后端库和 C++ runtime 推到设备：

```bash
export DEVICE_ROOT=/data/local/tmp/qnn_matmul_weight_static
adb -s 127.0.0.1:40404 shell "mkdir -p $DEVICE_ROOT"

adb -s 127.0.0.1:40404 push \
  build/qnn_weight_static_benchmark \
  $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so \
  $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so \
  $DEVICE_ROOT/
```

如果你的 SoC 不是 `v79`，需要把 `libQnnHtpV79Stub.so` 和 `lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so` 换成对应架构。


在 HTP/NPU 后端（`SM8750 / HTP v79` 上执行：）运行：

```bash
adb -s 127.0.0.1:40404 shell "
  cd /data/local/tmp/qnn_matmul_weight_static && \
  export LD_LIBRARY_PATH=\$PWD && \
  export ADSP_LIBRARY_PATH=\$PWD && \
  ./qnn_weight_static_benchmark \
    --backend ./libQnnHtp.so \
    --m 512 --k 2048 --n 2048 \
    --warmup 3 --iters 5 \
    --precisions fp32,fp16,int8,int16
"
# --m 512 --k 2048 --n 6144 \
# --m 512 --k 6144 --n 2048 \
# --m 512 --k 2048 --n 2048 \
# --m 1 --k 151936 --n 2048 \
```

输出结构如下。下面这份是 `SM8750 / HTP v79` 上的小矩阵 smoke test；这个 runtime 接受了 `fp32` 的 QNN API 张量配置，但这不等价于 HTP 内部在做原生 FP32 MatMul。

```text
QNN MatMul Weight Static Benchmark
backend_path : ./libQnnHtp.so
backend_kind : NPU
htp_soc_model: 69
htp_arch     : 79
shape        : A[512, 2048] x B[2048, 6144] -> C[512, 6144]
warmup       : 3
iterations   : 5

precision weight     status             valid       ctx(ms)     build(ms)     final(ms)     first(ms)          exec(ms)      free(ms)    ctx_bin(MB)     static(MB)    runtime(MB)       max_err      mean_err
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fp32      static     OK                  PASS        0.0292       85.0602      545.1886       10.1046    6.0671±0.8934       24.0706         4.3086        48.0000         4.0000        0.0026        0.0007
fp32      app_write  OK                  PASS        0.0377        0.4305      396.2505       52.3059   38.2210±2.6300       36.8942         0.2852         0.0000        52.0000        0.0026        0.0007
fp16      static     OK                  PASS        0.0377       14.0695      139.9085        7.3569    4.8455±0.0422       24.6860         4.3008        24.0000         2.0000        0.0026        0.0007
fp16      app_write  OK                  PASS        0.0391        0.3680     4126.0704       28.9443   23.9732±1.2695       52.0595         0.5664         0.0000        26.0000        0.0026        0.0007
int8      static     OK                  FAIL        0.0661        9.2544     1559.6377       16.3176   18.0157±5.1124       32.5441         8.6641        12.0000         1.0000        6.7834        2.2220
int8      app_write  OK                  FAIL        0.0447        0.3903      529.9657       35.4261   26.7496±4.7284       36.0866         0.3633         0.0000        13.0000        6.7834        2.2220
int16     static     OK                  PASS        0.0375       13.8047      482.1438       25.6598   22.8528±3.7121       28.8608         4.4336        24.0000         2.0000        0.0071        0.0027
int16     app_write  OK                  PASS        0.0357        0.4098     2056.0973       99.2368 122.3321±13.6945       63.8079         1.2305         0.0000        26.0000        0.0141        0.0054

context_binary_status
  fp32      static     ok
  fp32      app_write  ok
  fp16      static     ok
  fp16      app_write  ok
  int8      static     ok
  int8      app_write  ok
  int16     static     ok
  int16     app_write  ok

precision_error_summary
precision      cases       max_err      avg_mean_err
----------------------------------------------------
fp32              2        0.0026            0.0007
fp16              2        0.0026            0.0007
int8              2        6.7834            2.2220
int16             2        0.0141            0.0041

QNN MatMul Weight Static Benchmark
backend_path : ./libQnnHtp.so
backend_kind : NPU
htp_soc_model: 69
htp_arch     : 79
shape        : A[512, 6144] x B[6144, 2048] -> C[512, 2048]
warmup       : 3
iterations   : 5

precision weight     status             valid       ctx(ms)     build(ms)     final(ms)     first(ms)          exec(ms)      free(ms)    ctx_bin(MB)     static(MB)    runtime(MB)       max_err      mean_err
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fp32      static     OK                  PASS        0.0252       84.7228      715.4846       12.0498   11.9023±1.6982       31.6374        12.8633        48.0000        12.0000        0.0016        0.0005
fp32      app_write  OK                  PASS        0.0217        0.2437      196.9550       59.2926   39.9130±2.8257       20.9656         0.1562         0.0000        60.0000        0.0016        0.0005
fp16      static     OK                  PASS        0.0311       12.2270      196.3642       20.4973   10.8049±1.6617       26.2479        12.8047        24.0000         6.0000        0.0016        0.0005
fp16      app_write  OK                  PASS        0.0383        0.3783     2350.6353       45.8905   30.1110±6.5165       31.6772         0.2305         0.0000        30.0000        0.0016        0.0005
int8      static     OK                  PASS        0.0663        8.6619     1950.1152       32.4290   27.9270±8.7762       32.5944        12.2031        12.0000         3.0000        0.0351        0.0121
int8      app_write  OK                  PASS        0.0647        0.4667     8905.2090       39.7415   29.3041±6.1805       46.4648         0.5898         0.0000        15.0000        0.0351        0.0121
int16     static     OK                  PASS        0.0379       16.8233     1472.6296       45.7644   73.8026±4.9664       46.2043        13.2148        24.0000         6.0000        0.0071        0.0028
int16     app_write  OK                  PASS        0.0384        0.3794    16282.6409      247.3371 263.2424±21.9484      168.4643         7.4531         0.0000        30.0000        0.0140        0.0055

context_binary_status
  fp32      static     ok
  fp32      app_write  ok
  fp16      static     ok
  fp16      app_write  ok
  int8      static     ok
  int8      app_write  ok
  int16     static     ok
  int16     app_write  ok

precision_error_summary
precision      cases       max_err      avg_mean_err
----------------------------------------------------
fp32              2        0.0016            0.0005
fp16              2        0.0016            0.0005
int8              2        0.0351            0.0121
int16             2        0.0140            0.0042
```


## 3. 离线构图

前面的 benchmark 比较的是在线构图后的 `STATIC` 和 `APP_WRITE`。为了更接近部署场景，新加了一个独立程序：

```text
blog/qnn-matmul-weight-static/weight_switch_benchmark.cpp
```

它默认尝试三种运行模式，输出里的 `mode` 名称简化成：

- `app_write`：一张图，`weight` 是 `QNN_TENSOR_TYPE_APP_WRITE`；每轮执行前从磁盘读 input/weight 数据，再把当前权重作为第二个输入传给 `QnnGraph_execute()`。
- `static_load`：每个权重编译一张 `STATIC` 图，导出 context binary 并写到磁盘；每轮执行前临时读取一个 binary，再做 `contextCreateFromBinary + graphRetrieve + execute + free`。
- `update_static`(不支持暂时)：一张图，`weight` 是 `QNN_TENSOR_TYPE_UPDATEABLE_STATIC`；每轮执行前从磁盘读新权重，调用 `QnnTensor_updateGraphTensors()`，再 `QnnGraph_finalize()` 让更新生效，然后执行。

三种模式使用同样的 `M/K/N`、同样的输入数据、同样的权重数据和同一个 backend。程序会先为每个 `shape + precision` 生成公共 tensor/expected-output 文件，然后释放 fp32 参考数据和常驻权重 payload；各模式在计时路径里再按需从磁盘读取当前 input、weight、expected output 或 graph binary。这样可以避免测试时把所有权重和所有参考输出长期留在 CPU 内存里。

实际在 `SM8750 / HTP v79` 上测试时，`update_static` 的 `UPDATEABLE_STATIC + updateGraphTensors + finalize` 路径会被 HTP 拒绝，所以程序会把这一行标成 `SKIP`，并且不会因为这个已知不支持路径返回失败。输出表里重点看这些列：

- `precision`：本轮测试的张量精度，默认是 `fp16/int8/int16`。
- `prep(ms)`：模式自己的准备阶段耗时，包括构图、导出 binary 等一次性工作；公共 tensor/expected-output 文件生成在各 mode 计时前完成，不计入这一列，且不包含单独列出的 `final(ms)`。
- `ssd_d(ms)`：每轮从磁盘读取 input/weight 数据的时间。
- `ssd_g(ms)`：每轮从磁盘读取 graph binary 的时间；只有 `static_load` 会有这一列。
- `htp_w(ms)`：把图或权重真正写入 backend 的时间；`static_load` 里是 `contextCreateFromBinary + graphRetrieve`，`update_static` 里是 `updateGraphTensors + finalize`。
- `final(ms)`：构图后的首次 `QnnGraph_finalize()` 总耗时。
- `first(ms)`：第一轮完整路径耗时。
- `exec(ms)`：纯 `QnnGraph_execute()` 的均值和标准差。
- `binary(B)`：导出的 context binary 总大小。
- `static(B)`：静态权重占用的 payload 字节数。
- `runtime(B)`：每轮执行时应用侧需要准备的输入字节数。
- `--m 512 --k 2048 --n 6144`: 测试shape



### 3.1 编译

这个 benchmark 也由仓库根目录的 `CMakeLists.txt` 统一管理，Android 交叉编译：

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28

cmake --build build --target qnn_weight_switch_benchmark -j4
```

### 3.2 推送到设备

把 benchmark、HTP 后端库和 C++ runtime 推到设备：

```bash
export DEVICE_ROOT=/data/local/tmp/qnn_static_binary_switch
adb -s 127.0.0.1:40404 shell "mkdir -p $DEVICE_ROOT"

adb -s 127.0.0.1:40404 push \
  build/qnn_weight_switch_benchmark \
  $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so \
  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so \
  $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so \
  $DEVICE_ROOT/
```

如果你的 SoC 不是 `v79`，需要把 `libQnnHtpV79Stub.so` 和 `lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so` 换成对应架构。

在 HTP/NPU 后端运行：

```bash
adb -s 127.0.0.1:40404 shell "
  cd /data/local/tmp/qnn_static_binary_switch && \
  export LD_LIBRARY_PATH=\$PWD && \
  export ADSP_LIBRARY_PATH=\$PWD && \
  ./qnn_weight_switch_benchmark \
    --backend ./libQnnHtp.so \
    --precisions fp16,int8,int16
"
```

这个程序默认只把结果打印到 stdout，不会额外写日志文件。只有显式传 `--log <path>` 时才会把同样的结果落盘。

程序运行过程中会为了测量切换开销，临时写出 input / weight / expected-output / static graph binary 等中间文件；正常退出或异常退出时都会自动清理这些本地临时文件，不会在当前目录长期残留。
