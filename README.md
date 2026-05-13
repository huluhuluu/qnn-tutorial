# QNN_TUTORIAL

围绕 Qualcomm QNN / QAIRT 的中文教程仓库。

## 目录

| 文章 | 说明 | 状态 |
|------|------|------|
| [QNN 介绍](blog/qnn-intro/index.md) | 梳理`QNN`各部分关系与调用顺序 | ✅ 完成 |
| [QNN 环境](blog/qnn-setup/index.md) | 以 `MatMul` 为主线，串联`QNN`配置与运行 | ✅ 完成 |
| [QNN 环境 py版](blog/qnn-setup-py/index.md) | 以 `MatMul` 为主线，串联`QNN python`工具链配置与运行 | ✅ 完成 |
| [QNN 文档](blog/qnn-doc/index.md) | 常用文档阅读 | ✅ 完成 |
| [QNN MatMul 权重静态化与输入化](blog/qnn-matmul-weight-static/index.md) | 分析权重静态化对图体积和执行时间的影响 | ✅ 完成 |
| [QNN HTP OpPackage Softmax 源码分析](blog/qnn-oppackage-softmax/index.md) | 拆解 `ExampleOpPackageSoftmax.cpp` 的注册、优化和实现 | ✅ 完成 |

## 编译

根目录统一使用 `build/`：

```bash
cmake -S . -B build  \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build -j4
```

切换 host 和 Android toolchain 时，先清空 `build/` 再重新配置。


## 参考资源

- [Qualcomm AI Engine Direct 官方文档](https://docs.qualcomm.com/nav/home/api_overview.html?product=1601111740010412)
- 本地 SDK 文档：`/root/qairt/2.40.0.251030/docs/QNN` (与安装路径有关)
