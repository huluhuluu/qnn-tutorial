---
title: "QNN HTP OpPackage Softmax 源码分析"
date: 2026-04-27T10:10:00+08:00
lastmod: 2026-04-27T10:10:00+08:00
draft: false
description: "学习官方示例 Softmax Op的实现。"
slug: "qnn-oppackage-softmax"
tags: ["qnn", "htp", "oppackage"]
categories: ["qnn"]
comments: true
math: true
---

# QNN HTP OpPackage Softmax 源码分析

这篇为对照源码`/qairt/2.40.0.251030/examples/QNN/OpPackage/HTP/ExampleOpPackageSoftmax.cpp` 学习`QNN`如何自定义`Softmax`算子。它涵盖了：

- `Softmax` 在 HTP OpPackage 里的注册方式
- `Softmax` 参数顺序
- `Softmax -> Softmax_fp` 的图改写规则
- 普通 reference 版本
- FP16 + HVX 近似版本

## 1. 源码分析
### 1.1 头文件

源码开头包含了几类 HTP core 头文件：

```cpp
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
```

可以按职责理解：

| 头文件 | 作用 |
|---|---|
| `constraints.h` | 给优化规则写匹配条件，例如 dtype、scalar、常量值 |
| `op_package_feature_support.h` | OpPackage 特性支持相关定义 |
| `op_register_ext.h` | 注册自定义 op 的宏 |
| `optimize.h` | 图优化和 pattern rewrite 的宏 |
| `simple_reg.h` | 简化注册接口 |

这些宏不是普通 C++ 函数调用风格，而是 HTP backend 给 OpPackage 暴露的一套 DSL。

### 1.2 算子定义

#### 1.2.1 算子定义

源码第 27 行：

```cpp
BEGIN_PKG_OP_DEFINITION(PKG_Softmax);
```

这行给当前源文件里的 op 定义起内部名字 `PKG_Softmax`。它后面要和 `ExampleOpPackageInterface.cpp` 里的：

```cpp
DECLARE_PKG_OPS_OPTS_LIST(PKG_Softmax)
```

对应起来。即`Softmax.cpp` 里声明了包内容，`Interface.cpp` 把这个包内容挂到最终 OpPackage 接口里。

源码第 29 行：

```cpp
DEF_PACKAGE_PARAM_ORDER("Softmax", "beta", false, nullptr, "axis", false, nullptr)
```

定义了 `Softmax` 的参数顺序。它说明 `Softmax` 最多关心两个参数：

- `beta`
- `axis`

`Interface.cpp` 的 `validateOpConfig()` 对应这个约定：`Softmax` 允许 `numOfParams <= 2`。

#### 1.2.2 函数声明

源码第 32 到 36 行先声明两个实现：

```cpp
template <typename T_Ttype>
int softmaxWithbetaWrapper(T_Ttype &out, const T_Ttype &in, const Tensor &beta);

template <typename OutTtype, typename InTtype>
int softmax_fp_impl(OutTtype &out, const InTtype &in, const Tensor &beta);
```

它们代表两条路径：

- `softmaxWithbetaWrapper()`：普通模板路径，内部调用 reference 写法。
- `softmax_fp_impl()`：FP16 快路径，内部调用 `softmax_hf_approx()`。

#### 1.2.3 注册 Softmax 算子

源码第 43 到 44 行：

```cpp
DEF_PACKAGE_OP(softmaxWithbetaWrapper<Tensor>, "Softmax")
DEF_PACKAGE_OP((softmaxWithbetaWrapper<QuantUint8Tensor>), "Softmax")
```

把函数模板实例注册成名为 `Softmax` 的 op 实现。

源码第 56 到 59 行：

```cpp
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
                                  "Softmax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
```

也注册 `Softmax`，但多了两个信息：

- `SNAIL`：成本模型里比较慢的实现。
- `Flags::RESOURCE_HVX`：这个实现会占用 HVX 资源。

#### 1.2.4 Tensor 属性

源码第 61 行：

```cpp
DEF_TENSOR_PROPERTIES(Op("Softmax", "in", "Beta"), Flat("*"), MainMemory("*"))
```

定义 `Softmax` 的 tensor 属性：

- 匹配 `Softmax(in, Beta)` 这种形式。
- `Flat("*")` 表示相关 tensor 使用 flat 形态。
- `MainMemory("*")` 表示相关 tensor 在 main memory。

这里的 `*` 是 pattern 里的通配符，HTP graph optimizer 会拿这些属性判断某个 op 实现能不能接住当前图里的 tensor。

#### 1.2.5 注册 Softmax_fp 算子

源码第 64 到 72 行注册了两个 `Softmax_fp`：

```cpp
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmax_fp_impl<PlainFloat16Tensor, PlainFloat16Tensor>),
                                  "Softmax_fp",
                                  FAST,
                                  Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmax_fp_impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>),
                                  "Softmax_fp",
                                  FAST,
                                  Flags::RESOURCE_HVX)
```

这两条都标成 `FAST`，说明作者希望真正的 float 计算最后走 `Softmax_fp`。

区别在 tensor 类型：

- `PlainFloat16Tensor`：普通 FP16 tensor。
- `PlainFloat16Tensor_TCM`：放在 TCM 里的 FP16 tensor。

TCM(Tightly Coupled Memory) 是 HTP/DSP 侧更靠近计算单元的片上存储。这里注册 TCM 版本，是为了让 optimizer 在合适的内存布局下能选更快路径。

### 1.3 算子优化
#### 1.3.1 float32 -> fp16

源码第 77 到 93 行是一条关键优化规则：

```cpp
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    GRAPH_CLEANUP,
    relaxed_precision_flag,
    Op("Softmax", "In", "Beta", "axis"),
    AND(EQ(DTYPE_OF("In"), DType::Float32), EQ(DTYPE_OF("*"), DType::Float32)),
    ... Op("Softmax_fp", WITH_SIZE("In", Op(FROM_DEFAULT_PACKAGE("Cast"), "In")), "Beta") ...)
```

它做的事情是：

```text
Softmax(float32 In, Beta, axis)
-> Cast(In -> float16)
-> Softmax_fp(float16, Beta)
-> Cast(output -> float32)
```

触发条件是：

- 输入 `In` 是 `Float32`
- 输出也是 `Float32`
- 启用了 `relaxed_precision_flag`

所以这里不是无条件把 float32 改成 fp16，而是在精度允许时，把主计算切到 HTP 更友好的 FP16 实现。

#### 1.3.2 折叠前置乘法

源码第 95 到 110 行：

```cpp
Op("Softmax_fp",
   Op(FROM_DEFAULT_PACKAGE("QNN_OP_ELEMENT_WISE_MULTIPLY"),
      "In",
      Op(FROM_DEFAULT_PACKAGE("QNN_OP_CAST"), OPCONST("MaybeScalarVal"))),
   "Beta")
```

它匹配这种图：

```text
Softmax_fp(Multiply(In, Cast(scalar)), Beta)
```

如果 `MaybeScalarVal` 是正的 float32 scalar，就改写成：

```text
Softmax_fp(In, Beta * scalar)
```

写成公式形式就是：

$$
\exp(\beta \cdot x')
$$

其输入前面有一个正标量乘法：

$$
x' = s \cdot x
$$

那就可以变成：

$$
\exp(\beta \cdot s \cdot x)
$$

也就是把 `s` 合进 `beta`。

#### 1.3.3 默认参数规整

源码第 112 到 117 行有两条更简单的 rewrite：

```cpp
DEF_PACKAGE_OPTIMIZATION(QNN,
                         Op("Softmax", "In"),
                         OK,
                         Op("Softmax", "In", gen_ConstScalar_f32(1.0f)))

DEF_PACKAGE_OPTIMIZATION(QNN, Op("Softmax", "In", "Beta", "axis"), OK, Op("Softmax", "In", "Beta"))
```

第一条把：

```text
Softmax(In) -> Softmax(In, 1.0f)
```

第二条把：

```text
Softmax(In, Beta, axis) -> Softmax(In, Beta)
```

这个例子里的实现实际按最后一维做 softmax，没有使用 `axis` 参数。

#### 1.3.4 `softmax_hf_approx()` 总览

源码第 119 行开始是 FP16 快路径核心：

```cpp
void softmax_hf_approx(Float16 *pout, const Float16 *pin, float scale, int length)
```

输入含义：

- `pin`：一段 FP16 输入。
- `pout`：一段 FP16 输出。
- `scale`：外部传入的缩放系数，后面由 `beta` 和 `interface_scale()` 合成。
- `length`：最后一维长度，也就是 softmax 的归一化长度。

整体流程对应普通 softmax：

```text
max = reduce_max(x)
y = approx_exp(scale * (x - max))
sum = reduce_sum(y)
out = y / sum
```

只是这里全部手写成 HVX vector intrinsic。

## 1.4 伪代码

算子真正执行的核心计算是：

```text
softmax(x) = exp(scale * (x - max(x))) / sum(exp(scale * (x - max(x))))
```

这几段伪代码的调用关系是：

```text
softmax_fp_impl()
  └── 遍历输入 tensor 的 [B, H, W, D] 外层切片[B, H, W]
      └── 对每个连续的最后一维 [D] 切片调用 softmax_hf_approx()
          ├── reduce max 计算最大值
          ├── approx_exp2() 近似计算 exp 的主体
          ├── reduce sum 求和
          └── 原地归一化输出
```

其中 `approx_exp2()` 不是源码里的真实函数名，而是为了说明第 184 到 266 行的指数近似逻辑抽出来的概念函数。源码里这些逻辑直接内联写在 `softmax_hf_approx()` 的主循环中。

### 1.4.1 `softmax_fp_impl()` 伪代码

`softmax_fp_impl()` 本身不直接写 HVX 算法。它负责遍历输入的外层维度，把每个最后一维切片交给 `softmax_hf_approx()`：

```text
function softmax_fp_impl(out, in, beta_tensor):
    # 输出 tensor 的 shape 和输入一致。
    out.shape = in.shape

    # 示例实现按 4D tensor 处理：[B, H, W, D]。
    # D 是最后一维，也是 softmax 的归一化维度。
    B, H, W, D = in.shape

    # beta 是 softmax 指数缩放参数。
    # interface_scale 是输入 tensor 接口层的 scale，需要合进总 scale。
    scale = in.interface_scale() * beta_tensor[0]

    # 每个 [b, h, w, :] 是一段连续的 D 维 fp16 buffer。
    for b in 0 .. B-1:
      for h in 0 .. H-1:
        for w in 0 .. W-1:
            # 当前最后一维切片的输入/输出首地址。
            pin  = &in[b, h, w, 0]
            pout = &out[b, h, w, 0]

            # 对这个 D 维切片做 softmax。
            softmax_hf_approx(
                pout,
                pin,
                scale,
                length = D
            )
```

也就是说，这个示例实现里的归一化维度是最后一维 `D`。

### 1.4.2 `softmax_hf_approx()` 总体伪代码

`softmax_hf_approx()` 是真正的 FP16 + HVX 快路径。忽略向量 intrinsic 的细节，它做的是稳定版 softmax：

```text
out[i] = exp(scale * (in[i] - max(in))) / sum_j exp(scale * (in[j] - max(in)))
```

伪代码如下：

```text
function softmax_hf_approx(pout, pin, scale, length):
    # 换底：后面用 2^x 近似来实现 e^x。
    log2_scale = scale / ln(2)

    # 拟合 2^x 的三阶多项式系数，不是标准 Taylor 系数。
    c0 = 1.0
    c1 = 0.692850309695840
    c2 = 0.237504551482093
    c3 = 0.046751431261525

    # pass 1: 找 max，避免 exp 溢出。
    # 源码中每次读 64 个 fp16，先得到 vector 形式的部分最大值。
    xmax_vec = fp16_negative_infinity
    for block in pin split by 64 fp16 elements:
        x = load_fp16_vector(block)
        x = mask_tail_if_needed(x)
        xmax_vec = vector_max_fp16(xmax_vec, x)

    # 把 vector lane 内的最大值继续横向归约成一个标量 max。
    xmax = horizontal_max_fp16(xmax_vec)

    # pass 2: 计算未归一化 exp，并累加 sum。
    # pout 先被当成临时 buffer 保存 exp 近似值。
    sum_vec = 0
    for block in pin split by 64 fp16 elements:
        x = load_fp16_vector(block)
        x = mask_tail_if_needed(x)

        # 稳定版 softmax 的核心：先减全局 max。
        xd = x - xmax

        # fp16 扩展成两组 qf32，方便做指数近似。
        lo_qf32, hi_qf32 = widen_fp16_to_two_qf32_vectors(xd)

        # 进入 base-2 指数域。
        z_lo = lo_qf32 * log2_scale
        z_hi = hi_qf32 * log2_scale

        # 源码第 184 到 266 行内联做了这个近似。
        exp_lo = approx_exp2(z_lo)
        exp_hi = approx_exp2(z_hi)

        # 暂存未归一化 exp 到 pout。
        y_fp16 = narrow_qf32_pair_to_fp16(exp_lo, exp_hi)
        store_fp16(pout_block, y_fp16)

        # 同时累加 softmax 分母。
        sum_vec += exp_lo
        sum_vec += exp_hi

    # 把 vector 形式的部分和横向归约成标量 sum。
    sum = horizontal_sum_qf32(sum_vec)
    recip = 1.0 / sum

    # pass 3: 原地归一化。
    # pout 当前保存的是 exp 近似值，乘以 1/sum 后才是最终 softmax。
    for block in pout split by 64 fp16 elements:
        y = load_fp16_vector(block)
        lo_qf32, hi_qf32 = widen_fp16_to_two_qf32_vectors(y)

        out_lo = lo_qf32 * recip
        out_hi = hi_qf32 * recip

        out_fp16 = narrow_qf32_pair_to_fp16(out_lo, out_hi)
        store_fp16(pout_block, out_fp16)
```

### 1.4.3 `approx_exp2()` 伪代码

源码里没有单独的 `approx_exp2()` 函数，但第 184 到 266 行实际在做类似事情。先把自然指数换底：

$$
e^x = 2^{x / \ln 2}
$$

所以 kernel 里要近似的是 `2^z`。抽象成伪代码：

```text
function approx_exp2(z):
    # z 是 base-2 指数域的输入。
    # 源码先把 qf32 转成便于按 float32 bit 位拆分的形式。
    z_as_float = convert_qf32_to_float_bits(z)

    # 拆出 float32 exponent bits。
    exponent_bits = get_float_exponent_bits(z_as_float)

    # 限制 exponent，让多项式输入落在较小范围。
    exponent_limit = min(exponent_bits, 126 << 23)
    exponent_delta = exponent_bits - exponent_limit

    # 用受限 exponent 重组规格化输入。
    z_norm = replace_exponent(z_as_float, exponent_limit)

    # Horner 形式计算三阶多项式：
    # p = c0 + c1*z_norm + c2*z_norm^2 + c3*z_norm^3
    p = c3 * z_norm
    p = p + c2
    p = p * z_norm
    p = p + c1
    p = p * z_norm
    p = p + c0

    # repeated squaring：先构造多个幂次候选。
    p2  = p   * p
    p4  = p2  * p2
    p8  = p4  * p4
    p16 = p8  * p8
    p32 = p16 * p16
    p64 = p32 * p32

    # 根据 exponent_delta 选择对应幂次，恢复更大范围的指数效果。
    if exponent_delta == 1 << 23:
        return p2
    if exponent_delta == 2 << 23:
        return p4
    if exponent_delta == 3 << 23:
        return p8
    if exponent_delta == 4 << 23:
        return p16
    if exponent_delta == 5 << 23:
        return p32
    if exponent_delta > 5 << 23:
        return p64

    return p
```

### 1.5 算子实现
#### 1.5.1 exp 近似

源码第 124 到 130 行：

```cpp
scale /= float(log(2.0));
c0.f = 1.f;
c1.f = 0.692850309695840;
c2.f = 0.237504551482093;
c3.f = 0.046751431261525;
```

这里把自然指数换到底数 2 的指数域：

$$
e^x = 2^{x / \ln 2}
$$

然后，构造类似 `2^x` 的多项式近似。如下，`c0/c1/c2/c3` 是三阶多项式系数：

$$
c0 + c1 \cdot x + c2 \cdot x^2 + c3 \cdot x^3
$$

实际上，这里会先给$x$做上下范围的限制（$x = 2^k \times r$），然后用多项式近似小范围值$2 ^ r$，最后由于$2^x = 2^{2^k \cdot r} = (2^r)^{2^k}$，所以通过 repeated squaring 的方式构造 $p^2$、$p^4$、$p^8$... 来恢复更大范围的指数效果（就是前面的${2^k}$的指数部分）。

#### 1.5.2 找最大值

源码第 132 到 151 行完成 reduce max。

第 133 行：

```cpp
HVX_Vector xmax = Q6_Vh_vsplat_R(0xFC00);
```

`0xFC00` 是 half precision 的负无穷。`Q6_Vh_vsplat_R()` 把这个 16-bit 值广播到整个 HVX vector。

第 136 到 140 行每次读 64 个 FP16，寻找当前切片的最大值：

```cpp
for (int d = length; d > 63; d -= 64) {
    HVX_Vector xinval = vmemu(iptr);
    iptr++;
    xmax = Q6_Vhf_vmax_VhfVhf(xmax, xinval);
}
```

这里 `Vhf` 可以读成 vector half float。`Q6_Vhf_vmax_VhfVhf` 就是对两个 half-float vector 做逐元素 max。

第 141 到 145 行处理最后不足 64 个元素的情况：
```cpp
if ((length & 63) != 0) {
    HVX_Vector xinval         = vmemu(iptr);
    HVX_VectorPred qfinalmask = Q6_Q_vsetq2_R(length * 2);

    // mask 外的位置填成当前 xmax，避免尾部无效数据影响 max。
    xinval = Q6_V_vmux_QVV(qfinalmask, xinval, xmax);
    xmax   = Q6_Vhf_vmax_VhfVhf(xmax, xinval);
}
```

第 147 到 151 行把 vector 里的多个 lane 继续横向归约成一个最大值。
```cpp
// 横向归约：把 vector 内多个 lane 的最大值继续合并成一个 xmax。
int nshift = 2;
for (int i = 0; i < 6; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(xmax, xmax, nshift);
    xmax = Q6_Vhf_vmax_VhfVhf(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
}
```

#### 1.5.3. 计算未归一化 exp

源码第 154 到 285 行是 `softmax_hf_approx()` 的主计算部分。它做的是：

```text
tmp_i = approx_exp(scale * (x_i - max))
sum   = sum_i tmp_i
```

这里的 `tmp_i` 会先暂存到 `pout`，下一节再除以 `sum`。

进入主循环前，源码先准备常量、系数、指针和累加器：

```cpp
// zero vector，后面用于类型转换、加法归一化和清尾部 lane。
HVX_Vector vzero = Q6_V_vzero();

// fp16 的 1.0，0x3c00 是 half precision 1.0 的 bit pattern。
HVX_Vector voneh = Q6_Vh_vsplat_R(0x3c00);

// 把三阶多项式系数广播到 HVX vector。
HVX_Vector f0 = Q6_V_vsplat_R(c0.i);
HVX_Vector f1 = Q6_V_vsplat_R(c1.i);
HVX_Vector f2 = Q6_V_vsplat_R(c2.i);
HVX_Vector f3 = Q6_V_vsplat_R(c3.i);

// 这些 mask/常量用于拆 float32 的指数位和尾数部分。
HVX_Vector c7f800000 = Q6_V_vsplat_R(0x7f800000);
HVX_Vector c807fffff = Q6_V_vsplat_R(0x807fffff);

// vbeta 是已经除以 ln(2) 后的 scale：
// exp(scale_orig * x) = 2^((scale_orig / ln2) * x)
HVX_Vector vbeta = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(scaleu.i), vzero);

// 126 << 23 对应 float32 exponent field 的一个上限基准。
// 后面会用它限制/拆分指数部分，避免近似计算范围过大。
HVX_Vector c126 = Q6_V_vsplat_R(126 << 23);

// c1w, c2w, ... c5w 是指数位差值的比较常量。
// 后面根据 xexp 的大小选择 p、p^2、p^4、...、p^64。
HVX_Vector c1w = Q6_V_vsplat_R(1 << 23);
HVX_Vector c2w = Q6_Vw_vadd_VwVw(c1w, c1w);
HVX_Vector c3w = Q6_Vw_vadd_VwVw(c2w, c1w);
HVX_Vector c4w = Q6_Vw_vadd_VwVw(c3w, c1w);
HVX_Vector c5w = Q6_Vw_vadd_VwVw(c4w, c1w);

// vsumf 累加当前 softmax 切片所有未归一化 exp 的和。
HVX_Vector vsumf = Q6_V_vzero();

// optr 指向输出 buffer。此时输出 buffer 被用作 tmp buffer，
// 用来暂存未归一化的 exp 近似值。
HVX_Vector *optr = (HVX_Vector *)pout;
iptr = (HVX_Vector *)pin;
```

主循环每轮处理 64 个 FP16。第一步是读取输入并减最大值：

```cpp
for (int d = length; d > 0; d -= 64) {
  HVX_Vector x, x0, x1, p0, p1;
  HVX_VectorPair xdiff, p10;
  HVX_VectorPred q0, q1;

  // 读取当前 64 个 fp16 输入。
  x = vmemu(iptr);
  iptr++;

  // 稳定版 softmax 的关键：
  // xd = x - max(x)
  HVX_Vector xd = Q6_Vqf16_vsub_VhfVhf(x, xmax);

  // 把 fp16 差值扩展成两组 qf32 数据。
  // 返回 HVX_VectorPair，后面分别处理 lo/hi 两半。
  xdiff = Q6_Wqf32_vmpy_Vqf16Vhf(xd, voneh);
```

接着乘以 `vbeta`，进入 `2^x` 的近似域：

```cpp
  // x0/x1 = (x - max) * scale / ln(2)
  x0 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(xdiff), vbeta);
  x0 = Q6_Vsf_equals_Vqf32(x0);

  x1 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(xdiff), vbeta);
  x1 = Q6_Vsf_equals_Vqf32(x1);
```

后面这组 bit 操作是在拆 float32 的指数位和规格化尾数。直观理解是：把 `2^x` 的计算拆成“较小范围的多项式近似”和“指数缩放/幂次恢复”两部分。

```cpp
  // 取出 float32 的 exponent bits。
  HVX_Vector x0exp = Q6_V_vand_VV(x0, c7f800000);
  HVX_Vector x1exp = Q6_V_vand_VV(x1, c7f800000);

  // 把 exponent 限制到 c126 以内，构造后续多项式的输入范围。
  HVX_Vector x0explimit = Q6_Vw_vmin_VwVw(x0exp, c126);
  HVX_Vector x1explimit = Q6_Vw_vmin_VwVw(x1exp, c126);

  // xexp 保存剩余的指数位差值，用于后面选择 p^2/p^4/...。
  x0exp = Q6_Vw_vsub_VwVw(x0exp, x0explimit);
  x1exp = Q6_Vw_vsub_VwVw(x1exp, x1explimit);

  // xnorm 是规格化后的多项式输入。
  HVX_Vector x0norm = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x0), x0explimit);
  HVX_Vector x1norm = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x1), x1explimit);
```

然后用 Horner 形式计算三阶多项式：

```cpp
  // p0 = ((c3 * x0norm + c2) * x0norm + c1) * x0norm + c0
  // 即 p0 = c0 + c1*x + c2*x^2 + c3*x^3
  p0 = Q6_Vqf32_vmpy_VsfVsf(x0norm, f3);
  p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f2);
  x0norm = Q6_Vqf32_vadd_VsfVsf(x0norm, vzero);
  p0 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
  p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f1);
  p0 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
  p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f0);

  // p1 对应 vector 的另一半，计算方式相同。
  p1 = Q6_Vqf32_vmpy_VsfVsf(x1norm, f3);
  p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f2);
  x1norm = Q6_Vqf32_vadd_VsfVsf(x1norm, vzero);
  p1 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
  p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f1);
  p1 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
  p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f0);
```

接下来反复平方，构造 `p^2` 到 `p^64`。这一步服务于前面拆出来的 exponent 差值：指数位差值越大，就选择更高次的平方结果。

```cpp
  // 构造 p0^2, p0^4, p0^8, p0^16, p0^32, p0^64。
  HVX_Vector p0_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, p0);
  p0_2             = Q6_Vqf32_vadd_Vqf32Vsf(p0_2, vzero);
  HVX_Vector p0_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_2, p0_2);
  p0_4             = Q6_Vqf32_vadd_Vqf32Vsf(p0_4, vzero);
  HVX_Vector p0_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_4, p0_4);
  p0_8             = Q6_Vqf32_vadd_Vqf32Vsf(p0_8, vzero);
  HVX_Vector p0_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_8, p0_8);
  p0_16            = Q6_Vqf32_vadd_Vqf32Vsf(p0_16, vzero);
  HVX_Vector p0_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_16, p0_16);
  p0_32            = Q6_Vqf32_vadd_Vqf32Vsf(p0_32, vzero);
  HVX_Vector p0_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_32, p0_32);

  // p1^2 ... p1^64 同理。
  HVX_Vector p1_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, p1);
  p1_2             = Q6_Vqf32_vadd_Vqf32Vsf(p1_2, vzero);
  HVX_Vector p1_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_2, p1_2);
  p1_4             = Q6_Vqf32_vadd_Vqf32Vsf(p1_4, vzero);
  HVX_Vector p1_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_4, p1_4);
  p1_8             = Q6_Vqf32_vadd_Vqf32Vsf(p1_8, vzero);
  HVX_Vector p1_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_8, p1_8);
  p1_16            = Q6_Vqf32_vadd_Vqf32Vsf(p1_16, vzero);
  HVX_Vector p1_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_16, p1_16);
  p1_32            = Q6_Vqf32_vadd_Vqf32Vsf(p1_32, vzero);
  HVX_Vector p1_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_32, p1_32);
```

根据 `x0exp/x1exp` 的值选择对应幂次：

```cpp
  // 如果 xexp == 1 << 23，选择 p^2。
  q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c1w);
  q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c1w);
  p0 = Q6_V_vmux_QVV(q0, p0_2, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_2, p1);

  // 如果 xexp == 2 << 23，选择 p^4。
  q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c2w);
  q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c2w);
  p0 = Q6_V_vmux_QVV(q0, p0_4, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_4, p1);

  // 后面依次选择 p^8、p^16、p^32。
  q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c3w);
  q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c3w);
  p0 = Q6_V_vmux_QVV(q0, p0_8, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_8, p1);

  q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c4w);
  q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c4w);
  p0 = Q6_V_vmux_QVV(q0, p0_16, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_16, p1);

  q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c5w);
  q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c5w);
  p0 = Q6_V_vmux_QVV(q0, p0_32, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_32, p1);

  // 如果指数差值更大，选择 p^64。
  q0 = Q6_Q_vcmp_gt_VwVw(x0exp, c5w);
  q1 = Q6_Q_vcmp_gt_VwVw(x1exp, c5w);
  p0 = Q6_V_vmux_QVV(q0, p0_64, p0);
  p1 = Q6_V_vmux_QVV(q1, p1_64, p1);
```

最后把这一轮的未归一化 exp 写到 `pout`，并累加到 `vsumf`：

```cpp
  if (d >= 64) {
    // p0/p1 合成一个 vector pair，转成 fp16 后写到 pout。
    // 注意：这里的 pout 暂时保存的是未归一化 exp，不是最终 softmax。
    p10 = Q6_W_vcombine_VV(p1, p0);
    q6op_vstu_AV(optr, Q6_Vhf_equals_Wqf32(p10));
    optr++;

    // 累加 softmax 分母。
    vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p0);
    vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p1);
  } else {
    // 尾部不足 64 个 fp16 时，用 predicate 清掉无效 lane。
    // 这里 qf32 每个 lane 是 4 字节，所以 mask 长度乘 4。
    HVX_VectorPred Q0 = Q6_Q_vsetq2_R(4 * (((d & 63) + 1) / 2));
    HVX_VectorPred Q1 = Q6_Q_vsetq2_R(4 * (((d & 63) + 0) / 2));
    p0                = Q6_V_vmux_QVV(Q0, p0, vzero);
    p1                = Q6_V_vmux_QVV(Q1, p1, vzero);

    // 只累加有效 lane。
    vsumf             = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p0);
    vsumf             = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p1);

    // 只写回有效 fp16 元素。
    p10               = Q6_W_vcombine_VV(p1, p0);
    q6op_vstu_variable_ARV(optr, 2 * (d & 63), Q6_Vhf_equals_Wqf32(p10));
    optr++;
  }
}
```

#### 1.5.4 sum 并归一化

源码第 287 到 292 行把 `vsumf` 横向归约，得到整段 softmax 的分母：

```cpp
// vsumf 是 vector 形式的部分和。这里把 vector 内所有 lane 继续横向相加。
for (int i = 0, nshift = 4; i < 5; i++) {
  HVX_VectorPair temps = Q6_W_vshuff_VVR(vsumf, vsumf, nshift);
  vsumf                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
  nshift <<= 1;
}

// 转成 scalar float 格式，便于从 lane 0 取出。
vsumf = Q6_Vsf_equals_Vqf32(vsumf);
```

第 295 到 297 行取出 sum 并求倒数：

```cpp
// 横向归约后，每个 lane 都等价地携带总和；取 lane 0 即可。
sum.i = Q6_R_vextract_VR(vsumf, 0);
sum_recip.f = 1.0f / sum.f;

// 把 1/sum 广播成 vector，后面逐元素乘。
HVX_Vector vrecip = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(sum_recip.i), vzero);
HVX_Vector *ioptr = (HVX_Vector *)pout;
```

第 300 到 314 行第二次遍历输出 buffer，把前面暂存的 exp 乘以 `1 / sum`。这里的输入和输出都是 `pout`，相当于原地归一化：

```cpp
// 处理完整的 64 个 fp16 block。
for (int d = length; d > 63; d -= 64) {
  // 从 pout 读取前一节暂存的未归一化 exp，并扩展到 qf32。
  HVX_VectorPair xx = Q6_Wqf32_vmpy_VhfVhf(vmemu(&ioptr[0]), voneh);

  // lo/hi 两半分别乘以 1/sum。
  HVX_Vector xl =
      Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xx), vzero), vrecip);
  HVX_Vector xh =
      Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xx), vzero), vrecip);

  // 合并并转回 fp16，写回最终 softmax 输出。
  xx = Q6_W_vcombine_VV(xh, xl);
  q6op_vstu_AV(ioptr, Q6_Vhf_equals_Wqf32(xx));
  ioptr++;
}

// 处理尾部不足 64 个 fp16 的 block。
if ((length & 63) != 0) {
  HVX_VectorPair xx = Q6_Wqf32_vmpy_VhfVhf(vmemu(&ioptr[0]), voneh);
  HVX_Vector xl =
      Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xx), vzero), vrecip);
  HVX_Vector xh =
      Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xx), vzero), vrecip);

  xx = Q6_W_vcombine_VV(xh, xl);

  // 只写回有效的 fp16 元素。
  q6op_vstu_variable_ARV(ioptr, (length & 63) * 2, Q6_Vhf_equals_Wqf32(xx));
}
```

这一步对应：

$$
out_i = \frac{\exp(scale \cdot (x_i - max))}{\sum_j \exp(scale \cdot (x_j - max))}
$$

#### 1.5.5 `softmax_fp_impl()`

源码第 318 到 333 行是 `Softmax_fp` 的外层调度函数。它本身不写 HVX 计算细节，只负责按 `[B, H, W]` 遍历，把每段最后一维切片交给 `softmax_hf_approx()`：

```cpp
template <typename OutTtype, typename InTtype>
int softmax_fp_impl(OutTtype &out, const InTtype &in, const Tensor &beta) {
  // 输出 shape 与输入一致。
  out.set_dims(in);

  // 示例实现按 4D tensor 处理：[B, H, W, D]。
  // softmax 固定沿最后一维 D 做。
  auto [b_in, h_in, w_in, d_in] = in.dims();

  // beta 是 softmax 的指数缩放参数。
  // interface_scale 是输入 tensor 接口层的 scale，这里合进总 scale。
  float scale = in.interface_scale() * beta(0, 0, 0, 0);

  // 外层 B/H/W 逐切片遍历。
  // 每个切片是一段连续的 D 维 fp16 数据。
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        using T                               = typename InTtype::element_type;

        // 当前 [b, h, w, :] 的输入首地址。
        const T *pin                          = &in.get_raw(b, h, w, 0);

        // 当前 [b, h, w, :] 的输出首地址。
        typename OutTtype::element_type *pout = &out.get_raw(b, h, w, 0);

        // 对最后一维做 softmax。
        softmax_hf_approx(pout, pin, scale, d_in);
      }
    }
  }
  return GraphStatus::Success;
}
```

也就是说，这个示例的 `Softmax_fp` 固定做最后一维 softmax；`axis` 在前面的优化规则里已经被规整掉了。

## 1.6 reference 版本

源码第 337 到 367 行是普通 C++ 版本：

```cpp
template <typename Ttype>
int softmax_impl(Ttype &out, const Ttype &in, const float beta) {
  // debuglog("reference softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  // 示例实现按 4D tensor 处理：[B, H, W, D]。
  auto [b_in, h_in, w_in, d_in] = in.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        float max = in(b, h, w, 0);
        // 先找最大值，保证数值稳定。
        for (Idx d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          max               = fmaxf(inval, max);
        }
        float sum = 0;
        // 计算未归一化的 exp 并且求和
        for (Idx d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          sum += (out(b, h, w, d) = expf(beta * (inval - max)));
        }
        float const sum_recip = 1.0f / sum;
        // 归一化
        for (Idx d = 0; d < d_in; d++) {
          float const outval = out(b, h, w, d);
          out(b, h, w, d)    = outval * sum_recip;
        }
      }
    }
  }
  return GraphStatus::Success;
}
```

1. 先找最大值
2. 计算 `exp(beta * (x - max))`
3. 求和
4. 除以 sum

`softmaxWithbetaWrapper()` 只是把 `Tensor beta` 里的第一个 scalar 取出来，再调用这个 reference 实现。

# 2. `Q6_Vqf32_vmpy_Vqf32Vqf32` 等类型

这不是 QNN API，也不是 C++ 标准库函数。它是 Qualcomm Hexagon/HVX intrinsic。

可以按名字拆：

```text
Q6_ Vqf32 _vmpy_ Vqf32 Vqf32
|   |       |      |     |
|   |       |      |     +-- 第二个输入类型：qf32 vector
|   |       |      +-------- 第一个输入类型：qf32 vector
|   |       +--------------- 操作：vector multiply
|   +----------------------- 返回类型：qf32 vector
+--------------------------- Hexagon Q6 intrinsic 前缀
```

这里几个缩写大致这样读：

| 片段 | 含义 |
|---|---|
| `Q6` | Hexagon Q6 intrinsic 前缀 |
| `V` | HVX vector |
| `W` | HVX vector pair |
| `R` | scalar register |
| `Q` | predicate/mask |
| `vmpy` | vector multiply |
| `vadd` | vector add |
| `Vhf` | vector half-float |
| `Vqf32` | vector qfloat32/内部扩展浮点表示 |
| `Vsf` | vector single-float |

所以在这份 softmax 里，`Q6_Vqf32_vmpy_Vqf32Vqf32(a, b)` 主要出现在三类地方：

- 把 `(x - max)` 乘以 `beta` 缩放。
- 多项式近似里做乘法。
- 最后把未归一化输出乘以 `1 / sum`。

直觉上可以把它当成：

```cpp
for each lane:
  out[i] = a[i] * b[i]
```

但它的真实数据格式、lane 数、舍入行为、饱和行为要以 Hexagon intrinsic 文档为准。
