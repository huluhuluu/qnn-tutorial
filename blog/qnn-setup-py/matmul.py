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

torch.onnx.export(
    model,
    input_tensor,
    str(root / "matmul.onnx"),
    input_names=["A"],
    output_names=["Y"],
    opset_version=13,
    do_constant_folding=False,
)

A.tofile(root / "input.raw")
np.save(root / "reference.npy", Y)
(root / "input_list.txt").write_text(str((root / "input.raw").resolve()) + "\n")