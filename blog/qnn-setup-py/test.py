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