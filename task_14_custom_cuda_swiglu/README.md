# Task 14 — CUDA SwiGLU Extension

The PDF requires a real C++/CUDA implementation compiled through `torch.utils.cpp_extension` and benchmarked against PyTorch.

Build pattern:

```python
from torch.utils.cpp_extension import load
ext = load(name='swiglu_ext', sources=['swiglu.cpp','swiglu.cu'], verbose=True)
```

The CUDA kernel should compute `x * y * sigmoid(y)` elementwise on GPU. Benchmark with CUDA synchronization around both implementations; never print a hard-coded speedup. The checked-in `swiglu.cpp`/`swiglu.cu` files are the extension sources to complete/test on a CUDA-enabled machine.
