#include <torch/extension.h>

void launch_swiglu_kernel(const float* x, const float* y, float* out, int n);

torch::Tensor swiglu_forward(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    launch_swiglu_kernel(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swiglu_forward, "SwiGLU forward (CUDA)");
}
