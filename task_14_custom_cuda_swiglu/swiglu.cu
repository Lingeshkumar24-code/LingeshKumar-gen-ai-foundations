#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sig = 1.0f / (1.0f + expf(-y[idx]));
        out[idx] = x[idx] * (y[idx] * sig);
    }
}
