// Compile-time-specialized fused D3Q19 pull-stream + BGK collision.
//
// The host replaces the {{...}} tokens before NVRTC compilation. The benchmark
// fixes N=128, but parity runs use smaller power-of-two domains. GPU-native SoA
// is intentional; the mathematical update, FP32 precision, and conservation-form
// rest-population residual match the corrected TT-Metal implementation.
#define LBM_N {{N}}
#define LBM_MASK {{MASK}}
#define LBM_LOG2_N {{LOG2_N}}
#define LBM_LOG2_PLANE {{LOG2_PLANE}}
#define LBM_CELLS {{CELLS}}
#define LBM_BLOCK {{BLOCK}}

__device__ __constant__ int D3Q19_CX[19] =
    {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__device__ __constant__ int D3Q19_CY[19] =
    {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
__device__ __constant__ int D3Q19_CZ[19] =
    {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

__device__ __forceinline__ unsigned int wrap_sub_pow2(
    unsigned int x, int delta
) {
    return (x - (unsigned int)delta) & LBM_MASK;
}

__device__ __forceinline__ float d3q19_weight(int q) {
    return q == 0 ? (1.0f / 3.0f)
                  : (q <= 6 ? (1.0f / 18.0f) : (1.0f / 36.0f));
}

extern "C" __global__ __launch_bounds__(LBM_BLOCK) void cuda_lbm_step(
    const float* __restrict__ input,
    float* __restrict__ output,
    float omega
) {
    const unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= LBM_CELLS) return;

    const unsigned int x = cell >> LBM_LOG2_PLANE;
    const unsigned int y = (cell >> LBM_LOG2_N) & LBM_MASK;
    const unsigned int z = cell & LBM_MASK;

    float f[19];
    float rho = 0.0f;
    float mx = 0.0f;
    float my = 0.0f;
    float mz = 0.0f;

    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        const unsigned int sx = wrap_sub_pow2(x, D3Q19_CX[q]);
        const unsigned int sy = wrap_sub_pow2(y, D3Q19_CY[q]);
        const unsigned int sz = wrap_sub_pow2(z, D3Q19_CZ[q]);
        const unsigned int source =
            (sx << LBM_LOG2_PLANE) | (sy << LBM_LOG2_N) | sz;
        const float fq = input[(unsigned int)q * LBM_CELLS + source];
        f[q] = fq;
        rho += fq;
        mx += (float)D3Q19_CX[q] * fq;
        my += (float)D3Q19_CY[q] * fq;
        mz += (float)D3Q19_CZ[q] * fq;
    }

    const float inv_rho = 1.0f / rho;
    const float ux = mx * inv_rho;
    const float uy = my * inv_rho;
    const float uz = mz * inv_rho;
    const float u2 = ux * ux + uy * uy + uz * uz;

    float moving_sum = 0.0f;
    #pragma unroll
    for (int q = 1; q < 19; ++q) {
        const float cu = 3.0f *
            ((float)D3Q19_CX[q] * ux +
             (float)D3Q19_CY[q] * uy +
             (float)D3Q19_CZ[q] * uz);
        const float feq = d3q19_weight(q) * rho *
            (1.0f + cu + 0.5f * cu * cu - 1.5f * u2);
        const float value = f[q] - omega * (f[q] - feq);
        output[(unsigned int)q * LBM_CELLS + cell] = value;
        moving_sum += value;
    }

    output[cell] = rho - moving_sum;
}
