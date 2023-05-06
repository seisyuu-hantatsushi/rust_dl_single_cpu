
extern "C" __global__ void
tensor_add(const float *pX, const float *pY, float *pZ, int elements){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < elements) pZ[tid] = pX[tid] + pY[tid];
}

extern "C" __global__ void
tensor_sub(const float *pX, const float *pY, float *pZ, int elements){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < elements) pZ[tid] = pX[tid] - pY[tid];
}
