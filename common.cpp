#include "common.h"

#ifdef SPONGE_CPU_PHASE1
// CPU compatibility fallback for kernels that reference dynamic shared memory.
// This provides a link-visible symbol for `extern __shared__ char shared_memory[]`.
char shared_memory[64 * 1024 * 1024];

// Shared across all translation units for CPU kernel launch emulation.
thread_local dim3 __cpu_threadIdx_runtime(0u, 0u, 0u);
thread_local dim3 __cpu_blockIdx_runtime(0u, 0u, 0u);
thread_local dim3 __cpu_blockDim_runtime(1u, 1u, 1u);
thread_local dim3 __cpu_gridDim_runtime(1u, 1u, 1u);
#endif

void Reset_List(int *list, const int replace_element, const int element_numbers, const int threads)
{
    CUDA_LAUNCH(Reset_List,
        dim3((unsigned int)ceilf((float)element_numbers / threads)),
        dim3(static_cast<unsigned int>(threads)),
        element_numbers,
        list,
        replace_element);
}

void Reset_List(float *list, const float replace_element, const int element_numbers, const int threads)
{
    CUDA_LAUNCH(Reset_List,
        dim3((unsigned int)ceilf((float)element_numbers / threads)),
        dim3(static_cast<unsigned int>(threads)),
        element_numbers,
        list,
        replace_element);
}

__global__ void Reset_List(const int element_numbers, int *list, const int replace_element)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = replace_element;
    }
}
__global__ void Reset_List(const int element_numbers, float *list, const float replace_element)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = replace_element;
    }
}

void Scale_List(float *list, const float scaler, const int element_numbers, int threads)
{
    CUDA_LAUNCH(Scale_List,
        dim3((unsigned int)ceilf((float)element_numbers / threads)),
        dim3(static_cast<unsigned int>(threads)),
        element_numbers,
        list,
        scaler);
}

__global__ void Scale_List(const int element_numbers, float *list, float scaler)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = list[i] * scaler;
    }
}
__global__ void Copy_List(const int element_numbers, const int *origin_list, int *list)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = origin_list[i];
    }
}
__global__ void Copy_List(const int element_numbers, const float *origin_list, float *list)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = origin_list[i];
    }
}
__global__ void Inverse_List_Element(const int element_numbers, const float *origin_list, float *list)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < element_numbers)
    {
        list[i] = 1./origin_list[i];
    }
}

void Sum_Of_List(const int *list, int *sum, const int element_numbers, int threads)
{
    CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(static_cast<unsigned int>(threads)), element_numbers, list, sum);
}

void Sum_Of_List(const float *list, float *sum, const int end, const int start, int threads)
{
    CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(static_cast<unsigned int>(threads)), start, end, list, sum);
}

__global__ void Sum_Of_List(const int element_numbers,const int* list , int *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0] = 0;
    }
    __syncthreads();
    int lin = 0;
    for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x)
    {
        lin = lin + list[i];
    }
    atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int start, const int end, const float* list, float *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x + start; i < end; i = i + blockDim.x)
    {
        lin = lin + list[i];
    }
    atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int element_numbers, const float* list, float *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x)
    {
        lin = lin + list[i];
    }
    atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int element_numbers, const VECTOR* list, VECTOR *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0].x = 0.;
        sum[0].y = 0.;
        sum[0].z = 0.;
    }
    __syncthreads();
    VECTOR lin = { 0., 0., 0. };
    for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x)
    {
        lin.x = lin.x + list[i].x;
        lin.y = lin.y + list[i].y;
        lin.z = lin.z + list[i].z;
    }
    atomicAdd(&sum[0].x, lin.x);
    atomicAdd(&sum[0].y, lin.y);
    atomicAdd(&sum[0].z, lin.z);
}

__global__ void Setup_Rand_Normal_Kernel(const int float4_numbers, curandStatePhilox4_32_10_t *rand_state, const int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
    number, no offset */
    if (id < float4_numbers)
    {
        curand_init(seed, id, 0, &rand_state[id]);
    }
}
__global__ void Rand_Normal(const int float4_numbers, curandStatePhilox4_32_10_t *rand_state, float4 *rand_float4)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < float4_numbers)
    {
        rand_float4[i] = curand_normal4(&rand_state[i]);
    }
}

__global__ void Cuda_Debug_Print(const float *x)
{
    printf("DEBUG: %e\n", x[0]);
}

__global__ void Cuda_Debug_Print(const VECTOR *x)
{
    printf("DEBUG: %e %e %e\n", x[0].x, x[0].y, x[0].z);
}

__global__ void Cuda_Debug_Print(const int *x)
{
    printf("DEBUG: %d\n", x[0]);
}

int Check_2357_Factor(int number)
{
    int tempn;
    while (number>0)
    {
        if (number == 1)
            return 1;
        tempn = number / 2;
        if (tempn * 2 != number)
            break;
        number = tempn;
    }

    while (number>0)
    {
        if (number == 1)
            return 1;
        tempn = number / 3;
        if (tempn * 3 != number)
            break;
        number = tempn;
    }

    while (number>0)
    {
        if (number == 1)
            return 1;
        tempn = number / 5;
        if (tempn * 5 != number)
            break;
        number = tempn;
    }

    while (number>0)
    {
        if (number == 1)
            return 1;
        tempn = number / 7;
        if (tempn * 7 != number)
            break;
        number = tempn;
    }

    return 0;
}


static __global__ void upSweep(int* d_out, int* d_in, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (index < d) 
        {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;
            d_in[bi] += d_in[ai];
        }
        offset *= 2;
    }

    if (index == 0) 
    {
        d_out[n - 1] = d_in[n - 1];
        d_in[n - 1] = 0;
    }
}

static __global__ void downSweep(int* d_out, int* d_in, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = n;
    for (int d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (index < d) 
        {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;

            int temp = d_in[ai];
            d_in[ai] = d_in[bi];
            d_in[bi] += temp;
        }
    }

    __syncthreads();

    if (index < n) 
    {
        d_out[index] = d_in[index];
    }
}

Prefix_Sum::Prefix_Sum(int size)
{
    padded_size = 1;
    while (padded_size < size)
    {
        padded_size *= 2;
    }
    cudaMalloc((void**)&in, padded_size * sizeof(int));
    cudaMemset(in, 0, padded_size * sizeof(int));
    cudaMalloc((void**)&temp, padded_size * sizeof(int));
    cudaMemset(temp, 0, padded_size * sizeof(int));
    cudaMalloc((void**)&out, (padded_size + 1) * sizeof(int));
    cudaMemset(out, 0, (padded_size + 1) * sizeof(int));
    blockSize = std::min(padded_size, 1024);
    gridSize = (padded_size + blockSize - 1) / blockSize;
}

void Prefix_Sum::Scan()
{
    cudaMemcpy(temp, in, sizeof(int) * padded_size, cudaMemcpyDeviceToDevice);
    CUDA_LAUNCH(upSweep, dim3(static_cast<unsigned int>(gridSize)), dim3(static_cast<unsigned int>(blockSize)), out, temp, padded_size);
    CUDA_LAUNCH(downSweep, dim3(static_cast<unsigned int>(gridSize)), dim3(static_cast<unsigned int>(blockSize)), out, temp, padded_size);
}

int Get_Fft_Patameter(float length)
{
    int tempi = (int)ceil(length + 3) >> 2 << 2;

    if (tempi >= 60 && tempi <= 68)
        tempi = 64;
    else if (tempi >= 120 && tempi <= 136)
        tempi = 128;
    else if (tempi >= 240 && tempi <= 272)
        tempi = 256;
    else if (tempi >= 480 && tempi <= 544)
        tempi = 512;
    else if (tempi >= 960 && tempi <= 1088)
        tempi = 1024;

    while (1)
    {
        if (Check_2357_Factor(tempi))
            return tempi;
        tempi += 4;
    }
}
