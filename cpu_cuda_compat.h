#pragma once

/*
 * CPU compatibility shim for CUDA-centric C++ code.
 *
 * Use this in CPU migration builds to keep transitional code compiling while
 * kernels are being mechanically converted.
 *
 * Default safety policy:
 * - compile-time compatibility is provided;
 * - high-risk CUDA-library/JIT stubs fail fast at runtime.
 *
 * Override behavior for compile-only smoke runs with:
 * -DCPU_CUDA_COMPAT_ABORT_ON_STUB_USE=0
 *
 * Do not treat this as a permanent replacement for explicit backend migration.
 */

#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if __has_include(<fftw3.h>)
#include <fftw3.h>
#define CPU_CUDA_COMPAT_HAVE_FFTW3 1
#else
#define CPU_CUDA_COMPAT_HAVE_FFTW3 0
#endif

#ifndef CPU_CUDA_COMPAT_COMPILE_ONLY
#define CPU_CUDA_COMPAT_COMPILE_ONLY 0
#endif

#ifndef CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
#define CPU_CUDA_COMPAT_ABORT_ON_STUB_USE 1
#endif

#ifndef CPU_CUDA_COMPAT_USE_OPENMP
#if defined(_OPENMP)
#define CPU_CUDA_COMPAT_USE_OPENMP 1
#else
#define CPU_CUDA_COMPAT_USE_OPENMP 0
#endif
#endif

#ifndef CPU_CUDA_COMPAT_OMP_LAUNCH_DEFAULT
#define CPU_CUDA_COMPAT_OMP_LAUNCH_DEFAULT 0
#endif

static inline void cpu_cuda_stub_abort(const char* fn) {
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE && !CPU_CUDA_COMPAT_COMPILE_ONLY
    std::fprintf(stderr, "FATAL: CUDA stub called on CPU path: %s\n", fn);
    std::fflush(stderr);
    std::abort();
#else
    (void)fn;
#endif
}

static inline bool cpu_cuda_env_enabled(const char* value, bool default_value) {
    if (value == NULL || value[0] == '\0') return default_value;
    const unsigned char head = static_cast<unsigned char>(value[0]);
    const char lower = static_cast<char>(std::tolower(head));
    if (lower == '0' || lower == 'f' || lower == 'n') return false;
    return true;
}

static inline bool cpu_cuda_kernel_omp_launch_enabled() {
#if CPU_CUDA_COMPAT_USE_OPENMP
    static const bool enabled = []() {
        const char* env = std::getenv("SPONGE_CPU_OMP_LAUNCH");
        return cpu_cuda_env_enabled(env, CPU_CUDA_COMPAT_OMP_LAUNCH_DEFAULT != 0);
    }();
    return enabled;
#else
    return false;
#endif
}

static inline int cpu_cuda_kernel_omp_min_blocks() {
#if CPU_CUDA_COMPAT_USE_OPENMP
    static const int min_blocks = []() {
        const char* env = std::getenv("SPONGE_CPU_OMP_MIN_BLOCKS");
        if (env == NULL || env[0] == '\0') return 64;
        char* endptr = NULL;
        long parsed = std::strtol(env, &endptr, 10);
        if (endptr == env || parsed <= 0) return 64;
        if (parsed > 1048576) return 1048576;
        return static_cast<int>(parsed);
    }();
    return min_blocks;
#else
    return 0;
#endif
}

static inline bool cpu_cuda_in_parallel_region() {
#if CPU_CUDA_COMPAT_USE_OPENMP && defined(_OPENMP)
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

/* CUDA qualifiers */
#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__ static const
#endif

/* CUDA dim3 and thread symbols (compile-only placeholders) */
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    constexpr dim3(unsigned int vx = 1u, unsigned int vy = 1u, unsigned int vz = 1u)
        : x(vx), y(vy), z(vz) {}
};

/*
 * Launch runtime indices must be process-global (thread-local), not
 * translation-unit local. Otherwise kernels defined in one TU but launched
 * from another read stale thread/block state in CPU compatibility mode.
 */
extern thread_local dim3 __cpu_threadIdx_runtime;
extern thread_local dim3 __cpu_blockIdx_runtime;
extern thread_local dim3 __cpu_blockDim_runtime;
extern thread_local dim3 __cpu_gridDim_runtime;

#ifndef threadIdx
#define threadIdx (__cpu_threadIdx_runtime)
#endif
#ifndef blockIdx
#define blockIdx (__cpu_blockIdx_runtime)
#endif
#ifndef blockDim
#define blockDim (__cpu_blockDim_runtime)
#endif
#ifndef gridDim
#define gridDim (__cpu_gridDim_runtime)
#endif

template <typename KernelCall>
static inline void cpu_launch_kernel(dim3 grid, dim3 block, KernelCall&& kernel_call) {
    if (grid.x == 0u) grid.x = 1u;
    if (grid.y == 0u) grid.y = 1u;
    if (grid.z == 0u) grid.z = 1u;
    if (block.x == 0u) block.x = 1u;
    if (block.y == 0u) block.y = 1u;
    if (block.z == 0u) block.z = 1u;
    const int gx = static_cast<int>(grid.x);
    const int gy = static_cast<int>(grid.y);
    const int gz = static_cast<int>(grid.z);
    const bool one_dimensional = (gy == 1 && gz == 1 && block.y == 1u && block.z == 1u);
    const int total_blocks = gx * gy * gz;
    if (cpu_cuda_kernel_omp_launch_enabled() &&
        total_blocks >= cpu_cuda_kernel_omp_min_blocks()) {
#if CPU_CUDA_COMPAT_USE_OPENMP
        /*
         * Parallelize across blocks only. Threads inside one block remain serial
         * to preserve current CPU shim behavior for kernels that rely on
         * __syncthreads/__shared__ placeholders.
         */
#pragma omp parallel
        {
            __cpu_gridDim_runtime = grid;
            __cpu_blockDim_runtime = block;
            if (one_dimensional) {
                __cpu_threadIdx_runtime.y = 0u;
                __cpu_threadIdx_runtime.z = 0u;
#pragma omp for schedule(static)
                for (int block_linear = 0; block_linear < total_blocks; ++block_linear) {
                    __cpu_blockIdx_runtime.x = static_cast<unsigned int>(block_linear);
                    __cpu_blockIdx_runtime.y = 0u;
                    __cpu_blockIdx_runtime.z = 0u;
                    for (unsigned int tx = 0; tx < block.x; ++tx) {
                        __cpu_threadIdx_runtime.x = tx;
                        kernel_call();
                    }
                }
            } else {
#pragma omp for schedule(static)
                for (int block_linear = 0; block_linear < total_blocks; ++block_linear) {
                    const int bx = block_linear % gx;
                    const int by = (block_linear / gx) % gy;
                    const int bz = block_linear / (gx * gy);
                    __cpu_blockIdx_runtime.x = static_cast<unsigned int>(bx);
                    __cpu_blockIdx_runtime.y = static_cast<unsigned int>(by);
                    __cpu_blockIdx_runtime.z = static_cast<unsigned int>(bz);
                    for (unsigned int tz = 0; tz < block.z; ++tz) {
                        __cpu_threadIdx_runtime.z = tz;
                        for (unsigned int ty = 0; ty < block.y; ++ty) {
                            __cpu_threadIdx_runtime.y = ty;
                            for (unsigned int tx = 0; tx < block.x; ++tx) {
                                __cpu_threadIdx_runtime.x = tx;
                                kernel_call();
                            }
                        }
                    }
                }
            }
        }
        return;
#endif
    }
    __cpu_gridDim_runtime = grid;
    __cpu_blockDim_runtime = block;
    if (one_dimensional) {
        __cpu_blockIdx_runtime.y = 0u;
        __cpu_blockIdx_runtime.z = 0u;
        __cpu_threadIdx_runtime.y = 0u;
        __cpu_threadIdx_runtime.z = 0u;
        for (int bx = 0; bx < gx; ++bx) {
            __cpu_blockIdx_runtime.x = static_cast<unsigned int>(bx);
            for (unsigned int tx = 0; tx < block.x; ++tx) {
                __cpu_threadIdx_runtime.x = tx;
                kernel_call();
            }
        }
        return;
    }
    for (int bz = 0; bz < gz; ++bz) {
        __cpu_blockIdx_runtime.z = static_cast<unsigned int>(bz);
        for (int by = 0; by < gy; ++by) {
            __cpu_blockIdx_runtime.y = static_cast<unsigned int>(by);
            for (int bx = 0; bx < gx; ++bx) {
                __cpu_blockIdx_runtime.x = static_cast<unsigned int>(bx);
                for (unsigned int tz = 0; tz < block.z; ++tz) {
                    __cpu_threadIdx_runtime.z = tz;
                    for (unsigned int ty = 0; ty < block.y; ++ty) {
                        __cpu_threadIdx_runtime.y = ty;
                        for (unsigned int tx = 0; tx < block.x; ++tx) {
                            __cpu_threadIdx_runtime.x = tx;
                            kernel_call();
                        }
                    }
                }
            }
        }
    }
}

/* Vector types */
struct float3 {
    float x;
    float y;
    float z;
};

struct float2 {
    float x;
    float y;
};

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

static inline float4 make_float4(float x, float y, float z, float w) {
    float4 out{x, y, z, w};
    return out;
}

/* CUDA-style math intrinsics used in legacy kernels */
static inline float norm3df(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

static inline float rnorm3df(float x, float y, float z) {
    const float n = norm3df(x, y, z);
    return n == 0.0f ? 0.0f : 1.0f / n;
}

// Approximate CUDA erfcxf using exp(x^2) * erfc(x) for CPU compatibility.
static inline float erfcxf(float x) {
    return std::exp(x * x) * std::erfc(x);
}

/* Transitional atomicAdd shim */
static inline float atomicAdd(float* addr, float val) {
#ifdef _OPENMP
    if (!cpu_cuda_in_parallel_region()) {
        const float old = *addr;
        *addr += val;
        return old;
    }
    float old;
#pragma omp atomic capture
    {
        old = *addr;
        *addr += val;
    }
    return old;
#else
    const float old = *addr;
    *addr += val;
    return old;
#endif
}

static inline int atomicAdd(int* addr, int val) {
#ifdef _OPENMP
    if (!cpu_cuda_in_parallel_region()) {
        const int old = *addr;
        *addr += val;
        return old;
    }
    int old;
#pragma omp atomic capture
    {
        old = *addr;
        *addr += val;
    }
    return old;
#else
    const int old = *addr;
    *addr += val;
    return old;
#endif
}

static inline int atomicCAS(int* addr, int compare, int val) {
#ifdef _OPENMP
    if (!cpu_cuda_in_parallel_region()) {
        const int old = *addr;
        if (old == compare) {
            *addr = val;
        }
        return old;
    }
    int old;
#pragma omp critical(cpu_cuda_atomic_cas_int)
    {
        old = *addr;
        if (old == compare) {
            *addr = val;
        }
    }
    return old;
#else
    const int old = *addr;
    if (old == compare) {
        *addr = val;
    }
    return old;
#endif
}

static inline int atomicExch(int* addr, int val) {
#ifdef _OPENMP
    if (!cpu_cuda_in_parallel_region()) {
        const int old = *addr;
        *addr = val;
        return old;
    }
    int old;
#pragma omp critical(cpu_cuda_atomic_exch_int)
    {
        old = *addr;
        *addr = val;
    }
    return old;
#else
    const int old = *addr;
    *addr = val;
    return old;
#endif
}

/* CUDA sync/warp intrinsics (compile-only placeholders for serial CPU path) */
static inline void __syncthreads() {}

static inline void __threadfence_block() {}

static inline void __syncwarp(unsigned mask = 0xffffffffu) {
    (void)mask;
}

static inline unsigned __ballot_sync(unsigned mask, int pred) {
    (void)pred;
    return mask;
}

template <typename T>
static inline T __shfl_down_sync(unsigned mask, T val, unsigned delta, int width = 32) {
    (void)mask;
    (void)delta;
    (void)width;
    return val;
}

template <typename T>
static inline T __shfl_xor_sync(unsigned mask, T val, int lane_mask, int width = 32) {
    (void)mask;
    (void)lane_mask;
    (void)width;
    return val;
}

/* CUDA Runtime API stubs */
typedef int cudaError_t;
typedef cudaError_t cudaError;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidDevice = 10;
static const cudaError_t cudaErrorNotSupported = 801;

struct cudaDeviceProp {
    char name[256];
    std::size_t totalGlobalMem;
    int major;
    int minor;
};

static inline cudaError_t cudaMalloc(void** ptr, std::size_t size) {
    *ptr = std::malloc(size);
    return *ptr ? cudaSuccess : 1;
}

static inline cudaError_t cudaMallocHost(void** ptr, std::size_t size) {
    return cudaMalloc(ptr, size);
}

static inline cudaError_t cudaHostAlloc(void** ptr, std::size_t size, unsigned int) {
    return cudaMalloc(ptr, size);
}

static inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

static inline cudaError_t cudaFreeHost(void* ptr) {
    return cudaFree(ptr);
}

static inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

static inline cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    cudaStream_t = nullptr) {
    return cudaMemcpy(dst, src, count, kind);
}

static inline cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
    std::memset(dst, value, count);
    return cudaSuccess;
}

static inline cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t = nullptr) {
    return cudaMemset(dst, value, count);
}

static inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    *stream = nullptr;
    return cudaSuccess;
}

static inline cudaError_t cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}

static inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}

static inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = nullptr;
    return cudaSuccess;
}

static inline cudaError_t cudaEventDestroy(cudaEvent_t) {
    return cudaSuccess;
}

static inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) {
    return cudaSuccess;
}

static inline cudaError_t cudaEventSynchronize(cudaEvent_t) {
    return cudaSuccess;
}

static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    if (ms) {
        *ms = 0.0f;
    }
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceReset() {
    return cudaSuccess;
}

static inline cudaError_t cudaGetDeviceCount(int* count) {
    if (count) {
        *count = 1;
    }
    return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int) {
    return cudaSuccess;
}

static inline cudaError_t cudaGetDevice(int* device) {
    if (device) {
        *device = 0;
    }
    return cudaSuccess;
}

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    if (!prop) {
        return cudaErrorInvalidDevice;
    }
    std::memset(prop, 0, sizeof(cudaDeviceProp));
    std::snprintf(prop->name, sizeof(prop->name), "%s", "cpu_cuda_compat");
    prop->totalGlobalMem = static_cast<std::size_t>(64) << 30;
    prop->major = 0;
    prop->minor = 0;
    return cudaSuccess;
}

static inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

static inline const char* cudaGetErrorName(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorNotSupported: return "cudaErrorNotSupported";
        default: return "cpu_cuda_compat_error";
    }
}

static inline const char* cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "success";
        case cudaErrorInvalidDevice: return "invalid device";
        case cudaErrorNotSupported: return "not supported in cpu_cuda_compat";
        default: return "cpu_cuda_compat error";
    }
}

template <typename KernelFunc>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int* minGridSize,
    int* blockSize,
    KernelFunc,
    std::size_t = 0,
    int = 0) {
    if (minGridSize) {
        *minGridSize = 1;
    }
    if (blockSize) {
        *blockSize = 1;
    }
    return cudaSuccess;
}

/* CUDA Driver API stubs */
typedef int CUresult;
typedef void* CUmodule;
typedef void* CUfunction;

static const CUresult CUDA_SUCCESS = 0;
static const CUresult CUDA_ERROR_NOT_SUPPORTED = 801;

static inline CUresult cuInit(unsigned int) {
    return CUDA_SUCCESS;
}

static inline CUresult cuModuleLoadDataEx(CUmodule*, const void*, unsigned int, void*, void*) {
    cpu_cuda_stub_abort("cuModuleLoadDataEx");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*) {
    cpu_cuda_stub_abort("cuModuleGetFunction");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLaunchKernel(
    CUfunction,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    cudaStream_t,
    void**,
    void**) {
    cpu_cuda_stub_abort("cuLaunchKernel");
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* NVRTC stubs */
typedef int nvrtcResult;
typedef void* nvrtcProgram;

static const nvrtcResult NVRTC_SUCCESS = 0;
static const nvrtcResult NVRTC_ERROR_INTERNAL_ERROR = 11;

static inline nvrtcResult nvrtcCreateProgram(...) {
    cpu_cuda_stub_abort("nvrtcCreateProgram");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcCompileProgram(...) {
    cpu_cuda_stub_abort("nvrtcCompileProgram");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcGetProgramLogSize(...) {
    cpu_cuda_stub_abort("nvrtcGetProgramLogSize");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcGetProgramLog(...) {
    cpu_cuda_stub_abort("nvrtcGetProgramLog");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcGetPTXSize(...) {
    cpu_cuda_stub_abort("nvrtcGetPTXSize");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcGetPTX(...) {
    cpu_cuda_stub_abort("nvrtcGetPTX");
    return NVRTC_ERROR_INTERNAL_ERROR;
}
static inline nvrtcResult nvrtcDestroyProgram(...) {
    cpu_cuda_stub_abort("nvrtcDestroyProgram");
    return NVRTC_ERROR_INTERNAL_ERROR;
}

/* cuRAND compatibility (Philox4x32-10) */
struct uint2 {
    unsigned int x;
    unsigned int y;
};

struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

static inline uint2 make_uint2(unsigned int x, unsigned int y) {
    uint2 out{x, y};
    return out;
}

static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    uint4 out{x, y, z, w};
    return out;
}

struct curandStatePhilox4_32_10_t {
    uint4 ctr;
    uint4 output;
    uint2 key;
    unsigned int STATE;
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

static const unsigned int CPU_PHILOX_W32_0 = 0x9E3779B9u;
static const unsigned int CPU_PHILOX_W32_1 = 0xBB67AE85u;
static const unsigned int CPU_PHILOX_M4x32_0 = 0xD2511F53u;
static const unsigned int CPU_PHILOX_M4x32_1 = 0xCD9E8D57u;

static const float CPU_CURAND_2POW32_INV = 2.3283064e-10f;
static const float CPU_CURAND_2POW32_INV_2PI = 2.3283064e-10f * 6.2831855f;

static inline unsigned int cpu_curand_mulhilo32(unsigned int a, unsigned int b, unsigned int* hi) {
    const std::uint64_t product = (std::uint64_t)a * (std::uint64_t)b;
    if (hi != NULL) {
        *hi = (unsigned int)(product >> 32);
    }
    return (unsigned int)product;
}

static inline uint4 cpu_curand_philox4x32_round(uint4 ctr, uint2 key) {
    unsigned int hi0 = 0;
    unsigned int hi1 = 0;
    const unsigned int lo0 = cpu_curand_mulhilo32(CPU_PHILOX_M4x32_0, ctr.x, &hi0);
    const unsigned int lo1 = cpu_curand_mulhilo32(CPU_PHILOX_M4x32_1, ctr.z, &hi1);
    return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

static inline uint4 cpu_curand_philox4x32_10(uint4 ctr, uint2 key) {
    for (int round = 0; round < 10; ++round) {
        ctr = cpu_curand_philox4x32_round(ctr, key);
        if (round != 9) {
            key.x += CPU_PHILOX_W32_0;
            key.y += CPU_PHILOX_W32_1;
        }
    }
    return ctr;
}

static inline void cpu_curand_philox_state_incr(curandStatePhilox4_32_10_t* state) {
    if (state == NULL) return;
    if (++state->ctr.x) return;
    if (++state->ctr.y) return;
    if (++state->ctr.z) return;
    ++state->ctr.w;
}

static inline void cpu_curand_philox_state_incr(curandStatePhilox4_32_10_t* state, unsigned long long n) {
    if (state == NULL) return;
    const unsigned int nlo = (unsigned int)n;
    unsigned int nhi = (unsigned int)(n >> 32);
    state->ctr.x += nlo;
    if (state->ctr.x < nlo) {
        ++nhi;
    }
    state->ctr.y += nhi;
    if (nhi <= state->ctr.y) {
        return;
    }
    if (++state->ctr.z) {
        return;
    }
    ++state->ctr.w;
}

static inline void cpu_curand_philox_state_incr_hi(curandStatePhilox4_32_10_t* state, unsigned long long n) {
    if (state == NULL) return;
    const unsigned int nlo = (unsigned int)n;
    unsigned int nhi = (unsigned int)(n >> 32);
    state->ctr.z += nlo;
    if (state->ctr.z < nlo) {
        ++nhi;
    }
    state->ctr.w += nhi;
}

static inline void skipahead(unsigned long long n, curandStatePhilox4_32_10_t* state) {
    if (state == NULL) return;
    state->STATE += (unsigned int)(n & 3ULL);
    n /= 4ULL;
    if (state->STATE > 3U) {
        ++n;
        state->STATE -= 4U;
    }
    cpu_curand_philox_state_incr(state, n);
    state->output = cpu_curand_philox4x32_10(state->ctr, state->key);
}

static inline void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t* state) {
    if (state == NULL) return;
    cpu_curand_philox_state_incr_hi(state, n);
    state->output = cpu_curand_philox4x32_10(state->ctr, state->key);
}

static inline void curand_init(
    unsigned long long seed,
    unsigned long long sequence,
    unsigned long long offset,
    curandStatePhilox4_32_10_t* state) {
    if (state == NULL) return;
    state->ctr = make_uint4(0u, 0u, 0u, 0u);
    state->key = make_uint2((unsigned int)seed, (unsigned int)(seed >> 32));
    state->STATE = 0u;
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.0f;
    state->boxmuller_extra_double = 0.0;
    skipahead_sequence(sequence, state);
    skipahead(offset, state);
}

static inline unsigned int curand(curandStatePhilox4_32_10_t* state) {
    if (state == NULL) return 0u;
    unsigned int ret = 0u;
    switch (state->STATE++) {
        default:
            ret = state->output.x;
            break;
        case 1:
            ret = state->output.y;
            break;
        case 2:
            ret = state->output.z;
            break;
        case 3:
            ret = state->output.w;
            break;
    }
    if (state->STATE == 4U) {
        cpu_curand_philox_state_incr(state);
        state->output = cpu_curand_philox4x32_10(state->ctr, state->key);
        state->STATE = 0U;
    }
    return ret;
}

static inline uint4 curand4(curandStatePhilox4_32_10_t* state) {
    if (state == NULL) {
        return make_uint4(0u, 0u, 0u, 0u);
    }
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    const uint4 tmp = state->output;
    cpu_curand_philox_state_incr(state);
    state->output = cpu_curand_philox4x32_10(state->ctr, state->key);
    switch (state->STATE) {
        case 0:
            return tmp;
        case 1:
            r.x = tmp.y;
            r.y = tmp.z;
            r.z = tmp.w;
            r.w = state->output.x;
            break;
        case 2:
            r.x = tmp.z;
            r.y = tmp.w;
            r.z = state->output.x;
            r.w = state->output.y;
            break;
        case 3:
            r.x = tmp.w;
            r.y = state->output.x;
            r.z = state->output.y;
            r.w = state->output.z;
            break;
        default:
            return tmp;
    }
    return r;
}

static inline float2 cpu_curand_box_muller(unsigned int x, unsigned int y) {
    const float u = x * CPU_CURAND_2POW32_INV + (CPU_CURAND_2POW32_INV * 0.5f);
    const float v = y * CPU_CURAND_2POW32_INV_2PI + (CPU_CURAND_2POW32_INV_2PI * 0.5f);
    const float s = std::sqrt(-2.0f * std::log(u));
    float2 out;
    out.x = sinf(v) * s;
    out.y = cosf(v) * s;
    return out;
}

static inline float4 curand_normal4(curandStatePhilox4_32_10_t* state) {
    if (state == NULL) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const uint4 x = curand4(state);
    const float2 first = cpu_curand_box_muller(x.x, x.y);
    const float2 second = cpu_curand_box_muller(x.z, x.w);
    return make_float4(first.x, first.y, second.x, second.y);
}

/* cuFFT/cuComplex compatibility */
typedef int cufftHandle;
typedef int cufftResult;

struct cufftComplex {
    float x;
    float y;
};

static inline cufftComplex make_cuComplex(float x, float y) {
    cufftComplex out{x, y};
    return out;
}

static inline cufftComplex cuCdivf(cufftComplex a, cufftComplex b) {
    const float denom = b.x * b.x + b.y * b.y;
    if (denom == 0.0f) {
        return make_cuComplex(0.0f, 0.0f);
    }
    return make_cuComplex((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

static const cufftResult CUFFT_SUCCESS = 0;
static const cufftResult CUFFT_STUB_ERROR = 1001;
static const int CUFFT_R2C = 0;
static const int CUFFT_C2R = 1;

struct cpu_cufft_plan_entry {
    int type;
    int rank;
    int dims[3];
    int batch;
    fftwf_plan plan;
};

static inline std::unordered_map<int, cpu_cufft_plan_entry*>& cpu_cufft_plan_table() {
    static std::unordered_map<int, cpu_cufft_plan_entry*> table;
    return table;
}

static inline std::mutex& cpu_cufft_plan_mutex() {
    static std::mutex m;
    return m;
}

static inline int& cpu_cufft_next_handle() {
    static int next_handle = 1;
    return next_handle;
}

static inline cufftResult cpu_cufft_plan_store(cufftHandle* handle, cpu_cufft_plan_entry* entry) {
    if (handle == NULL || entry == NULL || entry->plan == NULL) {
        return CUFFT_STUB_ERROR;
    }
    std::lock_guard<std::mutex> lock(cpu_cufft_plan_mutex());
    const int h = cpu_cufft_next_handle()++;
    cpu_cufft_plan_table()[h] = entry;
    *handle = h;
    return CUFFT_SUCCESS;
}

static inline cpu_cufft_plan_entry* cpu_cufft_plan_lookup(cufftHandle handle) {
    std::lock_guard<std::mutex> lock(cpu_cufft_plan_mutex());
    auto& table = cpu_cufft_plan_table();
    auto it = table.find(handle);
    return it == table.end() ? NULL : it->second;
}

static inline cufftResult cufftPlanMany(
    cufftHandle* handle,
    int rank,
    int* n,
    int* inembed,
    int istride,
    int idist,
    int* onembed,
    int ostride,
    int odist,
    int type,
    int batch) {
#if CPU_CUDA_COMPAT_HAVE_FFTW3
    if (handle == NULL || n == NULL || rank <= 0 || rank > 3 || batch <= 0) {
        return CUFFT_STUB_ERROR;
    }
    cpu_cufft_plan_entry* entry = new cpu_cufft_plan_entry();
    entry->type = type;
    entry->rank = rank;
    entry->batch = batch;
    entry->dims[0] = rank > 0 ? n[0] : 1;
    entry->dims[1] = rank > 1 ? n[1] : 1;
    entry->dims[2] = rank > 2 ? n[2] : 1;
    entry->plan = NULL;

    std::size_t logical_real = 1;
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) {
            delete entry;
            return CUFFT_STUB_ERROR;
        }
        logical_real *= (std::size_t)n[i];
    }
    const int last_dim = n[rank - 1];
    const std::size_t logical_complex = logical_real / (std::size_t)last_dim * (std::size_t)(last_dim / 2 + 1);
    if (istride <= 0) istride = 1;
    if (ostride <= 0) ostride = 1;
    if (type == CUFFT_R2C) {
        if (idist <= 0) idist = (int)logical_real;
        if (odist <= 0) odist = (int)logical_complex;
    } else if (type == CUFFT_C2R) {
        if (idist <= 0) idist = (int)logical_complex;
        if (odist <= 0) odist = (int)logical_real;
    } else {
        delete entry;
        return CUFFT_STUB_ERROR;
    }

    float* dummy_real = (float*)fftwf_malloc(sizeof(float) * (std::size_t)odist * (std::size_t)batch);
    fftwf_complex* dummy_complex = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (std::size_t)idist * (std::size_t)batch);
    if (dummy_real == NULL || dummy_complex == NULL) {
        if (dummy_real) fftwf_free(dummy_real);
        if (dummy_complex) fftwf_free(dummy_complex);
        delete entry;
        return CUFFT_STUB_ERROR;
    }
    if (type == CUFFT_R2C) {
        entry->plan = fftwf_plan_many_dft_r2c(
            rank, n, batch,
            dummy_real, inembed, istride, idist,
            dummy_complex, onembed, ostride, odist,
            FFTW_ESTIMATE);
    } else {
        entry->plan = fftwf_plan_many_dft_c2r(
            rank, n, batch,
            dummy_complex, inembed, istride, idist,
            dummy_real, onembed, ostride, odist,
            FFTW_ESTIMATE);
    }
    fftwf_free(dummy_real);
    fftwf_free(dummy_complex);
    if (entry->plan == NULL) {
        delete entry;
        return CUFFT_STUB_ERROR;
    }
    return cpu_cufft_plan_store(handle, entry);
#else
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
    cpu_cuda_stub_abort("cufftPlanMany");
    return CUFFT_STUB_ERROR;
#else
    return CUFFT_SUCCESS;
#endif
#endif
}

static inline cufftResult cufftPlan3d(cufftHandle* handle, int nx, int ny, int nz, int type) {
#if CPU_CUDA_COMPAT_HAVE_FFTW3
    if (handle == NULL || nx <= 0 || ny <= 0 || nz <= 0) {
        return CUFFT_STUB_ERROR;
    }
    cpu_cufft_plan_entry* entry = new cpu_cufft_plan_entry();
    entry->type = type;
    entry->rank = 3;
    entry->dims[0] = nx;
    entry->dims[1] = ny;
    entry->dims[2] = nz;
    entry->batch = 1;
    entry->plan = NULL;
    const std::size_t real_count = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
    const std::size_t complex_count = (std::size_t)nx * (std::size_t)ny * (std::size_t)(nz / 2 + 1);
    float* dummy_real = (float*)fftwf_malloc(sizeof(float) * real_count);
    fftwf_complex* dummy_complex = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * complex_count);
    if (dummy_real == NULL || dummy_complex == NULL) {
        if (dummy_real) fftwf_free(dummy_real);
        if (dummy_complex) fftwf_free(dummy_complex);
        delete entry;
        return CUFFT_STUB_ERROR;
    }
    if (type == CUFFT_R2C) {
        entry->plan = fftwf_plan_dft_r2c_3d(nx, ny, nz, dummy_real, dummy_complex, FFTW_ESTIMATE);
    } else if (type == CUFFT_C2R) {
        entry->plan = fftwf_plan_dft_c2r_3d(nx, ny, nz, dummy_complex, dummy_real, FFTW_ESTIMATE);
    }
    fftwf_free(dummy_real);
    fftwf_free(dummy_complex);
    if (entry->plan == NULL) {
        delete entry;
        return CUFFT_STUB_ERROR;
    }
    return cpu_cufft_plan_store(handle, entry);
#else
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
    cpu_cuda_stub_abort("cufftPlan3d");
    return CUFFT_STUB_ERROR;
#else
    return CUFFT_SUCCESS;
#endif
#endif
}

static inline cufftResult cufftExecR2C(cufftHandle handle, float* in, cufftComplex* out) {
#if CPU_CUDA_COMPAT_HAVE_FFTW3
    cpu_cufft_plan_entry* entry = cpu_cufft_plan_lookup(handle);
    if (entry == NULL || entry->plan == NULL || entry->type != CUFFT_R2C || in == NULL || out == NULL) {
        return CUFFT_STUB_ERROR;
    }
    fftwf_execute_dft_r2c(entry->plan, in, reinterpret_cast<fftwf_complex*>(out));
    return CUFFT_SUCCESS;
#else
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
    cpu_cuda_stub_abort("cufftExecR2C");
    return CUFFT_STUB_ERROR;
#else
    return CUFFT_SUCCESS;
#endif
#endif
}

static inline cufftResult cufftExecC2R(cufftHandle handle, cufftComplex* in, float* out) {
#if CPU_CUDA_COMPAT_HAVE_FFTW3
    cpu_cufft_plan_entry* entry = cpu_cufft_plan_lookup(handle);
    if (entry == NULL || entry->plan == NULL || entry->type != CUFFT_C2R || in == NULL || out == NULL) {
        return CUFFT_STUB_ERROR;
    }
    fftwf_execute_dft_c2r(entry->plan, reinterpret_cast<fftwf_complex*>(in), out);
    return CUFFT_SUCCESS;
#else
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
    cpu_cuda_stub_abort("cufftExecC2R");
    return CUFFT_STUB_ERROR;
#else
    return CUFFT_SUCCESS;
#endif
#endif
}

static inline cufftResult cufftDestroy(cufftHandle handle) {
#if CPU_CUDA_COMPAT_HAVE_FFTW3
    std::lock_guard<std::mutex> lock(cpu_cufft_plan_mutex());
    auto& table = cpu_cufft_plan_table();
    auto it = table.find(handle);
    if (it == table.end()) {
        return CUFFT_STUB_ERROR;
    }
    cpu_cufft_plan_entry* entry = it->second;
    if (entry != NULL) {
        if (entry->plan != NULL) {
            fftwf_destroy_plan(entry->plan);
        }
        delete entry;
    }
    table.erase(it);
    return CUFFT_SUCCESS;
#else
#if CPU_CUDA_COMPAT_ABORT_ON_STUB_USE
    cpu_cuda_stub_abort("cufftDestroy");
    return CUFFT_STUB_ERROR;
#else
    return CUFFT_SUCCESS;
#endif
#endif
}
