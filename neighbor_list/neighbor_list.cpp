#include "neighbor_list.h"
#include <climits>
#include <cstdint>
#include <cstdlib>

#ifdef SPONGE_CPU_PHASE1
static bool Cpu_Debug_Env_On(const char* name)
{
    const char* value = std::getenv(name);
    return value != NULL && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void Cpu_Debug_Print_NL_Bucket_Stats(
    const GRID_INFORMATION& grid_info,
    int atom_numbers,
    int max_atom_in_grid_numbers)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_NL_BUCKET_STATS"))
    {
        return;
    }
    const int grid_numbers = grid_info.grid_numbers;
    if (grid_numbers <= 0 || grid_info.atom_numbers_in_grid_bucket == NULL)
    {
        return;
    }
    int* h_counts = (int*)malloc(sizeof(int) * grid_numbers);
    if (h_counts == NULL)
    {
        return;
    }
    cudaMemcpy(h_counts, grid_info.atom_numbers_in_grid_bucket, sizeof(int) * grid_numbers, cudaMemcpyDeviceToHost);
    long long total = 0;
    int nonzero = 0;
    int max_count = -1;
    int max_grid = -1;
    int ge_limit = 0;
    for (int i = 0; i < grid_numbers; ++i)
    {
        const int c = h_counts[i];
        total += c;
        if (c > 0)
        {
            ++nonzero;
        }
        if (c > max_count)
        {
            max_count = c;
            max_grid = i;
        }
        if (c >= max_atom_in_grid_numbers)
        {
            ++ge_limit;
        }
    }
    std::fprintf(
        stderr,
        "CPU_DEBUG_NL total=%lld expected=%d nonzero=%d max=%d max_grid=%d ge_limit=%d limit=%d\n",
        total,
        atom_numbers,
        nonzero,
        max_count,
        max_grid,
        ge_limit,
        max_atom_in_grid_numbers);
    free(h_counts);
}

static inline bool Cpu_Is_Aligned_Int_Pointer(const int* ptr)
{
    if (ptr == NULL)
    {
        return false;
    }
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    if (addr < 4096u)
    {
        return false;
    }
    return (addr % alignof(int)) == 0u;
}

static void Cpu_Debug_Print_GridSerial_Stats(
    const int* atom_in_grid_serial,
    int atom_numbers,
    int grid_numbers)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_NL_GRID_SERIAL"))
    {
        return;
    }
    if (atom_in_grid_serial == NULL || atom_numbers <= 0)
    {
        return;
    }
    int* h_serial = (int*)malloc(sizeof(int) * atom_numbers);
    if (h_serial == NULL)
    {
        return;
    }
    cudaMemcpy(h_serial, atom_in_grid_serial, sizeof(int) * atom_numbers, cudaMemcpyDeviceToHost);
    int min_v = INT_MAX;
    int max_v = INT_MIN;
    int invalid = 0;
    for (int i = 0; i < atom_numbers; ++i)
    {
        const int v = h_serial[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v < 0 || v >= grid_numbers)
        {
            ++invalid;
        }
    }
    std::fprintf(
        stderr,
        "CPU_DEBUG_NL_GRID min=%d max=%d invalid=%d atom_numbers=%d grid_numbers=%d\n",
        min_v,
        max_v,
        invalid,
        atom_numbers,
        grid_numbers);
    free(h_serial);
}

static void Cpu_Debug_Print_NL_List_Stats(
    const ATOM_GROUP* d_nl,
    int atom_numbers,
    int max_neighbor_numbers)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_NL_LIST_STATS"))
    {
        return;
    }
    if (d_nl == NULL || atom_numbers <= 0)
    {
        return;
    }
    ATOM_GROUP* h_nl = (ATOM_GROUP*)malloc(sizeof(ATOM_GROUP) * atom_numbers);
    if (h_nl == NULL)
    {
        return;
    }
    cudaMemcpy(h_nl, d_nl, sizeof(ATOM_GROUP) * atom_numbers, cudaMemcpyDeviceToHost);
    long long total_neighbors = 0;
    int max_neighbors = -1;
    int max_neighbors_atom = -1;
    int invalid_neighbor_count = 0;
    int overflow_count = 0;
    for (int atom_i = 0; atom_i < atom_numbers; ++atom_i)
    {
        int n = h_nl[atom_i].atom_numbers;
        if (n < 0 || n > max_neighbor_numbers)
        {
            ++overflow_count;
            if (n < 0) n = 0;
            if (n > max_neighbor_numbers) n = max_neighbor_numbers;
        }
        total_neighbors += n;
        if (n > max_neighbors)
        {
            max_neighbors = n;
            max_neighbors_atom = atom_i;
        }
        int* serial = h_nl[atom_i].atom_serial;
        if (serial == NULL)
        {
            invalid_neighbor_count += n;
            continue;
        }
        for (int k = 0; k < n; ++k)
        {
            const int atom_j = serial[k];
            if (atom_j < 0 || atom_j >= atom_numbers)
            {
                ++invalid_neighbor_count;
            }
        }
    }
    std::fprintf(
        stderr,
        "CPU_DEBUG_NL_LIST total_neighbors=%lld max_neighbors=%d max_neighbors_atom=%d invalid_neighbor_ids=%d overflow_count=%d\n",
        total_neighbors,
        max_neighbors,
        max_neighbors_atom,
        invalid_neighbor_count,
        overflow_count);
    free(h_nl);
}

static void Cpu_Debug_Print_NL_Excluded_Residual(
    const ATOM_GROUP* d_nl,
    int atom_numbers,
    int max_neighbor_numbers,
    const int* d_excluded_list_start,
    const int* d_excluded_list,
    const int* d_excluded_atom_numbers)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_NL_EXCLUDED_RESIDUAL"))
    {
        return;
    }
    if (d_nl == NULL ||
        atom_numbers <= 0 ||
        max_neighbor_numbers <= 0 ||
        d_excluded_list_start == NULL ||
        d_excluded_list == NULL ||
        d_excluded_atom_numbers == NULL)
    {
        return;
    }

    ATOM_GROUP* h_nl = (ATOM_GROUP*)malloc(sizeof(ATOM_GROUP) * atom_numbers);
    int* h_excluded_list_start = (int*)malloc(sizeof(int) * atom_numbers);
    int* h_excluded_atom_numbers = (int*)malloc(sizeof(int) * atom_numbers);
    if (h_nl == NULL || h_excluded_list_start == NULL || h_excluded_atom_numbers == NULL)
    {
        free(h_nl);
        free(h_excluded_list_start);
        free(h_excluded_atom_numbers);
        return;
    }

    cudaMemcpy(h_nl, d_nl, sizeof(ATOM_GROUP) * atom_numbers, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_excluded_list_start, d_excluded_list_start, sizeof(int) * atom_numbers, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_excluded_atom_numbers, d_excluded_atom_numbers, sizeof(int) * atom_numbers, cudaMemcpyDeviceToHost);

    int excluded_total = 0;
    for (int atom_i = 0; atom_i < atom_numbers; ++atom_i)
    {
        const int start = h_excluded_list_start[atom_i];
        const int count = h_excluded_atom_numbers[atom_i];
        const int end = start + count;
        if (end > excluded_total)
        {
            excluded_total = end;
        }
    }
    if (excluded_total <= 0)
    {
        std::fprintf(stderr, "CPU_DEBUG_NL_EXCLUDED residual=0 self_neighbors=0 (no excluded entries)\n");
        free(h_nl);
        free(h_excluded_list_start);
        free(h_excluded_atom_numbers);
        return;
    }

    int* h_excluded_list = (int*)malloc(sizeof(int) * excluded_total);
    if (h_excluded_list == NULL)
    {
        free(h_nl);
        free(h_excluded_list_start);
        free(h_excluded_atom_numbers);
        return;
    }
    cudaMemcpy(h_excluded_list, d_excluded_list, sizeof(int) * excluded_total, cudaMemcpyDeviceToHost);

    long long residual_excluded_pairs = 0;
    long long self_neighbors = 0;
    int worst_atom = -1;
    int worst_count = -1;
    for (int atom_i = 0; atom_i < atom_numbers; ++atom_i)
    {
        int atom_residual = 0;
        const int start = h_excluded_list_start[atom_i];
        const int count = h_excluded_atom_numbers[atom_i];
        const int end = start + count;
        int n = h_nl[atom_i].atom_numbers;
        if (n < 0) n = 0;
        if (n > max_neighbor_numbers) n = max_neighbor_numbers;
        int* serial = h_nl[atom_i].atom_serial;
        if (serial == NULL)
        {
            continue;
        }
        for (int k = 0; k < n; ++k)
        {
            const int atom_j = serial[k];
            if (atom_j == atom_i)
            {
                ++self_neighbors;
            }
            for (int p = start; p < end; ++p)
            {
                if (atom_j == h_excluded_list[p])
                {
                    ++residual_excluded_pairs;
                    ++atom_residual;
                    break;
                }
            }
        }
        if (atom_residual > worst_count)
        {
            worst_count = atom_residual;
            worst_atom = atom_i;
        }
    }
    std::fprintf(
        stderr,
        "CPU_DEBUG_NL_EXCLUDED residual=%lld self_neighbors=%lld worst_atom=%d worst_count=%d excluded_total=%d\n",
        residual_excluded_pairs,
        self_neighbors,
        worst_atom,
        worst_count,
        excluded_total);

    free(h_excluded_list);
    free(h_nl);
    free(h_excluded_list_start);
    free(h_excluded_atom_numbers);
}
#endif

static void Initial_Neighbor_Grid(
    GRID_POINTER **gpointer, GRID_BUCKET **bucket, int **atom_numbers_in_grid_bucket,
    float half_cutoff_with_skin, GRID_INFORMATION *grid_info,
    const int in_bucket_atom_numbers_max, VECTOR box_length)
{


    float half_cutoff = half_cutoff_with_skin;

    grid_info[0].Nx = floorf(box_length.x / half_cutoff);
    grid_info[0].Ny = floorf(box_length.y / half_cutoff);
    grid_info[0].Nz = floorf(box_length.z / half_cutoff);
    grid_info[0].grid_N = { grid_info[0].Nx, grid_info[0].Ny, grid_info[0].Nz };

    grid_info[0].grid_length.x = (float)box_length.x / grid_info[0].Nx;
    grid_info[0].grid_length_inverse.x = 1. / grid_info[0].grid_length.x;
    grid_info[0].grid_length.y = (float)box_length.y / grid_info[0].Ny;
    grid_info[0].grid_length_inverse.y = 1. / grid_info[0].grid_length.y;
    grid_info[0].grid_length.z = (float)box_length.z / grid_info[0].Nz;
    grid_info[0].grid_length_inverse.z = 1. / grid_info[0].grid_length.z;

    grid_info[0].Nxy = grid_info[0].Nx*grid_info[0].Ny;
    grid_info[0].grid_numbers = grid_info[0].Nz*grid_info[0].Nxy;

    Cuda_Malloc_Safely((void **)&atom_numbers_in_grid_bucket[0], sizeof(int)*(grid_info[0].grid_numbers+1));
    CUDA_LAUNCH(Reset_List,
        dim3(static_cast<unsigned int>(ceilf(((float)grid_info[0].grid_numbers + 1) / 32))),
        dim3(32),
        grid_info[0].grid_numbers + 1,
        atom_numbers_in_grid_bucket[0],
        0);

    Malloc_Safely((void**)&grid_info[0].h_bucket,sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1)); 
    for (int i = 0; i < grid_info[0].grid_numbers + 1; i = i + 1)
    {
        Cuda_Malloc_Safely((void**)&grid_info[0].h_bucket[i].atom_serial, sizeof(int)* in_bucket_atom_numbers_max);
        CUDA_LAUNCH(Reset_List,
            dim3(static_cast<unsigned int>(ceilf((float)in_bucket_atom_numbers_max / 32))),
            dim3(32),
            in_bucket_atom_numbers_max,
            grid_info[0].h_bucket[i].atom_serial,
            -1);
    }
    Cuda_Malloc_Safely((void**)&bucket[0], sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1));
    cudaMemcpy(bucket[0], grid_info[0].h_bucket, sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1), cudaMemcpyHostToDevice);
    //free(h_bucket);

    GRID_POINTER lin_pointer;
    lin_pointer.grid_serial = (int*)malloc(sizeof(int)* 125);
    int Nx;
    int Ny;
    int Nz;
    int xx;
    int yy;
    int zz;
    int count;
    int small_out;
    Malloc_Safely((void**)&grid_info[0].h_pointer, sizeof(GRID_POINTER)*grid_info[0].grid_numbers);
    for (int i = 0; i < grid_info[0].grid_numbers; i = i + 1)
    {
        Nz = i / grid_info[0].Nxy;
        Ny = (i - grid_info[0].Nxy*Nz) / grid_info[0].Nx;
        Nx = i - grid_info[0].Nxy*Nz - grid_info[0].Nx*Ny;
        count = 0;
        for (int l = -2; l <= 2; l = l + 1)
        {
            for (int m = -2; m <= 2; m = m + 1)
            {
                for (int n = -2; n <= 2; n = n + 1)
                {
                    small_out = 0;
                    xx = Nx + l;
                    //处理小盒子越边界
                    //盒子大小大于5、未超过边界、等于4且只超出一个边界的，不处理
                    if (grid_info->Nx >= 5 || (xx >= 0 && xx < grid_info->Nx) || (grid_info->Nx == 4 && ((Nx == 0 && l == -1) || (Nx == 3 && l == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (xx < 0)
                        {
                            xx = xx + grid_info[0].Nx;
                        }
                        else if (xx >= grid_info[0].Nx)
                        {
                            xx = xx - grid_info[0].Nx;
                        }
                    }
                    yy = Ny + m;
                    //处理小盒子越边界
                    if (grid_info->Ny >= 5 || (yy >= 0 && yy < grid_info->Ny) || (grid_info->Ny == 4 && ((Ny == 0 && m == -1) || (Ny == 3 && m == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (yy < 0)
                        {
                            yy = yy + grid_info[0].Ny;
                        }
                        else if (yy >= grid_info[0].Ny)
                        {
                            yy = yy - grid_info[0].Ny;
                        }
                    }
                    
                    zz = Nz + n;
                    //处理小盒子越边界
                    if (grid_info->Nz >= 5 || (zz >= 0 && zz < grid_info->Nz) || (grid_info->Nz == 4 && ((Nz == 0 && n == -1) || (Nz == 3 && n == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (zz < 0)
                        {
                            zz = zz + grid_info[0].Nz;
                        }
                        else if (zz >= grid_info[0].Nz)
                        {
                            zz = zz - grid_info[0].Nz;
                        }
                    }

                    if (!small_out)
                    {
                        lin_pointer.grid_serial[count] = zz*grid_info[0].Nxy + yy*grid_info[0].Nx + xx;
                        
                    }
                    else
                    {
                        lin_pointer.grid_serial[count] = grid_info->grid_numbers;
                    }
                    count = count + 1;
                }
            }
        }//for l m n
        std::sort(&lin_pointer.grid_serial[0], lin_pointer.grid_serial + 125);
        Cuda_Malloc_Safely((void**)&grid_info[0].h_pointer[i].grid_serial, sizeof(int)* 125);//5*5*5
        cudaMemcpy(grid_info[0].h_pointer[i].grid_serial, lin_pointer.grid_serial, sizeof(int)* 125, cudaMemcpyHostToDevice);
    }
    Cuda_Malloc_Safely((void**)&gpointer[0], sizeof(GRID_POINTER)*grid_info[0].grid_numbers);
    
    cudaMemcpy(gpointer[0], grid_info[0].h_pointer, sizeof(GRID_POINTER)*grid_info[0].grid_numbers, cudaMemcpyHostToDevice);
}

static __global__ void Clear_Grid_Bucket(
    const int grid_numbers,
    int *atom_numbers_in_grid_bucket,
    GRID_BUCKET *bucket,
    const int max_atom_in_grid_numbers)
{
    int grid_serial = blockDim.x*blockIdx.x + threadIdx.x;
    if (grid_serial < grid_numbers)
    {
        int clear_count = atom_numbers_in_grid_bucket[grid_serial];
        if (clear_count < 0)
        {
            clear_count = 0;
        }
        else if (clear_count > max_atom_in_grid_numbers)
        {
            clear_count = max_atom_in_grid_numbers;
        }

        int *atom_serial = bucket[grid_serial].atom_serial;
#ifdef SPONGE_CPU_PHASE1
        if (!Cpu_Is_Aligned_Int_Pointer(atom_serial))
        {
            atom_numbers_in_grid_bucket[grid_serial] = 0;
            return;
        }
#else
        if (atom_serial == NULL)
        {
            atom_numbers_in_grid_bucket[grid_serial] = 0;
            return;
        }
#endif
        for (int i = 0; i < clear_count; i = i + 1)
        {
            atom_serial[i] = -1;
        }
        atom_numbers_in_grid_bucket[grid_serial] = 0;
    }
}

static __global__ void Find_Atom_In_Grid_Serial(const int atom_numbers, const VECTOR grid_length_inverse, const VECTOR *crd, const INT_VECTOR grid_N, const int gridxy, int *atom_in_grid_serial)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR local_crd = crd[atom_i];
        int Nx = floorf(local_crd.x*grid_length_inverse.x);
        int Ny = floorf(local_crd.y*grid_length_inverse.y);
        int Nz = floorf(local_crd.z*grid_length_inverse.z);
        Nx = Nx % grid_N.int_x;
        Nx += (Nx < 0) * grid_N.int_x;
        Ny = Ny % grid_N.int_y;
        Ny += (Ny < 0) * grid_N.int_y;
        Nz = Nz % grid_N.int_z;
        Nz += (Nz < 0) * grid_N.int_z;
        atom_in_grid_serial[atom_i] = Nz*gridxy + Ny*grid_N.int_x + Nx;
    }
}
static __global__ void Put_Atom_In_Grid_Bucket(const int atom_numbers, const int *atom_in_grid_serial, GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket,
    const int grid_numbers, const int max_atom_in_grid_numbers, int *exceed_max_grid_numbers)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {

        int grid_serial = atom_in_grid_serial[atom_i];
#ifdef SPONGE_CPU_PHASE1
        if (grid_serial < 0 || grid_serial >= grid_numbers)
        {
            exceed_max_grid_numbers[0] = 1;
            return;
        }
        int a = atom_numbers_in_grid_bucket[grid_serial];
        if (a >= max_atom_in_grid_numbers)
        {
            exceed_max_grid_numbers[0] = 1;
            return;
        }
        bucket[grid_serial].atom_serial[a] = atom_i;
        atom_numbers_in_grid_bucket[grid_serial] = a + 1;
#else
        if (grid_serial < 0 || grid_serial >= grid_numbers)
        {
            exceed_max_grid_numbers[0] = 1;
            return;
        }
        GRID_BUCKET bucket_i = bucket[grid_serial];
        int a = atom_numbers_in_grid_bucket[grid_serial];
        if (a >= max_atom_in_grid_numbers)
        {
            exceed_max_grid_numbers[0] = 1;
            return;
        }
        atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
        if (bucket_i.atom_serial[a] != atom_i)
        {

            while (true)
            {
                a = a + 1;
                if (a >= max_atom_in_grid_numbers)
                {
                    exceed_max_grid_numbers[0] = 1;
                    break;
                }
                atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
                if (bucket_i.atom_serial[a] == atom_i)
                {
                    if (atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1) > max_atom_in_grid_numbers)
                    {
                        exceed_max_grid_numbers[0] = 1;
                    }
                    break;
                }
            }

        }
        else
        {
            if (atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1) > max_atom_in_grid_numbers)
            {
                exceed_max_grid_numbers[0] = 1;
            }
        }
#endif
    }
}

static __global__ void Find_atom_neighbors_gridly(
    const int atom_numbers, const VECTOR *crd, const VECTOR box_length, const VECTOR box_length_inverse,
    const int *atom_in_grid_serial, const GRID_POINTER *gpointer, const GRID_BUCKET *bucket, const int *atom_numbers_in_grid_bucket,
    ATOM_GROUP *nl, const float cutoff_skin_square, const int max_atom_numbers_in_gird, const int max_neighbor_numbers, int *exceed_max_neighbors)
{
    if (threadIdx.y < 125)
    {
        int grid_i = blockIdx.x;
        int grid_j = gpointer[grid_i].grid_serial[threadIdx.y];
        int atom_i, atom_j;
        VECTOR dr;
        float dr2;
        extern __shared__ char shared_memory[];

        VECTOR *sm_crd = (VECTOR *)shared_memory;
        int *sm_bucket_i = (int*)(shared_memory + sizeof(UNSIGNED_INT_VECTOR)* max_atom_numbers_in_gird);
        ATOM_GROUP *sm_nl = (ATOM_GROUP *)(shared_memory + (sizeof(UNSIGNED_INT_VECTOR)+sizeof(int))* max_atom_numbers_in_gird);

        int *bucket_i = bucket[grid_i].atom_serial;
        int *bucket_j = bucket[grid_j].atom_serial;
        int atom_numbers_in_grid_i = atom_numbers_in_grid_bucket[grid_i];
        int atom_numbers_in_grid_j = atom_numbers_in_grid_bucket[grid_j];
        if (threadIdx.x == 0)
        {
            for (int i = threadIdx.y; i < atom_numbers_in_grid_i; i += blockDim.y)
            {
                atom_i = bucket_i[i];
                sm_crd[i] = crd[atom_i];
                sm_bucket_i[i] = atom_i;        
                sm_nl[i] = nl[atom_i];
                nl[atom_i].atom_numbers = 0;
            }
        }
        __syncthreads();
        VECTOR crd_j;
        ATOM_GROUP nl_i;
        for (int j = threadIdx.x; j < atom_numbers_in_grid_j; j += blockDim.x)
        {
            atom_j = bucket_j[j];
            crd_j = crd[atom_j];

            for (int i = 0; i < atom_numbers_in_grid_i; i++)
            {
                atom_i = sm_bucket_i[i];
                nl_i = sm_nl[i];

                if (atom_j > atom_i)
                {            
                    dr = Get_Periodic_Displacement(crd_j, sm_crd[i], box_length, box_length_inverse);
                    dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                    if (dr2 < cutoff_skin_square)
                    {
                        nl_i.atom_numbers = atomicAdd(&nl[atom_i].atom_numbers, 1);
                        if (nl_i.atom_numbers > max_neighbor_numbers)
                        {
                            exceed_max_neighbors[0] = 1;
                        }
                        nl_i.atom_serial[nl_i.atom_numbers] = atom_j;
                    }
                }
            }
        }
    }
}

#ifdef SPONGE_CPU_PHASE1
static __global__ void Find_atom_neighbors_gridly_cpu_serial(
    const VECTOR *crd, const VECTOR box_length, const VECTOR box_length_inverse,
    const GRID_POINTER *gpointer, const GRID_BUCKET *bucket, const int *atom_numbers_in_grid_bucket,
    ATOM_GROUP *nl, const float cutoff_skin_square, const int max_neighbor_numbers, int *exceed_max_neighbors)
{
    if (!(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0))
    {
        return;
    }
    const int grid_i = blockIdx.x;
    const int atom_numbers_in_grid_i = atom_numbers_in_grid_bucket[grid_i];
    int* bucket_i = bucket[grid_i].atom_serial;
    for (int i = 0; i < atom_numbers_in_grid_i; ++i)
    {
        const int atom_i = bucket_i[i];
        nl[atom_i].atom_numbers = 0;
    }
    for (int neighbor_grid_idx = 0; neighbor_grid_idx < 125; ++neighbor_grid_idx)
    {
        const int grid_j = gpointer[grid_i].grid_serial[neighbor_grid_idx];
        const int atom_numbers_in_grid_j = atom_numbers_in_grid_bucket[grid_j];
        int* bucket_j = bucket[grid_j].atom_serial;
        for (int j = 0; j < atom_numbers_in_grid_j; ++j)
        {
            const int atom_j = bucket_j[j];
            const VECTOR crd_j = crd[atom_j];
            for (int i = 0; i < atom_numbers_in_grid_i; ++i)
            {
                const int atom_i = bucket_i[i];
                if (atom_j <= atom_i)
                {
                    continue;
                }
                VECTOR dr = Get_Periodic_Displacement(crd_j, crd[atom_i], box_length, box_length_inverse);
                const float dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                if (dr2 < cutoff_skin_square)
                {
                    const int write_index = nl[atom_i].atom_numbers;
                    if (write_index >= max_neighbor_numbers)
                    {
                        exceed_max_neighbors[0] = 1;
                        continue;
                    }
                    nl[atom_i].atom_serial[write_index] = atom_j;
                    nl[atom_i].atom_numbers = write_index + 1;
                }
            }
        }
    }
}
#endif

static __global__ void Is_need_refresh_neighbor_list_cuda(const int atom_numbers,const VECTOR *crd, const VECTOR *old_crd,
    const VECTOR box_length, const float half_skin_square,int *need_refresh_flag)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        VECTOR r1 = crd[i];
        VECTOR r2 = old_crd[i];
        r1 = Get_Periodic_Displacement(r1, r2, box_length);
        float r1_2 = r1.x*r1.x + r1.y*r1.y + r1.z*r1.z;
        if (r1_2>half_skin_square)
        {
            atomicExch(&need_refresh_flag[0], 1);
        }
    }
}

static __global__ void Delete_Excluded_Atoms_Serial_In_Neighbor_List
(const int atom_numbers, ATOM_GROUP *nl, const int *excluded_list_start,const int *excluded_list,const int *excluded_atom_numbers)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int excluded_number = excluded_atom_numbers[atom_i];
        if (excluded_number > 0 )
        {
            int list_start = excluded_list_start[atom_i];
            int atom_min = excluded_list[list_start];
            int list_end = list_start + excluded_number;
            int atom_max = excluded_list[list_end - 1];
            ATOM_GROUP nl_i = nl[atom_i];
            int atomnumbers_in_nl_lin = nl_i.atom_numbers;
            int atom_j;
            int excluded_atom_numbers_lin = list_end-list_start;
            int excluded_atom_numbers_count = 0;
            for (int i = 0; i < atomnumbers_in_nl_lin; i = i + 1)
            {
                atom_j = nl_i.atom_serial[i];
                if (atom_j<atom_min || atom_j>atom_max)
                {
                    continue;
                }
                else
                {
                    for (int j = list_start; j < list_end; j = j + 1)
                    {
                        if (atom_j == excluded_list[j])
                        {
                            atomnumbers_in_nl_lin = atomnumbers_in_nl_lin - 1;
                            nl_i.atom_serial[i] = nl_i.atom_serial[atomnumbers_in_nl_lin];
                            excluded_atom_numbers_count = excluded_atom_numbers_count + 1;
                            i=i-1;
                        }
                    }
                    if (excluded_atom_numbers_count < excluded_atom_numbers_lin)
                    {

                    }
                    else
                    {
                        break;
                    }//break
                }//in the range of excluded min to max
            }//cycle for neighbors
            nl[atom_i].atom_numbers = atomnumbers_in_nl_lin;
        }//if need excluded
    }
}

static __global__ void Refresh_Neighbor_List
(int *refresh_sign, const int thread,
const int atom_numbers, VECTOR *crd, VECTOR *old_crd,
int *atom_in_grid_serial,
const float skin, const VECTOR box_length,
const GRID_INFORMATION grid_info, const GRID_POINTER *gpointer,
GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket,
ATOM_GROUP *d_nl, int *excluded_list_start, int * excluded_list, int * excluded_numbers, float cutoff_skin_square, 
const int max_atom_in_grid_numbers, const int max_neighbor_numbers, int* d_exceed_max_grid_numbers, int* d_exceed_max_neighbor_atoms)
{
    if (refresh_sign[0])
    {

        CUDA_LAUNCH(Clear_Grid_Bucket,
            dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers / thread))),
            dim3(static_cast<unsigned int>(thread)),
            grid_info.grid_numbers,
            atom_numbers_in_grid_bucket,
            bucket,
            max_atom_in_grid_numbers);

        CUDA_LAUNCH(Find_Atom_In_Grid_Serial,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / thread))),
            dim3(static_cast<unsigned int>(thread)),
            atom_numbers,
            grid_info.grid_length_inverse,
            crd,
            grid_info.grid_N,
            grid_info.Nxy,
            atom_in_grid_serial);

        CUDA_LAUNCH(Copy_List,
            dim3(static_cast<unsigned int>(ceilf((float)3. * atom_numbers / thread))),
            dim3(static_cast<unsigned int>(thread)),
            3 * atom_numbers,
            (float*)crd,
            (float*)old_crd);

        CUDA_LAUNCH(Put_Atom_In_Grid_Bucket,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / thread))),
            dim3(static_cast<unsigned int>(thread)),
            atom_numbers,
            atom_in_grid_serial,
            bucket,
            atom_numbers_in_grid_bucket,
            grid_info.grid_numbers,
            max_atom_in_grid_numbers,
            d_exceed_max_grid_numbers);

#ifdef SPONGE_CPU_PHASE1
        CUDA_LAUNCH(Find_atom_neighbors_gridly_cpu_serial,
            dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers))),
            dim3(1, 1, 1),
            crd,
            box_length,
            1.0f / box_length,
            gpointer,
            bucket,
            atom_numbers_in_grid_bucket,
            d_nl,
            cutoff_skin_square,
            max_neighbor_numbers,
            d_exceed_max_neighbor_atoms);
#else
        CUDA_LAUNCH_SHARED(Find_atom_neighbors_gridly,
            dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers))),
            dim3(8, 128),
            (sizeof(int) + sizeof(UNSIGNED_INT_VECTOR) + sizeof(ATOM_GROUP)) * max_atom_in_grid_numbers,
            atom_numbers,
            crd,
            box_length,
            1.0f / box_length,
            atom_in_grid_serial,
            gpointer,
            bucket,
            atom_numbers_in_grid_bucket,
            d_nl,
            cutoff_skin_square,
            max_atom_in_grid_numbers,
            max_neighbor_numbers,
            d_exceed_max_neighbor_atoms);
#endif

        CUDA_LAUNCH(Delete_Excluded_Atoms_Serial_In_Neighbor_List,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / thread))),
            dim3(static_cast<unsigned int>(thread)),
            atom_numbers,
            d_nl,
            excluded_list_start,
            excluded_list,
            excluded_numbers);

    }
}


static void Refresh_Neighbor_List_No_Check
(const int atom_numbers, VECTOR *crd, VECTOR *old_crd,
int *atom_in_grid_serial,
const float skin, const VECTOR box_length,
const GRID_INFORMATION grid_info, const GRID_POINTER *gpointer,
GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket,
ATOM_GROUP *d_nl, int *excluded_list_start, int * excluded_list, int * excluded_numbers, float cutoff_skin_square, 
const int max_atom_in_grid_numbers, const int max_neighbor_numbers, int* d_exceed_max_grid_numbers, int* d_exceed_max_neighbor_numbers)
{
#ifdef SPONGE_CPU_PHASE1
    // In CPU compatibility mode, refresh invalid bucket pointers from the canonical host table.
    if (grid_info.bucket != NULL && grid_info.h_bucket != NULL)
    {
        for (int grid_serial = 0; grid_serial <= grid_info.grid_numbers; ++grid_serial)
        {
            if (!Cpu_Is_Aligned_Int_Pointer(grid_info.bucket[grid_serial].atom_serial))
            {
                grid_info.bucket[grid_serial].atom_serial = grid_info.h_bucket[grid_serial].atom_serial;
            }
        }
    }
#endif
    CUDA_LAUNCH(Clear_Grid_Bucket,
        dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers / 32))),
        dim3(32),
        grid_info.grid_numbers,
        grid_info.atom_numbers_in_grid_bucket,
        grid_info.bucket,
        max_atom_in_grid_numbers);

    CUDA_LAUNCH(Find_Atom_In_Grid_Serial,
        dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 32))),
        dim3(32),
        atom_numbers,
        grid_info.grid_length_inverse,
        crd,
        grid_info.grid_N,
        grid_info.Nxy,
        grid_info.atom_in_grid_serial);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Print_GridSerial_Stats(atom_in_grid_serial, atom_numbers, grid_info.grid_numbers);
#endif

    cudaMemcpy(old_crd, crd, sizeof(VECTOR)*atom_numbers, cudaMemcpyDeviceToDevice);

    CUDA_LAUNCH(Put_Atom_In_Grid_Bucket,
        dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 32))),
        dim3(32),
        atom_numbers,
        atom_in_grid_serial,
        bucket,
        atom_numbers_in_grid_bucket,
        grid_info.grid_numbers,
        max_atom_in_grid_numbers,
        d_exceed_max_grid_numbers);

#ifdef SPONGE_CPU_PHASE1
    CUDA_LAUNCH(Find_atom_neighbors_gridly_cpu_serial,
        dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers))),
        dim3(1, 1, 1),
        crd,
        box_length,
        1.0f / box_length,
        gpointer,
        bucket,
        atom_numbers_in_grid_bucket,
        d_nl,
        cutoff_skin_square,
        max_neighbor_numbers,
        d_exceed_max_neighbor_numbers);
#else
    CUDA_LAUNCH_SHARED(Find_atom_neighbors_gridly,
        dim3(static_cast<unsigned int>(ceilf((float)grid_info.grid_numbers))),
        dim3(8, 128),
        (sizeof(int) + sizeof(UNSIGNED_INT_VECTOR) + sizeof(ATOM_GROUP)) * max_atom_in_grid_numbers,
        atom_numbers,
        crd,
        box_length,
        1.0f / box_length,
        atom_in_grid_serial,
        gpointer,
        bucket,
        atom_numbers_in_grid_bucket,
        d_nl,
        cutoff_skin_square,
        max_atom_in_grid_numbers,
        max_neighbor_numbers,
        d_exceed_max_neighbor_numbers);
#endif
 
    CUDA_LAUNCH(Delete_Excluded_Atoms_Serial_In_Neighbor_List,
        dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 32))),
        dim3(32),
        atom_numbers,
        d_nl,
        excluded_list_start,
        excluded_list,
        excluded_numbers);
}



void NEIGHBOR_LIST::Neighbor_List_Update(VECTOR *crd, int *d_excluded_list_start, int *d_excluded_list, int *d_excluded_numbers,
    int forced_update, int forced_check)
{
    if (is_initialized)
    {    
        if (forced_update) //如果强制要求更新就强制更新
        {
            cudaMemset(is_need_refresh_neighbor_list, -1, sizeof(float));
            Refresh_Neighbor_List_No_Check
                (atom_numbers, crd, old_crd,
                grid_info.atom_in_grid_serial,
                skin, box_length,
                grid_info, grid_info.gpointer,
                grid_info.bucket, grid_info.atom_numbers_in_grid_bucket,
                d_nl, d_excluded_list_start, d_excluded_list, d_excluded_numbers, cutoff_with_skin_square, 
                max_atom_in_grid_numbers, max_neighbor_numbers, d_exceed_max_grid_atoms, d_exceed_max_neighbor_atoms);
#ifdef SPONGE_CPU_PHASE1
            Cpu_Debug_Print_NL_Bucket_Stats(grid_info, atom_numbers, max_atom_in_grid_numbers);
            Cpu_Debug_Print_NL_List_Stats(d_nl, atom_numbers, max_neighbor_numbers);
            Cpu_Debug_Print_NL_Excluded_Residual(d_nl, atom_numbers, max_neighbor_numbers, d_excluded_list_start, d_excluded_list, d_excluded_numbers);
#endif
        }
        else if (refresh_interval > 0 && !forced_check) //如果是恒步长更新且不强制要求检查是否更新
        {
            if (refresh_count % refresh_interval == 0)
            {
                cudaMemset(is_need_refresh_neighbor_list, -1, sizeof(float));
                Refresh_Neighbor_List_No_Check
                    (atom_numbers, crd, old_crd,
                    grid_info.atom_in_grid_serial,
                    skin, box_length,
                    grid_info, grid_info.gpointer,
                    grid_info.bucket, grid_info.atom_numbers_in_grid_bucket,
                    d_nl, d_excluded_list_start, d_excluded_list, d_excluded_numbers, cutoff_with_skin_square,
                    max_atom_in_grid_numbers, max_neighbor_numbers, d_exceed_max_grid_atoms, d_exceed_max_neighbor_atoms);
#ifdef SPONGE_CPU_PHASE1
                Cpu_Debug_Print_NL_Bucket_Stats(grid_info, atom_numbers, max_atom_in_grid_numbers);
                Cpu_Debug_Print_NL_List_Stats(d_nl, atom_numbers, max_neighbor_numbers);
                Cpu_Debug_Print_NL_Excluded_Residual(d_nl, atom_numbers, max_neighbor_numbers, d_excluded_list_start, d_excluded_list, d_excluded_numbers);
#endif
            }
            else
            {
                cudaMemset(is_need_refresh_neighbor_list, 0, sizeof(float));
            }
            refresh_count += 1;
        }
        else //其余情况
        {
            cudaMemset(is_need_refresh_neighbor_list, 0, sizeof(int));
            CUDA_LAUNCH(Is_need_refresh_neighbor_list_cuda,
                dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 128))),
                dim3(128),
                atom_numbers,
                crd,
                old_crd,
                box_length,
                skin_permit * skin_permit * half_skin_square,
                is_need_refresh_neighbor_list);
            CUDA_LAUNCH(Refresh_Neighbor_List,
                dim3(1),
                dim3(1),
                is_need_refresh_neighbor_list,
                32,
                atom_numbers,
                crd,
                old_crd,
                grid_info.atom_in_grid_serial,
                skin,
                box_length,
                grid_info,
                grid_info.gpointer,
                grid_info.bucket,
                grid_info.atom_numbers_in_grid_bucket,
                d_nl,
                d_excluded_list_start,
                d_excluded_list,
                d_excluded_numbers,
                cutoff_with_skin_square,
                max_atom_in_grid_numbers,
                max_neighbor_numbers,
                d_exceed_max_grid_atoms,
                d_exceed_max_neighbor_atoms);
#ifdef SPONGE_CPU_PHASE1
            Cpu_Debug_Print_NL_Bucket_Stats(grid_info, atom_numbers, max_atom_in_grid_numbers);
            Cpu_Debug_Print_NL_List_Stats(d_nl, atom_numbers, max_neighbor_numbers);
            Cpu_Debug_Print_NL_Excluded_Residual(d_nl, atom_numbers, max_neighbor_numbers, d_excluded_list_start, d_excluded_list, d_excluded_numbers);
#endif
        }
    }
}

void NEIGHBOR_LIST::Initial_Malloc()
{
    Cuda_Malloc_Safely((void **)&old_crd, sizeof(VECTOR)*atom_numbers);
    Cuda_Malloc_Safely((void **)&uint_crd, sizeof(UNSIGNED_INT_VECTOR)*atom_numbers);
    Malloc_Safely((void**)&h_nl, sizeof(ATOM_GROUP)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_nl, sizeof(ATOM_GROUP)*atom_numbers);
    int* temp;
    Cuda_Malloc_Safely((void**)&temp, sizeof(int) * atom_numbers * max_neighbor_numbers);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        h_nl[i].atom_numbers = 0;
        h_nl[i].atom_serial = temp + i * max_neighbor_numbers;
    }
    cudaMemcpy(d_nl, h_nl, sizeof(ATOM_GROUP)*atom_numbers, cudaMemcpyHostToDevice);
    Cuda_Malloc_Safely((void**)&is_need_refresh_neighbor_list, sizeof(int));
    CUDA_LAUNCH(Reset_List, dim3(1), dim3(1), 1, is_need_refresh_neighbor_list, 0);
    Cuda_Malloc_Safely((void**)&grid_info.atom_in_grid_serial, sizeof(int)*atom_numbers);
}


void NEIGHBOR_LIST::Initial(CONTROLLER *controller, int md_atom_numbers, VECTOR box_length, float cut, float skin, const char * module_name)
{
    
    if (module_name == NULL)
    {
        strcpy(this->module_name, "neighbor_list");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    /*===========================
    从mdin中读取控制信息
    ============================*/
    controller[0].printf("START INITIALIZING NEIGHBOR LIST:\n");
    atom_numbers = md_atom_numbers;
    refresh_interval = 0;
    if (controller[0].Command_Exist(this->module_name, "refresh_interval"))
    {
        controller->Check_Int(this->module_name, "refresh_interval", "NEIGHBOR_LIST::Initial");
        refresh_interval = atoi(controller[0].Command(this->module_name, "refresh_interval"));
    }
    max_atom_in_grid_numbers = 128;
    if (controller[0].Command_Exist(this->module_name, "max_atom_in_grid_numbers"))
    {
        controller->Check_Int(this->module_name, "max_atom_in_grid_numbers", "NEIGHBOR_LIST::Initial");
        max_atom_in_grid_numbers = atoi(controller[0].Command(this->module_name, "max_atom_in_grid_numbers"));
    }
    
    max_neighbor_numbers = 1200;
    if (controller[0].Command_Exist(this->module_name, "max_neighbor_numbers"))
    {
        controller->Check_Int(this->module_name, "max_neighbor_numbers", "NEIGHBOR_LIST::Initial");
        max_neighbor_numbers = atoi(controller[0].Command(this->module_name, "max_neighbor_numbers"));
    }
    skin_permit = 1.0;
    if (controller[0].Command_Exist(this->module_name, "skin_permit"))
    {
        controller->Check_Float(this->module_name, "skin_permit", "NEIGHBOR_LIST::Initial");
        skin_permit = 2.*atof(controller[0].Command(this->module_name, "skin_permit"));//以外界的0.5等于不变（即程序内的1.）
    }
    this->skin = skin;
    this->cutoff = cut;
    cutoff_square = cutoff*cutoff;
    cutoff_with_skin = cutoff + skin;
    half_cutoff_with_skin = 0.5*cutoff_with_skin;
    cutoff_with_skin_square = cutoff_with_skin*cutoff_with_skin;
    if (cutoff_with_skin >= box_length.x / 2
        || cutoff_with_skin >= box_length.y / 2
        || cutoff_with_skin >= box_length.z / 2)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "NEIGHBOR_LIST::Initial",
            "Reason:\n\tThe non bonded cutoff + skin should be no more than the half of shortest length of the box\n");
    }
    half_skin_square = 0.25*skin*skin;
    this->box_length = box_length;
    this->quarter_crd_to_uint_crd_cof = 0.25f * CONSTANT_UINT_MAX_FLOAT / box_length;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;


    /*===========================
    //初始化格子信息
    ============================*/
    Initial_Malloc();
    exceed_max_grid_atoms = 0;
    exceed_max_neighbor_atoms = 0;
    Cuda_Malloc_And_Copy_Safely((void**)&d_exceed_max_grid_atoms, &exceed_max_grid_atoms, sizeof(int), "d_exceed_max_grid_atoms");
    Cuda_Malloc_And_Copy_Safely((void**)&d_exceed_max_neighbor_atoms, &exceed_max_neighbor_atoms, sizeof(int), "d_exceed_max_neighbor_atoms");
    Initial_Neighbor_Grid(
        &grid_info.gpointer, &grid_info.bucket, &grid_info.atom_numbers_in_grid_bucket,
        half_cutoff_with_skin, &grid_info,
        max_atom_in_grid_numbers, box_length);
    is_initialized = 1;
    controller->printf("    grid dimension is %d %d %d\n", grid_info.Nx, grid_info.Ny, grid_info.Nz);
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING NEIGHBOR LIST\n\n");
}


void NEIGHBOR_LIST::Update_Volume(VECTOR box_length)
{
    if (!is_initialized)
        return;
    this->box_length = box_length;
    this->quarter_crd_to_uint_crd_cof = 0.25f * CONSTANT_UINT_MAX_FLOAT/ box_length;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;

    grid_info.grid_length.x = (float)box_length.x / grid_info.Nx;
    grid_info.grid_length.y = (float)box_length.y / grid_info.Ny;
    grid_info.grid_length.z = (float)box_length.z / grid_info.Nz;

    grid_info.grid_length_inverse = 1.0f / grid_info.grid_length;
}

void NEIGHBOR_LIST::Check_Overflow(CONTROLLER* controller)
{
    if (is_initialized)
    {
        cudaMemcpy(&exceed_max_grid_atoms, d_exceed_max_grid_atoms, sizeof(int), cudaMemcpyDeviceToHost);
        if (exceed_max_grid_atoms)
        {
            controller->Throw_SPONGE_Error(spongeErrorOverflow, "NEIGHBOR_LIST::Check_Overflow", 
                "Reason:\n\tOverflow occured in neighbor searching. 'neighbor_list_max_atom_in_grid_numbers' should be larger\n");
        }
        cudaMemcpy(&exceed_max_neighbor_atoms, d_exceed_max_neighbor_atoms, sizeof(int), cudaMemcpyDeviceToHost);
        if (exceed_max_neighbor_atoms)
        {
            controller->Throw_SPONGE_Error(spongeErrorOverflow, "NEIGHBOR_LIST::Check_Overflow",
                "Reason:\n\tOverflow occured in neighbor searching. 'neighbor_list_max_neighbor_numbers' should be larger\n");
        }
    }
}


void NEIGHBOR_LIST::Clear()
{
    if (is_initialized == 1)
    {
        is_initialized = 0;
        cudaFree(old_crd);
        old_crd = NULL;
        cudaFree(uint_crd);
        uint_crd = NULL;
        cudaFree(is_need_refresh_neighbor_list);
        is_need_refresh_neighbor_list = NULL;

        cudaFree(h_nl->atom_serial);
        free(h_nl);
        h_nl = NULL;
        cudaFree(d_nl);
        d_nl = NULL;

        cudaFree(grid_info.atom_in_grid_serial);
        grid_info.atom_in_grid_serial = NULL;

        cudaFree(grid_info.atom_numbers_in_grid_bucket);
        grid_info.atom_numbers_in_grid_bucket = NULL;

        cudaMemcpy(grid_info.h_bucket, grid_info.bucket, sizeof(GRID_BUCKET)*grid_info.grid_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_info.grid_numbers; i = i + 1)
        {
            cudaFree(grid_info.h_bucket[i].atom_serial);
        }
        cudaFree(grid_info.bucket);
        grid_info.bucket = NULL;
        free(grid_info.h_bucket);
        grid_info.h_bucket = NULL;

        cudaMemcpy(grid_info.h_pointer, grid_info.gpointer, sizeof(GRID_BUCKET)*grid_info.grid_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_info.grid_numbers; i = i + 1)
        {
            cudaFree(grid_info.h_pointer[i].grid_serial);
        }
        cudaFree(grid_info.gpointer);
        grid_info.gpointer = NULL;
        free(grid_info.h_pointer);
        grid_info.h_pointer = NULL;
    }
}
