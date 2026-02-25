#include "SHAKE.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

static inline bool Shake_Virial_Debug_On()
{
    const char* value = std::getenv("SPONGE_DEBUG_SHAKE_VIRIAL");
    return value != NULL && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static __global__ void Constrain_Force_Cycle
(const int constrain_pair_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
const CONSTRAIN_PAIR *constrain_pair, const VECTOR *pair_dr,
VECTOR *test_frc)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr;
        float frc_abs;
        
        VECTOR dr0 = pair_dr[pair_i];
        dr.x = ((int)(uint_crd[cp.atom_i_serial].uint_x - uint_crd[cp.atom_j_serial].uint_x)) * scaler.x;
        dr.y = ((int)(uint_crd[cp.atom_i_serial].uint_y - uint_crd[cp.atom_j_serial].uint_y)) * scaler.y;
        dr.z = ((int)(uint_crd[cp.atom_i_serial].uint_z - uint_crd[cp.atom_j_serial].uint_z)) * scaler.z;
        float r_1 = rnorm3df(dr.x, dr.y, dr.z);
        frc_abs = 0.5 * (dr * dr - cp.constant_r*cp.constant_r) / (dr * dr0)*cp.constrain_k;

        VECTOR frc_lin = frc_abs * dr0;

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}
static __global__ void Refresh_Uint_Crd(const int atom_numbers, const VECTOR *crd, const VECTOR box_length_inverse, UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *test_frc,
    const float *mass_inverse, const float half_exp_gamma_plus_half)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        float mass_lin = mass_inverse[atom_i];
        //mass_lin为mass的倒数，frc_lin已经乘以dt^2
        crd_lin.x = crd_lin.x + half_exp_gamma_plus_half*frc_lin.x*mass_lin;
        crd_lin.y = crd_lin.y + half_exp_gamma_plus_half*frc_lin.y*mass_lin;
        crd_lin.z = crd_lin.z + half_exp_gamma_plus_half*frc_lin.z*mass_lin;

        crd_lin.x = crd_lin.x * box_length_inverse.x;
        crd_lin.y = crd_lin.y * box_length_inverse.y;
        crd_lin.z = crd_lin.z * box_length_inverse.z;

        crd_lin.x -= floorf(crd_lin.x); 
        crd_lin.y -= floorf(crd_lin.y);
        crd_lin.z -= floorf(crd_lin.z);

        uint_crd[atom_i].uint_x = crd_lin.x * UINT_MAX;
        uint_crd[atom_i].uint_y = crd_lin.y * UINT_MAX;
        uint_crd[atom_i].uint_z = crd_lin.z * UINT_MAX;
    }
}

static __global__ void Last_Crd_To_dr
(const int constarin_pair_numbers, const UNSIGNED_INT_VECTOR *atom_crd,
const VECTOR scaler, const CONSTRAIN_PAIR *constrain_pair, VECTOR *pair_dr)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constarin_pair_numbers)
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr = Get_Periodic_Displacement(atom_crd[cp.atom_i_serial],
                                              atom_crd[cp.atom_j_serial],
                                              scaler);
        pair_dr[pair_i] = dr;
    }
}

static __global__ void Refresh_Crd_Vel(const int atom_numbers, const float dt_inverse, const float dt, VECTOR *crd, VECTOR *vel, const VECTOR *test_frc,
    const float *mass_inverse, const float exp_gamma, const float half_exp_gamma_plus_half)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        VECTOR vel_lin = vel[atom_i];
        float mass_lin = mass_inverse[atom_i];

        frc_lin.x = frc_lin.x*mass_lin;
        frc_lin.y = frc_lin.y*mass_lin;
        frc_lin.z = frc_lin.z*mass_lin;//mass实际为mass的倒数，frc_lin已经乘以dt^2

        crd_lin.x = crd_lin.x + half_exp_gamma_plus_half*frc_lin.x;
        crd_lin.y = crd_lin.y + half_exp_gamma_plus_half*frc_lin.y;
        crd_lin.z = crd_lin.z + half_exp_gamma_plus_half*frc_lin.z;


        vel_lin.x = (vel_lin.x + exp_gamma*frc_lin.x*dt_inverse);
        vel_lin.y = (vel_lin.y + exp_gamma*frc_lin.y*dt_inverse);
        vel_lin.z = (vel_lin.z + exp_gamma*frc_lin.z*dt_inverse);

        crd[atom_i] = crd_lin;
        vel[atom_i] = vel_lin;
    }
}

void SHAKE::Initial_Simple_Constrain(CONTROLLER *controller, CONSTRAIN *constrain, const char *module_name)
{

    //从传入的参数复制基本信息
    this->constrain = constrain;
    if (module_name == NULL)
    {
        strcpy(this->module_name, "shake");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (constrain->constrain_pair_numbers > 0)
    {
        controller[0].printf("START INITIALIZING SHAKE:\n");
        iteration_numbers = 25;
        if (controller[0].Command_Exist(this->module_name, "iteration_numbers"))
        {
            controller->Check_Float(this->module_name, "iteration_numbers", "SHAKE::Initial_Simple_Constrain");
            int scanf_ret = sscanf(controller[0].Command(this->module_name, "iteration_numbers"), "%d", &iteration_numbers);
        }
        controller[0].printf("    constrain iteration step is %d\n", iteration_numbers);

        step_length = 1.0f;
        if (controller[0].Command_Exist(this->module_name, "step_length"))
        {
            controller->Check_Float(this->module_name, "step_length", "SHAKE::Initial_Simple_Constrain");
            int scanf_ret = sscanf(controller[0].Command(this->module_name, "step_length"), "%f", &step_length);
        }
        controller[0].printf("    constrain step length is %.2f\n", step_length);

        Cuda_Malloc_Safely((void**)&constrain_frc, sizeof(VECTOR)*constrain->atom_numbers);
        Cuda_Malloc_Safely((void**)&test_uint_crd, sizeof(UNSIGNED_INT_VECTOR)*constrain->atom_numbers);
        Cuda_Malloc_Safely((void**)&last_pair_dr, sizeof(VECTOR)*constrain->constrain_pair_numbers);
        Cuda_Malloc_Safely((void**)&d_pair_virial, sizeof(float)*constrain->constrain_pair_numbers);
        Cuda_Malloc_Safely((void**)&d_virial, sizeof(float));

        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller[0].printf("    structure last modify date is %d\n", last_modify_date);
        }
        controller[0].printf("END INITIALIZING SHAKE\n\n");
        is_initialized = 1;
    }
    else
    {
        controller[0].printf("SHAKE IS NOT INITIALIZED\n\n");
    }

}

void SHAKE::Remember_Last_Coordinates(VECTOR *crd, UNSIGNED_INT_VECTOR *uint_crd, VECTOR scaler)
{
    if (is_initialized)
    {
        //获得分子模拟迭代中上一步的距离信息
        CUDA_LAUNCH(Last_Crd_To_dr,
            dim3(static_cast<unsigned int>(ceilf((float)constrain->constrain_pair_numbers / 128))),
            dim3(128),
            constrain->constrain_pair_numbers,
            uint_crd,
            scaler,
            constrain->constrain_pair,
            last_pair_dr);
    }
}

static __global__ void Constrain_Force_Cycle_With_Virial
(const int constrain_pair_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
const CONSTRAIN_PAIR *constrain_pair, const VECTOR *pair_dr,
VECTOR *test_frc, float *d_atom_virial)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr0 = pair_dr[pair_i];
        VECTOR dr = Get_Periodic_Displacement(uint_crd[cp.atom_i_serial], uint_crd[cp.atom_j_serial], scaler);
        float r_1 = rnorm3df(dr.x, dr.y, dr.z);
        float frc_abs = 0.5 * (dr * dr - cp.constant_r*cp.constant_r) / (dr * dr0)*cp.constrain_k;
        VECTOR frc_lin = frc_abs * dr0;
        d_atom_virial[pair_i] -= frc_lin * dr0;

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}

static __global__ void pressure_fix(float *pressure, float *virial, float factor)
{
    pressure[0] += factor * virial[0];
}

void SHAKE::Constrain
(VECTOR *crd, VECTOR *vel, const float *mass_inverse, const float *d_mass, VECTOR box_length, int need_pressure, float *d_pressure)
{
    if (is_initialized)
    {
        //清空约束力和维里
        CUDA_LAUNCH(Reset_List,
            dim3(static_cast<unsigned int>(ceilf((float)3. * constrain->atom_numbers / 128))),
            dim3(128),
            3 * constrain->atom_numbers,
            (float*)constrain_frc,
            0.);
        if (need_pressure > 0)
        {
            CUDA_LAUNCH(Reset_List,
                dim3(static_cast<unsigned int>(ceilf((float)constrain->constrain_pair_numbers / 1024.0f))),
                dim3(1024),
                constrain->constrain_pair_numbers,
                d_pair_virial,
                0.0f);
            CUDA_LAUNCH(Reset_List, dim3(1), dim3(1), 1, d_virial, 0.0f);
        }
        for (int i = 0; i < iteration_numbers; i = i + 1)
        {
            CUDA_LAUNCH(Refresh_Uint_Crd,
                dim3(static_cast<unsigned int>(ceilf((float)constrain->atom_numbers / 128))),
                dim3(128),
                constrain->atom_numbers,
                crd,
                1.0f / box_length,
                test_uint_crd,
                constrain_frc,
                mass_inverse,
                constrain->x_factor);
            if (need_pressure > 0)
            {
                CUDA_LAUNCH(Constrain_Force_Cycle_With_Virial,
                    dim3(static_cast<unsigned int>(ceilf((float)constrain->constrain_pair_numbers / 128))),
                    dim3(128),
                    constrain->constrain_pair_numbers,
                    test_uint_crd,
                    constrain->uint_dr_to_dr_cof,
                    constrain->constrain_pair,
                    last_pair_dr,
                    constrain_frc,
                    d_pair_virial);
            }
            else
            {
                CUDA_LAUNCH(Constrain_Force_Cycle,
                    dim3(static_cast<unsigned int>(ceilf((float)constrain->constrain_pair_numbers / 128))),
                    dim3(128),
                    constrain->constrain_pair_numbers,
                    test_uint_crd,
                    constrain->uint_dr_to_dr_cof,
                    constrain->constrain_pair,
                    last_pair_dr,
                    constrain_frc);
            }
        }
        if (need_pressure > 0)
        {
            float pressure_before = 0.0f;
            if (Shake_Virial_Debug_On())
            {
                cudaMemcpy(&pressure_before, d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
            }
            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), constrain->constrain_pair_numbers, d_pair_virial, d_virial);
            const float pressure_factor = 1 / constrain->dt / constrain->dt / 3.0 / constrain->volume;
            float virial_sum = 0.0f;
            if (Shake_Virial_Debug_On())
            {
                cudaMemcpy(&virial_sum, d_virial, sizeof(float), cudaMemcpyDeviceToHost);
                std::vector<float> host_pair_virial(static_cast<size_t>(constrain->constrain_pair_numbers));
                cudaMemcpy(
                    host_pair_virial.data(),
                    d_pair_virial,
                    sizeof(float) * static_cast<size_t>(constrain->constrain_pair_numbers),
                    cudaMemcpyDeviceToHost);
                double pair_sum = 0.0;
                double pair_abs_sum = 0.0;
                double pair_pos_sum = 0.0;
                double pair_neg_sum = 0.0;
                int pair_pos_count = 0;
                int pair_neg_count = 0;
                int pair_zero_count = 0;
                float pair_min = 0.0f;
                float pair_max = 0.0f;
                if (!host_pair_virial.empty())
                {
                    pair_min = host_pair_virial[0];
                    pair_max = host_pair_virial[0];
                    for (size_t i = 0; i < host_pair_virial.size(); i += 1)
                    {
                        const float v = host_pair_virial[i];
                        pair_sum += static_cast<double>(v);
                        pair_abs_sum += std::fabs(static_cast<double>(v));
                        if (v > 0.0f)
                        {
                            pair_pos_sum += static_cast<double>(v);
                            pair_pos_count += 1;
                        }
                        else if (v < 0.0f)
                        {
                            pair_neg_sum += static_cast<double>(v);
                            pair_neg_count += 1;
                        }
                        else
                        {
                            pair_zero_count += 1;
                        }
                        if (v < pair_min) pair_min = v;
                        if (v > pair_max) pair_max = v;
                    }
                }
                static int pair_debug_call = 0;
                pair_debug_call += 1;
                const char* dump_prefix = std::getenv("SPONGE_DEBUG_SHAKE_PAIRVIRIAL_DUMP_PREFIX");
                if (dump_prefix != NULL && dump_prefix[0] != '\0')
                {
                    char dump_file[1024];
                    std::snprintf(dump_file, sizeof(dump_file), "%s_call%d.csv", dump_prefix, pair_debug_call);
                    std::vector<CONSTRAIN_PAIR> host_pairs(static_cast<size_t>(constrain->constrain_pair_numbers));
                    std::vector<VECTOR> host_last_pair_dr(static_cast<size_t>(constrain->constrain_pair_numbers));
                    cudaMemcpy(
                        host_pairs.data(),
                        constrain->constrain_pair,
                        sizeof(CONSTRAIN_PAIR) * static_cast<size_t>(constrain->constrain_pair_numbers),
                        cudaMemcpyDeviceToHost);
                    cudaMemcpy(
                        host_last_pair_dr.data(),
                        last_pair_dr,
                        sizeof(VECTOR) * static_cast<size_t>(constrain->constrain_pair_numbers),
                        cudaMemcpyDeviceToHost);
                    FILE* fp = std::fopen(dump_file, "w");
                    if (fp != NULL)
                    {
                        std::fprintf(
                            fp,
                            "pair_index,atom_i,atom_j,constant_r,constrain_k,last_pair_dr_x,last_pair_dr_y,last_pair_dr_z,virial_value\n");
                        for (size_t i = 0; i < host_pair_virial.size(); i += 1)
                        {
                            const CONSTRAIN_PAIR& cp = host_pairs[i];
                            const VECTOR& dr0 = host_last_pair_dr[i];
                            std::fprintf(
                                fp,
                                "%zu,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                                i,
                                cp.atom_i_serial,
                                cp.atom_j_serial,
                                cp.constant_r,
                                cp.constrain_k,
                                dr0.x,
                                dr0.y,
                                dr0.z,
                                host_pair_virial[i]);
                        }
                        std::fclose(fp);
                    }
                }
                std::printf(
                    "DEBUG_SHAKE_PAIRVIRIAL call=%d n=%d sum=%e abs_sum=%e min=%e max=%e pos_sum=%e neg_sum=%e pos_n=%d neg_n=%d zero_n=%d\n",
                    pair_debug_call,
                    constrain->constrain_pair_numbers,
                    pair_sum,
                    pair_abs_sum,
                    pair_min,
                    pair_max,
                    pair_pos_sum,
                    pair_neg_sum,
                    pair_pos_count,
                    pair_neg_count,
                    pair_zero_count);
            }
            CUDA_LAUNCH(pressure_fix, dim3(1), dim3(1), d_pressure, d_virial, pressure_factor);
            if (Shake_Virial_Debug_On())
            {
                float pressure_after = 0.0f;
                cudaMemcpy(&pressure_after, d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
                static int debug_call = 0;
                debug_call += 1;
                std::printf(
                    "DEBUG_SHAKE_VIRIAL call=%d pair_n=%d virial_sum=%e factor=%e pressure_before=%e pressure_after=%e pressure_delta=%e\n",
                    debug_call,
                    constrain->constrain_pair_numbers,
                    virial_sum,
                    pressure_factor,
                    pressure_before,
                    pressure_after,
                    pressure_after - pressure_before);
                std::fflush(stdout);
            }
        }

        CUDA_LAUNCH(Refresh_Crd_Vel,
            dim3(static_cast<unsigned int>(ceilf((float)constrain->atom_numbers / 128))),
            dim3(128),
            constrain->atom_numbers,
            constrain->dt_inverse,
            constrain->dt,
            crd,
            vel,
            constrain_frc,
            mass_inverse,
            constrain->v_factor,
            constrain->x_factor);
    }
}


void SHAKE::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(last_pair_dr);
        last_pair_dr = NULL;

        cudaFree(constrain_frc);
        constrain_frc = NULL;

        cudaFree(test_uint_crd);
        test_uint_crd = NULL;

        cudaFree(d_pair_virial);
        d_pair_virial = NULL;

        cudaFree(d_virial);
        d_virial = NULL;
    }
}
