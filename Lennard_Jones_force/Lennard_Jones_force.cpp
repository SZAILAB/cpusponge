#include "Lennard_Jones_force.h"
#include <cstdio>
#include <cctype>
#include <cstdlib>

// 由LJ坐标和转化系数求距离
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers, UINT_VECTOR_LJ_TYPE *new_crd, const int *LJ_type)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].LJ_type = LJ_type[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd, const float *charge)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].uint_x = crd[atom_i].uint_x;
        new_crd[atom_i].uint_y = crd[atom_i].uint_y;
        new_crd[atom_i].uint_z = crd[atom_i].uint_z;
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].uint_x = crd[atom_i].uint_x;
        new_crd[atom_i].uint_y = crd[atom_i].uint_y;
        new_crd[atom_i].uint_z = crd[atom_i].uint_z;
    }
}

static __global__ void device_add(float *variable, const float adder)
{
    variable[0] += adder;
}

template<bool need_force, bool need_energy, bool need_virial, bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_CUDA(
    const int atom_numbers, const ATOM_GROUP* nl,
    const UINT_VECTOR_LJ_TYPE* uint_crd, const VECTOR boxlength,
    const float* LJ_type_A, const float* LJ_type_B, const float cutoff,
    VECTOR* frc, const float pme_beta, float* atom_energy, float* atom_lj_virial, float* atom_direct_cf_energy)
{
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i];
        VECTOR frc_record = { 0., 0., 0. };
        float virial_lj = 0.;
        float energy_lj = 0.;
        float energy_coulomb = 0.;
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_TYPE r2 = uint_crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                if (need_force)
                {
                    float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs = Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    atomicAdd(frc + atom_j, -frc_lin);
                }
                if (need_coulomb && (need_energy || need_virial))
                {
                    energy_coulomb += Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                }
                if (need_energy)
                {
                    energy_lj += Get_LJ_Energy(r1, r2, dr_abs, A, B);
                }
                if (need_virial)
                {
                    virial_lj += Get_LJ_Virial(r1, r2, dr_abs, A, B);
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record);
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_lj);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_lj_virial + atom_i, virial_lj);
        }
    }
}

#ifdef SPONGE_CPU_PHASE1
static inline bool Cpu_Debug_LJ_Max_Pair_On()
{
    static int inited = 0;
    static bool on = false;
    if (!inited)
    {
        const char* s = std::getenv("SPONGE_CPU_DEBUG_LJ_MAX_PAIR");
        on = (s != NULL && s[0] != '\0' && std::strcmp(s, "0") != 0);
        inited = 1;
    }
    return on;
}

template<bool need_force, bool need_energy, bool need_virial, bool need_coulomb>
static inline void Lennard_Jones_And_Direct_Coulomb_CPU_Serial(
    const int atom_numbers, const ATOM_GROUP* nl,
    const UINT_VECTOR_LJ_TYPE* uint_crd, const VECTOR boxlength,
    const float* LJ_type_A, const float* LJ_type_B, const float cutoff,
    VECTOR* frc, const float pme_beta, float* atom_energy, float* atom_lj_virial, float* atom_direct_cf_energy)
{
    const bool debug_max_pair = Cpu_Debug_LJ_Max_Pair_On();
    float max_abs_frc = -1.0f;
    int max_i = -1;
    int max_j = -1;
    float max_pair_dist = 0.0f;
    float min_dist = 1.0e30f;
    int min_i = -1;
    int min_j = -1;

    for (int atom_i = 0; atom_i < atom_numbers; ++atom_i)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i];
        VECTOR frc_record = { 0.0f, 0.0f, 0.0f };
        float virial_lj = 0.0f;
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        for (int j = 0; j < nl_i.atom_numbers; ++j)
        {
            const int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_TYPE r2 = uint_crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                if (need_force)
                {
                    float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs = Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs -= frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    frc[atom_j] = frc[atom_j] - frc_lin;

                    if (debug_max_pair)
                    {
                        const float abs_frc = std::fabs(frc_abs);
                        if (abs_frc > max_abs_frc)
                        {
                            max_abs_frc = abs_frc;
                            max_i = atom_i;
                            max_j = atom_j;
                            max_pair_dist = dr_abs;
                        }
                    }
                }
                if (debug_max_pair)
                {
                    if (dr_abs < min_dist)
                    {
                        min_dist = dr_abs;
                        min_i = atom_i;
                        min_j = atom_j;
                    }
                }
                if (need_coulomb && (need_energy || need_virial))
                {
                    energy_coulomb += Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                }
                if (need_energy)
                {
                    energy_lj += Get_LJ_Energy(r1, r2, dr_abs, A, B);
                }
                if (need_virial)
                {
                    virial_lj += Get_LJ_Virial(r1, r2, dr_abs, A, B);
                }
            }
        }
        if (need_force)
        {
            frc[atom_i] = frc[atom_i] + frc_record;
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            atom_direct_cf_energy[atom_i] += energy_coulomb;
        }
        if (need_energy)
        {
            atom_energy[atom_i] += energy_lj;
        }
        if (need_virial)
        {
            atom_lj_virial[atom_i] += virial_lj;
        }
    }
    if (debug_max_pair)
    {
        std::fprintf(
            stderr,
            "CPU_DEBUG_LJ_SERIAL atom_numbers=%d max_abs_frc=%e max_pair_i=%d max_pair_j=%d max_pair_dr=%e min_dr=%e min_i=%d min_j=%d\n",
            atom_numbers,
            max_abs_frc,
            max_i,
            max_j,
            max_pair_dist,
            min_dist,
            min_i,
            min_j);
    }
}
#endif

void LENNARD_JONES_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int)*atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float)*pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float)*pair_type_numbers);

    Cuda_Malloc_Safely((void**)&d_LJ_energy_sum, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_atom, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_LJ_type, sizeof(int)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_A, sizeof(float)*pair_type_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_B, sizeof(float)*pair_type_numbers);
}

static __global__ void Total_C6_Get(int atom_numbers, int *atom_lj_type, float *d_lj_b, float *d_factor)
{
    int i, j;
    float temp_sum = 0;
    int x, y;
    int itype, jtype, atom_pair_LJ_type;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
    {
        itype = atom_lj_type[i];
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers; j += gridDim.y * blockDim.y)
        {
            jtype = atom_lj_type[j];
            y = (jtype - itype);
            x = y >> 31;
            y = (y^x) - x;
            x = jtype + itype;
            jtype = (x + y) >> 1;
            x = (x - y) >> 1;
            atom_pair_LJ_type = (jtype*(jtype + 1) >> 1) + x;
            temp_sum += d_lj_b[atom_pair_LJ_type];
        }
    }
    atomicAdd(d_factor, temp_sum);
}

void LENNARD_JONES_INFORMATION::Initial(CONTROLLER *controller, float cutoff, VECTOR box_length, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        FILE *fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"), "r");

        int scanf_ret = fscanf(fp, "%d %d", &atom_numbers, &atom_type_numbers);
        controller[0].printf("    atom_numbers is %d\n", atom_numbers);
        controller[0].printf("    atom_LJ_type_number is %d\n", atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", h_LJ_A + i);
            h_LJ_A[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", h_LJ_B + i);
            h_LJ_B[i] *= 6.0f;
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%d", h_atom_LJ_type + i);
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        Initial_From_AMBER_Parm(controller[0].Command("amber_parm7"), controller[0]);
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
        Cuda_Malloc_Safely((void **)&uint_crd_with_LJ, sizeof(UINT_VECTOR_LJ_TYPE)* atom_numbers);
        CUDA_LAUNCH(Copy_LJ_Type_To_New_Crd,
            dim3(static_cast<unsigned int>(ceilf((float)this->atom_numbers / 32))),
            dim3(32),
            atom_numbers,
            uint_crd_with_LJ,
            d_atom_LJ_type);
        controller[0].printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;
        float *d_factor = NULL;
        Cuda_Malloc_Safely((void**)&d_factor, sizeof(float));
        Reset_List(d_factor, 0.0f, 1, 1);
        CUDA_LAUNCH(Total_C6_Get, dim3(4, 4), dim3(32, 32), atom_numbers, d_atom_LJ_type, d_LJ_B, d_factor);
        cudaMemcpy(&long_range_factor, d_factor, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_factor);

        long_range_factor *= -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        this->volume = box_length.x * box_length.y * box_length.z;
        controller[0].printf("        long range correction factor is: %e\n", long_range_factor);
        controller[0].printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

void LENNARD_JONES_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        free(h_atom_LJ_type);
        cudaFree(d_atom_LJ_type);

        free(h_LJ_A);
        free(h_LJ_B);
        cudaFree(d_LJ_A);
        cudaFree(d_LJ_B);

        free(h_LJ_energy_atom);
        cudaFree(d_LJ_energy_atom);
        cudaFree(d_LJ_energy_sum);

        cudaFree(uint_crd_with_LJ);

        h_atom_LJ_type = NULL;
        d_atom_LJ_type = NULL;

        h_LJ_A = NULL;
        h_LJ_B = NULL;
        d_LJ_A = NULL;
        d_LJ_B = NULL;

        h_LJ_energy_atom = NULL;
        d_LJ_energy_atom = NULL;
        d_LJ_energy_sum = NULL;

        uint_crd_with_LJ = NULL;
    }
}


void LENNARD_JONES_INFORMATION::Long_Range_Correction(int need_pressure, float *d_virial, int need_potential, float *d_potential)
{
    if (is_initialized)
    {
        if (need_pressure > 0)
        {
            CUDA_LAUNCH(device_add, dim3(1), dim3(1), d_virial, long_range_factor * 6.0f / volume);
        }
        if (need_potential > 0)
        {
            CUDA_LAUNCH(device_add, dim3(1), dim3(1), d_potential, long_range_factor / volume);
        }
    }
}

void LENNARD_JONES_INFORMATION::Initial_From_AMBER_Parm(const char *file_name, CONTROLLER controller)
{
    FILE *parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    controller.printf("    Start reading LJ information from AMBER file:\n");

    while (true)
    {
        char temps[CHAR_LENGTH_MAX];
        char temp_first_str[CHAR_LENGTH_MAX];
        char temp_second_str[CHAR_LENGTH_MAX];
        if (!fgets(temps, CHAR_LENGTH_MAX, parm))
        {
            break;
        }
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "POINTERS") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
            controller.printf("        atom_numbers is %d\n", atom_numbers);
            scanf_ret = fscanf(parm, "%d\n", &atom_type_numbers);
            controller.printf("        atom_LJ_type_number is %d\n", atom_type_numbers);
            pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;

            LJ_Malloc();
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "ATOM_TYPE_INDEX") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ type index\n");
            int atomljtype;
            for (int i = 0; i < atom_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%d\n", &atomljtype);
                h_atom_LJ_type[i] = atomljtype - 1;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "LENNARD_JONES_ACOEF") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ A\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_A[i] = 12.0f* lin;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "LENNARD_JONES_BCOEF") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ B\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_B[i] = (float)6.* lin;
            }
        }
    }
    controller.printf("    End reading LJ information from AMBER file:\n");
    fclose(parm);
    is_initialized = 1;
    Parameter_Host_To_Device();
}

void LENNARD_JONES_INFORMATION::Parameter_Host_To_Device()
{
    cudaMemcpy(d_LJ_B, h_LJ_B, sizeof(float)*pair_type_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LJ_A, h_LJ_A, sizeof(float)*pair_type_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_LJ_type, h_atom_LJ_type, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
}

void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc,
    const ATOM_GROUP *nl, const float pme_beta, const int need_atom_energy, float *atom_energy,
    const int need_virial, float *atom_lj_virial, float *atom_direct_pme_energy)
{
    if (is_initialized)
    {
        CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
            dim3((this->atom_numbers + 1023) / 1024),
            dim3(1024),
            this->atom_numbers,
            uint_crd,
            uint_crd_with_LJ,
            charge);
        if (atom_numbers == 0)
        {
            if (need_atom_energy || need_virial)
                cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            return;
        }
#ifdef SPONGE_CPU_PHASE1
        if (need_atom_energy || need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
        }
        const char* use_serial_env = std::getenv("SPONGE_CPU_USE_LJ_SERIAL");
        bool use_serial = false;
        if (use_serial_env != NULL && use_serial_env[0] != '\0')
        {
            const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(use_serial_env[0])));
            use_serial = (c == '1' || c == 't' || c == 'y' || c == 'o');
        }
        if (use_serial)
        {
            if (!need_atom_energy && !need_virial)
            {
                Lennard_Jones_And_Direct_Coulomb_CPU_Serial<true, false, false, true>(
                    atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_A, d_LJ_B, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
            }
            else if (need_atom_energy && !need_virial)
            {
                Lennard_Jones_And_Direct_Coulomb_CPU_Serial<true, true, false, true>(
                    atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_A, d_LJ_B, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
            }
            else if (!need_atom_energy && need_virial)
            {
                Lennard_Jones_And_Direct_Coulomb_CPU_Serial<true, false, true, true>(
                    atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_A, d_LJ_B, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
            }
            else
            {
                Lennard_Jones_And_Direct_Coulomb_CPU_Serial<true, true, true, true>(
                    atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_A, d_LJ_B, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
            }
            return;
        }
#endif
        if (!need_atom_energy && !need_virial)
        {
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, false, true>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else if (need_atom_energy && !need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, false, true>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else if (!need_atom_energy && need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, true, true>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, true, true>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
    }
}

void LENNARD_JONES_INFORMATION::LJ_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
    const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
    const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 1024))),
            dim3(1024),
            atom_numbers,
            uint_crd,
            uint_crd_with_LJ,
            charge);
        if (!need_atom_energy && !need_virial)
        {
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, false, false>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else if (need_atom_energy && !need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, false, false>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else if (!need_atom_energy && need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, true, false>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
        else
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, true, false>),
                dim3((atom_numbers + 31) / 32),
                dim3(thread_LJ),
                atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_A, d_LJ_B, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);
        }
    }
}

float LENNARD_JONES_INFORMATION::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, const float pme_beta, const float* charge, float* pme_direct_energy, int is_download)
{
    if (is_initialized)
    {
        CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 32))),
            dim3(32),
            atom_numbers,
            uint_crd,
            uint_crd_with_LJ,
            charge);
        Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
        CUDA_LAUNCH(Reset_List,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 1024.0f))),
            dim3(1024),
            atom_numbers,
            pme_direct_energy,
            0.0f);
        CUDA_LAUNCH((Lennard_Jones_And_Direct_Coulomb_CUDA<false, true, false, true>),
            dim3((atom_numbers + 31) / 32),
            dim3(thread_LJ),
            atom_numbers, nl,
            uint_crd_with_LJ, uint_dr_to_dr_cof,
            d_LJ_A, d_LJ_B, cutoff, NULL,
            pme_beta, d_LJ_energy_atom, NULL, pme_direct_energy);
        Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
        CUDA_LAUNCH(device_add, dim3(1), dim3(1), d_LJ_energy_sum, long_range_factor / volume);
        if (is_download)
        {
            cudaMemcpy(&h_LJ_energy_sum, this->d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
            return h_LJ_energy_sum;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}

void LENNARD_JONES_INFORMATION::Update_Volume(VECTOR box_length)
{
    if (!is_initialized)
        return;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    this->volume = box_length.x * box_length.y * box_length.z;
}

void LENNARD_JONES_INFORMATION::Energy_Device_To_Host()
{
    cudaMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
}
