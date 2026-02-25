#include "cross_PME.h"

void CROSS_PME::Initial(CONTROLLER* controller, const int atom_numbers, const  int PME_Nall, const float lambda_lj, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "cross_pme");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist("charge_perturbated"))
    {
        controller->Check_Int("charge_perturbated", "CROSS_PME::Initial");
        charge_perturbated = atoi(controller->Command("charge_perturbated"));
    }
    else
    {
        controller->Warn("Missing value of charge_perturbated for TI, set to default 0");
    }
    if (charge_perturbated < 1)
    {
        controller->printf("CROSS PME IS NOT INITIALIZED\n\n");
        return;
    }
    bool exist_A = controller->Command_Exist("chargeA_in_file");
    bool exist_B = controller->Command_Exist("chargeB_in_file");
    if (!exist_A && !exist_B)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "CROSS_PME::Initial", "Reason:\n\tchargeA_in_file and chargeB_in_file are not provided but charge_perturbated > 0\n");
    }
    else if (!exist_A && exist_B)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "CROSS_PME::Initial", "Reason:\n\tchargeB_in_file is provided but chargeA_in_file is not\n");
    }
    else if (!exist_B && exist_A)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "CROSS_PME::Initial", "Reason:\n\tchargeA_in_file is provided but chargeB_in_file is not\n");
    }
    else
    {
        controller->printf("START INITIALIZING CROSS PME\n");
        controller->printf("    charge is perturbated with lj to a power of %d\n", charge_perturbated);
        float lambda_factor = 1;
        if (charge_perturbated > 1)
        {
            lambda_factor = charge_perturbated * powf(lambda_lj, charge_perturbated - 1);
        }
        Malloc_Safely((void**)&charge_B_A, sizeof(float) * atom_numbers);
        FILE* fa, * fb;
        Open_File_Safely(&fa, controller->Command("chargeA_in_file"), "r");
        Open_File_Safely(&fb, controller->Command("chargeB_in_file"), "r");
        int ret, file_atom_numbers;
        float temp;
        ret = fscanf(fa, "%d", &file_atom_numbers);
        if (ret != 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe format of chargeA_in_file is bad\n");
        }
        else if (file_atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe number of atoms in chargeA_in_file is different \
from the number of atoms provided by the core\n");
        }
        ret = fscanf(fb, "%d", &file_atom_numbers);
        if (ret != 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe format of chargeB_in_file is bad\n");
        }
        else if (file_atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe number of atoms in chargeB_in_file is different \
from the number of atoms provided by the core\n");
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            ret = fscanf(fb, "%f", charge_B_A + i);
            if (ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe format of chargeB_in_file is bad\n");
            }
            ret = fscanf(fa, "%f", &temp);
            if (ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CROSS_PME::Initial", "Reason:\n\tThe format of chargeA_in_file is bad\n");
            }
            charge_B_A[i] -= temp;
            charge_B_A[i] *= lambda_factor;
        }
        fclose(fa);
        fclose(fb);
        Cuda_Malloc_And_Copy_Safely((void**)&d_charge_B_A, charge_B_A, sizeof(float) * atom_numbers, "d_charge_B_A");
        Cuda_Malloc_Safely((void**)&PME_Q_B_A, sizeof(float) * PME_Nall);
        Cuda_Malloc_Safely((void**)&d_cross_reciprocal_ene, sizeof(float));
        Cuda_Malloc_Safely((void**)&d_cross_self_ene, sizeof(float));
        Cuda_Malloc_Safely((void**)&charge_sum_B_A, sizeof(float));
        Cuda_Malloc_Safely((void**)&d_cross_correction_atom_energy, sizeof(float) * atom_numbers);
        Cuda_Malloc_Safely((void**)&d_cross_correction_ene, sizeof(float));
        is_initialized = 1;
        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller->printf("    structure last modify date is %d\n", last_modify_date);
        }
        controller->printf("END INITIALIZING CROSS PME\n\n");
    }
}

static __global__ void device_add(float* ene, float factor, float* charge_sum, float* charge_sum_B_A)
{
    ene[0] += factor * charge_sum[0] * charge_sum_B_A[0];
}

static __global__ void PME_Cross_Excluded_Energy_Correction
(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR sacler,
    const float* charge, const float* charge_B_A, const float pme_beta,
    const int* excluded_list_start, const int* excluded_list, const int* excluded_atom_numbers,
    float* ene)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int excluded_number = excluded_atom_numbers[atom_i];
        if (excluded_number > 0)
        {
            int list_start = excluded_list_start[atom_i];
            int list_end = list_start + excluded_number;
            int atom_j;
            int int_x;
            int int_y;
            int int_z;

            float charge_i = charge[atom_i];
            float charge_B_A_i = charge_B_A[atom_i];
            float charge_j, charge_B_A_j;
            float dr_abs;
            float beta_dr;

            UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
            VECTOR dr;
            float dr2;

            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                atom_j = excluded_list[i];
                r2 = uint_crd[atom_j];
                charge_j = charge[atom_j];
                charge_B_A_j = charge_B_A[atom_j];

                int_x = r2.uint_x - r1.uint_x;
                int_y = r2.uint_y - r1.uint_y;
                int_z = r2.uint_z - r1.uint_z;
                dr.x = sacler.x * int_x;
                dr.y = sacler.y * int_y;
                dr.z = sacler.z * int_z;
                dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                //假设剔除表中的原子对距离总是小于cutoff的，正常体系
                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta * dr_abs;

                ene_lin -= (charge_i * charge_B_A_j + charge_B_A_i * charge_j) * erff(beta_dr) / dr_abs;

            }//atom_j cycle
            atomicAdd(ene + atom_i, ene_lin);
        }//if need excluded
    }
}

float CROSS_PME::Get_Partial_H_Partial_Lambda(Particle_Mesh_Ewald *pme, const int atom_numbers,
    const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR uint_dr_to_dr_cof, const float* d_charge, const ATOM_GROUP *nl,
    int * d_excluded_list_start, int *d_excluded_list, int *d_excluded_numbers, float* d_direct_ene)
{
    if (is_initialized)
    {
        CUDA_LAUNCH(PME_Atom_Near,
            dim3(static_cast<unsigned int>(atom_numbers / 32 + 1)),
            dim3(32),
            uint_crd, pme->PME_atom_near, pme->PME_Nin,
            CONSTANT_UINT_MAX_INVERSED * pme->fftx, CONSTANT_UINT_MAX_INVERSED * pme->ffty, CONSTANT_UINT_MAX_INVERSED * pme->fftz,
            atom_numbers, pme->fftx, pme->ffty, pme->fftz,
            pme->PME_kxyz, pme->PME_uxyz, pme->PME_frxyz);

        CUDA_LAUNCH(Reset_List, dim3(static_cast<unsigned int>(pme->PME_Nall / 1024 + 1)), dim3(1024), pme->PME_Nall, pme->PME_Q, 0);
        CUDA_LAUNCH(Reset_List, dim3(static_cast<unsigned int>(pme->PME_Nall / 1024 + 1)), dim3(1024), pme->PME_Nall, this->PME_Q_B_A, 0);

        CUDA_LAUNCH(PME_Q_Spread,
            dim3(static_cast<unsigned int>(atom_numbers / pme->thread_PME.x + 1)),
            dim3(pme->thread_PME),
            pme->PME_atom_near, d_charge, pme->PME_frxyz,
            pme->PME_Q, pme->PME_kxyz, atom_numbers);

        CUDA_LAUNCH(PME_Q_Spread,
            dim3(static_cast<unsigned int>(atom_numbers / pme->thread_PME.x + 1)),
            dim3(pme->thread_PME),
            pme->PME_atom_near, this->d_charge_B_A, pme->PME_frxyz, this->PME_Q_B_A, pme->PME_kxyz, atom_numbers);

        cufftExecR2C(pme->PME_plan_r2c, (float*)pme->PME_Q, (cufftComplex*)pme->PME_FQ);

        CUDA_LAUNCH(PME_BCFQ,
            dim3(static_cast<unsigned int>(pme->PME_Nfft / 1024 + 1)),
            dim3(1024),
            pme->PME_FQ,
            pme->PME_BC,
            pme->PME_Nfft);

        cufftExecC2R(pme->PME_plan_c2r, (cufftComplex*)pme->PME_FQ, (float*)pme->PME_FBCFQ);

        CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), pme->PME_Nall, this->PME_Q_B_A, pme->PME_FBCFQ, this->d_cross_reciprocal_ene);

        CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), atom_numbers, d_charge, this->d_charge_B_A, this->d_cross_self_ene);

        CUDA_LAUNCH(Scale_List, dim3(1), dim3(1), 1, this->d_cross_self_ene, -2 * pme->beta / sqrtf(CONSTANT_Pi));

        CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, d_charge, pme->charge_sum);
        CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, this->d_charge_B_A, this->charge_sum_B_A);

        CUDA_LAUNCH(device_add, dim3(1), dim3(1), this->d_cross_self_ene, pme->neutralizing_factor, pme->charge_sum, this->charge_sum_B_A);

        CUDA_LAUNCH(Reset_List, dim3(static_cast<unsigned int>((atom_numbers + 1023) / 1024)), dim3(1024), atom_numbers, this->d_cross_correction_atom_energy, 0.0f);
        CUDA_LAUNCH(PME_Cross_Excluded_Energy_Correction,
            dim3(static_cast<unsigned int>((atom_numbers + 31) / 32)),
            dim3(32),
            atom_numbers, uint_crd, uint_dr_to_dr_cof,
            d_charge, this->d_charge_B_A, pme->beta, d_excluded_list_start, d_excluded_list, d_excluded_numbers, this->d_cross_correction_atom_energy);
        CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, this->d_cross_correction_atom_energy, this->d_cross_correction_ene);

        cudaMemcpy(&cross_reciprocal_ene, d_cross_reciprocal_ene, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cross_self_ene, d_cross_self_ene, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cross_correction_ene, d_cross_correction_ene, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cross_direct_ene, d_direct_ene, sizeof(float), cudaMemcpyDeviceToHost);
        dH_dlambda = cross_reciprocal_ene + cross_self_ene + cross_correction_ene + cross_direct_ene;
        return dH_dlambda;
    }
    else
    {
        return NAN;
    }
}
