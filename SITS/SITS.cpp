#include "SITS.h"

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

static __global__ void device_add(float* variable, const float* adder)
{
    variable[0] += adder[0];
}

template<bool need_force, bool need_energy, bool need_virial, bool need_coulomb>
static __global__ void Selective_Lennard_Jones_And_Direct_Coulomb_CUDA(
    const int atom_numbers, const ATOM_GROUP* nl,
    const UINT_VECTOR_LJ_TYPE* uint_crd, const VECTOR boxlength,
    const float* LJ_type_A, const float* LJ_type_B, const int* atom_sys_mark, const float cutoff,
    VECTOR* frc, VECTOR* frc_enhancing, const float pme_beta, float* atom_energy, float* atom_energy_enhancing,
    float* atom_virial, float* atom_virial_enhancing, float* atom_direct_cf_energy, const float pwwp_factor)
{
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i];
        int atom_mark_i = atom_sys_mark[atom_i];
        VECTOR frc_record = { 0.0f, 0.0f, 0.0f }, frc_enhancing_record = { 0.0f, 0.0f, 0.0f };
        float virial_record = 0.0f, virial_enhancing = 0.0f;
        float energy_total = 0.0f, energy_enhancing = 0.0f, energy_coulomb = 0.0f;
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_TYPE r2 = uint_crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
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
                    frc_lin = factor * frc_lin;
                    frc_enhancing_record = frc_enhancing_record + frc_lin;
                }
                if (need_coulomb && (need_energy || need_virial))
                {
                    float energy_lin = Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    energy_coulomb += energy_lin;
                    energy_enhancing += factor * energy_lin;
                    virial_enhancing += factor * Get_Direct_Coulomb_Virial(r1, r2, dr_abs, pme_beta);
                }
                if (need_energy)
                {
                    float energy_lin = Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    energy_total += energy_lin;
                    energy_enhancing += factor * energy_lin;
                }
                if (need_virial)
                {
                    float virial_lin = Get_LJ_Virial(r1, r2, dr_abs, A, B);
                    virial_record += virial_lin;
                    virial_enhancing += factor * virial_lin;
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record);
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing);
        }
    }
}


template<bool need_force, bool need_energy, bool need_virial, bool need_coulomb, bool need_du_dlambda>
static __global__ void Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA(
    const int atom_numbers, const ATOM_GROUP* nl,
    const UINT_VECTOR_LJ_FEP_TYPE* uint_crd, const VECTOR boxlength, const int* atom_sys_mark,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA, const float* LJ_type_BB, const float cutoff,
    VECTOR* frc, VECTOR* frc_enhancing, const float pme_beta, float* atom_energy, float* atom_energy_enhancing, 
    float* atom_virial, float* atom_virial_enhancing, float* atom_direct_cf_energy, float* atom_du_dlambda_lj, float* atom_du_dlambda_direct, float* atom_du_dlambda_enhancing,
    const float lambda, const float alpha, const float p, const float input_sigma_6, const float input_sigma_6_min, const float pwwp_factor)
{
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
    if (atom_i < atom_numbers)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i];
        VECTOR frc_record = { 0., 0., 0. }, frc_enhancing_record = {0.0f, 0.0f, 0.0f};
        float virial_lj = 0., virial_enhancing = 0.0f;
        float energy_total = 0., energy_enhancing = 0.0f;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
        //float du_dlambda_enhancing = 0.0f;
        int atom_mark_i = atom_sys_mark[atom_i];
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_FEP_TYPE r2 = uint_crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
                int atom_pair_LJ_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                int atom_pair_LJ_type_B = Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                float AA = LJ_type_AA[atom_pair_LJ_type_A];
                float AB = LJ_type_AB[atom_pair_LJ_type_A];
                float BA = LJ_type_BA[atom_pair_LJ_type_B];
                float BB = LJ_type_BB[atom_pair_LJ_type_B];
                if (BA * AA != 0 || BA + AA == 0)
                {
                    if (need_force)
                    {
                        float frc_abs = lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        frc_enhancing_record = frc_enhancing_record + factor * frc_lin;
                        atomicAdd(frc + atom_j, -frc_lin);
                        atomicAdd(frc_enhancing + atom_j, -factor * frc_lin);
                    }
                    if (need_coulomb && (need_energy || need_virial))
                    {
                        float ene = Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                        energy_coulomb += ene;
                        energy_enhancing += factor * ene;
                    }
                    if (need_energy)
                    {
                        float ene = lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB);
                        energy_total += ene;
                        energy_enhancing += factor * ene;
                    }
                    if (need_virial)
                    {
                        float vir = lambda_ * Get_LJ_Virial(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Virial(r1, r2, dr_abs, BA, BB);
                        virial_lj += vir;
                        virial_enhancing += factor * vir;
                        virial_enhancing += factor * Get_Direct_Coulomb_Virial(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_abs, BA, BB) - Get_LJ_Energy(r1, r2, dr_abs, AA, AB);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
                else
                {
                    float sigma_A = Get_Soft_Core_Sigma(AA, AB, input_sigma_6, input_sigma_6_min);
                    float sigma_B = Get_Soft_Core_Sigma(BA, BB, input_sigma_6, input_sigma_6_min);
                    float dr_softcore_A = Get_Soft_Core_Distance(AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                    float dr_softcore_B = Get_Soft_Core_Distance(BB, BA, sigma_B, dr_abs, alpha, p, 1 - lambda);
                    if (need_force)
                    {
                        float frc_abs = lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs, dr_softcore_A, AA, AB)
                            + lambda * Get_Soft_Core_LJ_Force(r1, r2, dr_abs, dr_softcore_B, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = lambda_ * Get_Soft_Core_Direct_Coulomb_Force(r1, r2, dr_abs, dr_softcore_A, pme_beta)
                                + lambda * Get_Soft_Core_Direct_Coulomb_Force(r1, r2, dr_abs, dr_softcore_B, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        frc_enhancing_record = frc_enhancing_record + factor * frc_lin;
                        atomicAdd(frc + atom_j, -frc_lin);
                        atomicAdd(frc_enhancing + atom_j, -factor * frc_lin);
                    }
                    if (need_coulomb && (need_energy || need_virial))
                    {
                        float ene = lambda_ * Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A, pme_beta)
                            + lambda * Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B, pme_beta);
                        energy_coulomb += ene;
                        energy_enhancing += factor * ene;
                        virial_enhancing += factor * (lambda_ * Get_Soft_Core_Direct_Coulomb_Virial(r1, r2, dr_abs, dr_softcore_A, pme_beta) +
                            lambda * Get_Soft_Core_Direct_Coulomb_Virial(r1, r2, dr_abs, dr_softcore_B, pme_beta));
                    }
                    if (need_energy)
                    {
                        float ene = lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB)
                            + lambda * Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB);
                        energy_total += ene;
                        energy_enhancing += factor * ene;
                    }
                    if (need_virial)
                    {
                        float vir = lambda_ * Get_Soft_Core_LJ_Virial(r1, r2, dr_abs, dr_softcore_A, AA, AB)
                            + lambda * Get_Soft_Core_LJ_Virial(r1, r2, dr_abs, dr_softcore_B, BA, BB);
                        virial_lj += vir;
                        virial_enhancing += vir;
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB)
                            - Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB);
                        du_dlambda_lj += Get_Soft_Core_dU_dlambda(Get_LJ_Force(r1, r2, dr_softcore_A, AA, AB), sigma_A, dr_softcore_A, alpha, p, lambda)
                            - Get_Soft_Core_dU_dlambda(Get_LJ_Force(r1, r2, dr_softcore_B, BA, BB), sigma_B, dr_softcore_B, alpha, p, lambda_);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B, pme_beta)
                                - Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A, pme_beta);
                            du_dlambda_direct += Get_Soft_Core_dU_dlambda(Get_Direct_Coulomb_Force(r1, r2, dr_softcore_B, pme_beta), sigma_B, dr_softcore_B, alpha, p, lambda_)
                                - Get_Soft_Core_dU_dlambda(Get_Direct_Coulomb_Force(r1, r2, dr_softcore_A, pme_beta), sigma_A, dr_softcore_A, alpha, p, lambda);
                            du_dlambda_direct += lambda * Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_softcore_B, pme_beta)
                                + lambda_ * Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_softcore_A, pme_beta);
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record);
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_lj);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing);
        }
        if (need_du_dlambda)
        {
            Warp_Sum_To(atom_du_dlambda_lj, du_dlambda_lj);
            if (need_coulomb)
            {
                Warp_Sum_To(atom_du_dlambda_direct, du_dlambda_direct);
            }
        }
    }
}

static __device__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

static __global__ void SITS_Record_Ene_CUDA(float * ene_record, const float *enhancing_energy, const float pe_a, const float pe_b)
{
    *ene_record =  pe_a * *enhancing_energy + pe_b;
}

static __global__ void SITS_Update_gf_CUDA(const int kn, float *gf,
    const float *ene_record, const float *log_nk, const float *beta_k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        gf[i] = -beta_k[i] * ene_record[0] + log_nk[i];
    }
}

static __global__ void SITS_Update_gfsum_CUDA(const int kn, float *gfsum, const float *gf)
{
    if (threadIdx.x == 0)
    {
        gfsum[0] = -FLT_MAX;
    };
    for (int i = 0; i < kn; i = i + 1)
    {
        gfsum[0] = log_add_log(gfsum[0], gf[i]);
    }
}

static __global__ void SITS_Update_log_pk_CUDA(const int kn, float *log_pk,
    const float *gf, const float *gfsum, const int reset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        float gfi = gf[i];
        log_pk[i] = ((float)reset) * gfi + ((float)(1 - reset)) * log_add_log(log_pk[i], gfi - gfsum[0]);
    }
}

static __global__ void SITS_Update_log_mk_inverse_CUDA(const int kn,
    float *log_weight, float *log_mk_inverse, float *log_norm_old, 
    float *log_norm, const float *log_pk, const float *log_nk)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn - 1)
    {
        log_weight[i] = (log_pk[i] + log_pk[i + 1]) * 0.5;
        log_mk_inverse[i] = log_nk[i] - log_nk[i + 1];
        log_norm_old[i] = log_norm[i];
        log_norm[i] = log_add_log(log_norm[i], log_weight[i]);
        log_mk_inverse[i] = log_add_log(log_mk_inverse[i] + log_norm_old[i] - log_norm[i], log_pk[i + 1] - log_pk[i] + log_mk_inverse[i] + log_weight[i] - log_norm[i]);
    }
}

static __global__ void SITS_Update_log_nk_inverse_CUDA(const int kn,
    float *log_nk_inverse,     const float *log_mk_inverse)
{
    for (int i = 0; i < kn - 1; i++)
    {
        log_nk_inverse[i + 1] = log_nk_inverse[i] + log_mk_inverse[i];
    }
}

static __global__ void SITS_Update_nk_CUDA(const int kn,
    float *log_nk, float *nk, const float *log_nk_inverse)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn )
    {
        log_nk[i] = -log_nk_inverse[i];
        nk[i] = exp(log_nk[i]);
    }
}

static __global__ void SITS_For_Enhanced_Force_Calculate_NkExpBetakU_CUDA
(const int k_numbers, const float* beta_k, const float* nk, float* nkexpbetaku,
    const float* ene, const float beta0, const float pe_a, const float pe_b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < k_numbers)
    {
        nkexpbetaku[i] = nk[i] * expf(-(beta_k[i] - beta0) * (pe_a * ene[0] + pe_b));
    }
}

static __global__ void SITS_For_Enhanced_Force_Sum_Of_Above_CUDA
(const int k_numbers, const float *nkexpbetaku, const float *beta_k, float *sum_of_above)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        sum_of_above[0] = 0.;
    }
    __threadfence_block();
    float lin = 0.;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < k_numbers)
    {
        lin = lin + beta_k[i]*nkexpbetaku[i];
    }
    atomicAdd(sum_of_above, lin);
}

static __global__ void SITS_For_Enhanced_Force_Sum_Of_NkExpBetakU_CUDA
(const int k_numbers,const float *nkexpbetaku, float *sum_of_below)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        sum_of_below[0] = 0.;
    }
    __threadfence_block();
    float lin = 0.;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < k_numbers)
    {
        lin = lin + nkexpbetaku[i];
    }
    atomicAdd(sum_of_below, lin);
}

static __global__ void SITS_For_Enhanced_Force_Protein_Water_CUDA(const int atom_numbers, const int * atom_sys_mark,
    VECTOR * md_frc, const VECTOR * enhancing_frc, float *md_ene, const float *enhancing_ene, const float b_a,
    const int need_pressure, float *md_virial, const float *virial_enhancing, const float factor_minus_one)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        md_ene[0] += factor_minus_one * (enhancing_ene[0] - b_a);
        if (need_pressure)
        {
            md_virial[0] += factor_minus_one * virial_enhancing[0];
        }
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i = i + gridDim.x * blockDim.x)
    {
        md_frc[i] = md_frc[i] + factor_minus_one * enhancing_frc[i];
    }
}

static __global__ void ESITS_Get_Current_Fb(const float *enhancing_energy, float *factor,
    const float pe_a, const float pe_b, const float fb_bias)
{
    float ene = enhancing_energy[0];
    if (ene > pe_b)
    {
        factor[0] = 1 - pe_a / (ene - pe_b + pe_a) + fb_bias;
    }
    else
    {
        factor[0] = fb_bias;
    }
}

static void SITS_Get_Current_Fb(const int atom_numbers, const int *atom_sys_mark,
    const float *energy_enhancing,
    const int k_numbers,float *nkexpbetaku,
    const float *beta_k, const float *n_k, const float beta0,
    float *sum_a,float *sum_b,float *factor,
    const float pe_a, const float pe_b, const float pwwp_enhance_factor)
{
    CUDA_LAUNCH(SITS_For_Enhanced_Force_Calculate_NkExpBetakU_CUDA,
        dim3((k_numbers + 63) / 64), dim3(64),
        k_numbers, beta_k, n_k, nkexpbetaku, energy_enhancing, beta0, pe_a, pe_b);

    CUDA_LAUNCH(SITS_For_Enhanced_Force_Sum_Of_NkExpBetakU_CUDA,
        dim3((k_numbers + 127) / 128), dim3(128),
        k_numbers, nkexpbetaku, sum_b);

    CUDA_LAUNCH(SITS_For_Enhanced_Force_Sum_Of_Above_CUDA,
        dim3((k_numbers + 127) / 128), dim3(128),
        k_numbers, nkexpbetaku, beta_k, sum_a);
}

static __global__ void SITS_Get_Current_Fb_Final(float *sum_a, float *sum_b, const float beta_0, const float fb_bias, float *factor)
{
    factor[0] = sum_a[0] / sum_b[0] / beta_0 + fb_bias;
}

void CLASSIC_SITS_INFORMATION::Initial(CONTROLLER * controller, SITS_INFORMATION * sits)
{
    is_initialized = 1;
    sits_controller = sits;
    record_count = 0;
    fb_interval = 1;
    if (controller[0].Command_Exist(sits->module_name, "fb_interval"))
    {
        controller->Check_Int(sits->module_name, "fb_interval", "CLASSIC_SITS_INFORMATION::Initial");
        fb_interval = atoi(controller[0].Command(sits->module_name, "fb_interval"));
    }
    controller->printf("    SITS fb update interval set to %d\n", fb_interval);
    if (sits->sits_mode == SITS_MODE_EMPIRICAL)
    {
        if (controller[0].Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a", "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller[0].Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller[0].printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller[0].Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b", "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller[0].Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller[0].printf("    SITS_pe_b set to %f\n", pe_b);
        k_numbers = 0;
        nk_fix = 0;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode != SITS_MODE_OBSERVATION)
    {
        if (controller[0].Command_Exist(sits->module_name, "k_numbers"))
        {
            controller->Check_Int(sits->module_name, "k_numbers", "CLASSIC_SITS_INFORMATION::Initial");
            k_numbers = atoi(controller[0].Command("SITS_k_numbers"));
            if (k_numbers <= 0)
            {
                controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "CLASSIC_SITS_INFORMATION::Initial", "Reason:\n\tSITS k numbers cannot be smaller than 0\n");
            }
        }
        else
        {
            k_numbers = 40;
        }
        controller[0].printf("    k numbers is %d\n",k_numbers);
        Memory_Allocate();

        controller[0].printf("    Read %s temperature information.\n", sits->module_name);
        float *beta_k_tmp;
        Malloc_Safely((void**)&beta_k_tmp, sizeof(float)*k_numbers);
        if (controller[0].Command_Exist(sits->module_name, "T_low"))
        {
            if (!controller[0].Command_Exist(sits->module_name, "T_high"))
            {
                controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial", 
                    "Reason:\n\tSITS T high must be explicitly given with SITS T low in mdin\n");
            }
            controller->Check_Float(sits->module_name, "T_low", "CLASSIC_SITS_INFORMATION::Initial");
            controller->Check_Float(sits->module_name, "T_high", "CLASSIC_SITS_INFORMATION::Initial");
            float T_low = atof(controller[0].Command(sits->module_name, "T_low"));
            float T_high = atof(controller[0].Command(sits->module_name, "T_high"));
            float T_space = (T_high - T_low) / (k_numbers - 1);
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_k_tmp[i] = 1.0/(CONSTANT_kB * (T_low + T_space * i));
            }
        }
        else if (controller[0].Command_Exist(sits->module_name, "T"))
        {
            const char* char_pt = controller[0].Command(sits->module_name, "T");
            for (int i = 0; i < k_numbers; ++i)
            {
                float tmp_T;
                sscanf(char_pt, "%f", &tmp_T);
                if (i != k_numbers - 1)
                {
                    while(*char_pt != '/' && *char_pt != '\0')
                        ++char_pt;
                    if (*char_pt == '/')
                        ++char_pt;
                    if (*char_pt == '\0')
                    {
                        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "CLASSIC_SITS_INFORMATION::Initial", "Reason:\n\tthe number of temperatures SITS_T != SITS_k_numbers\n");
                    }
                }
                beta_k_tmp[i] = 1.0/(CONSTANT_kB * tmp_T);
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial", "Reason:\n\tSITS T must be explicitly given in mdin.\n");
        }
        cudaMemcpy(beta_k, beta_k_tmp, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);
        free(beta_k_tmp);
        if (controller[0].Command_Exist(sits->module_name, "record_interval"))
        {
            controller->Check_Int(sits->module_name, "record_interval", "CLASSIC_SITS_INFORMATION::Initial");
            record_interval = atoi(controller[0].Command(sits->module_name, "record_interval"));
        }
        else
        {
            record_interval = 1;
        }
        controller[0].printf("    SITS record interval set to %d\n", record_interval);

        if (controller[0].Command_Exist(sits->module_name, "update_interval"))
        {
            controller->Check_Int(sits->module_name, "update_interval", "CLASSIC_SITS_INFORMATION::Initial");
            update_interval = atoi(controller[0].Command(sits->module_name, "update_interval"));
        }
        else
        {
            update_interval = 100;
        }
        controller[0].printf("    SITS update interval set to %d\n", update_interval);

        if (controller[0].Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a", "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller[0].Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller[0].printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller[0].Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b", "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller[0].Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller[0].printf("    SITS_pe_b set to %f\n", pe_b);

        if (controller[0].Command_Exist(sits->module_name, "fb_bias"))
        {
            controller->Check_Float(sits->module_name, "fb_bias", "CLASSIC_SITS_INFORMATION::Initial");
            fb_bias = atof(controller[0].Command(sits->module_name, "fb_bias"));
        }
        else
        {
            fb_bias = 0.0;
        }
        controller[0].printf("    SITS_fb_bias set to %f\n", fb_bias);
    
        reset = 1;

        int nk_rest;
        if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_rest = 0;
        }
        else
        {
            nk_rest = 1;
        }
        if (controller[0].Command_Exist(sits->module_name, "nk_rest"))
        {
            nk_rest = controller->Get_Bool(sits->module_name, "nk_rest", "CLASSIC_SITS_INFORMATION::Initial");
        }
        float * beta_lin;
        Malloc_Safely((void**)&beta_lin, sizeof(float)*k_numbers);

        for (int i = 0; i < k_numbers; ++i)
            beta_lin[i] = -FLT_MAX;

        cudaMemcpy(log_norm_old, beta_lin, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(log_norm, beta_lin, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);
        cudaMemset(log_nk_inverse, 0, sizeof(float)*k_numbers);

        if (nk_rest == 0)
        {
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_lin[i] = 0.0;
            }
        }
        else
        {
            FILE * nk_read_file;
            if (controller[0].Command_Exist(sits->module_name, "nk_in_file"))
            {
                controller[0].printf("    Read Nk from %s\n", controller[0].Command(sits->module_name, "nk_in_file"));
                Open_File_Safely(&nk_read_file, controller[0].Command(sits->module_name, "nk_in_file"),"r");
                for (int i = 0; i < k_numbers; ++i)
                {
                    int retval = fscanf(nk_read_file, "%f", beta_lin + i);
                    beta_lin[i] = logf(beta_lin[i]);
                }
            }
            else
            {
                controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial", "Reason:\n\tSITS_nk_in_file must be given when SITS_nk_rest = 1 or SITS_mode = production\n");
            }
        }
        cudaMemcpy(log_nk, beta_lin, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);

        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = -beta_lin[i];
        }
        cudaMemcpy(log_nk_inverse, beta_lin, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);
        
        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = expf(-beta_lin[i]);
        }
        cudaMemcpy(Nk, beta_lin, sizeof(float)*k_numbers, cudaMemcpyHostToDevice);

        free(beta_lin);
        Reset_List(factor, 1.0, 1);
        
        if (controller[0].Command_Exist(sits->module_name, "nk_fix"))
        {
            nk_fix = controller->Get_Bool(sits->module_name, "nk_fix", "CLASSIC_SITS_INFORMATION::Initial");
        }
        else if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_fix = 0;
        }
        else
        {
            nk_fix = 1;
        }
        controller[0].printf("    SITS nk fix is: %d\n", nk_fix);
        if (nk_fix == 0)
        {
            if (controller[0].Command_Exist(sits->module_name, "nk_rest_file"))
            {
                strcpy(nk_rest_file_name, controller[0].Command(sits->module_name, "nk_rest_file"));
            }
            else
            {
                strcpy(nk_rest_file_name, sits->module_name);
                strcat(nk_rest_file_name, "_nk_rest.txt");
            }
            controller[0].printf("    Restart Nk will be written in %s\n", nk_rest_file_name);
            if (controller[0].Command_Exist(sits->module_name, "nk_traj_file"))
            {
                Open_File_Safely(&nk_traj_file, controller->Command(sits->module_name, "nk_traj_file"), "wb");
                controller->printf("    Trajectory Nk will be written in %s\n",  controller->Command(sits->module_name, "nk_traj_file"));
            }
            else
            {
                Open_File_Safely(&nk_traj_file, "SITS_nk_traj.dat", "wb");
                controller->printf("    Trajectory Nk will be written in %s\n", "SITS_nk_traj.dat");
            }
        }
    }
}


void CLASSIC_SITS_INFORMATION::Memory_Allocate()
{
    Cuda_Malloc_Safely((void**)&ene_recorded, sizeof(float));
    Cuda_Malloc_Safely((void**)&gf, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&gfsum, sizeof(float));
    Cuda_Malloc_Safely((void**)&log_weight, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_mk_inverse, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_norm_old, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_norm, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_pk, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_nk_inverse, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&log_nk, sizeof(float)*k_numbers);

    Cuda_Malloc_Safely((void**)&beta_k, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&NkExpBetakU, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&Nk, sizeof(float)*k_numbers);
    Cuda_Malloc_Safely((void**)&sum_a, sizeof(float));
    Cuda_Malloc_Safely((void**)&sum_b, sizeof(float));
    Cuda_Malloc_Safely((void**)&factor, sizeof(float));

    Malloc_Safely((void**)&nk_record_cpu, sizeof(float)*k_numbers);
    Malloc_Safely((void**)&log_norm_record_cpu, sizeof(float)*k_numbers);
}


void CLASSIC_SITS_INFORMATION::Clear()
{
    cudaFree(ene_recorded);
    cudaFree(gf);
    cudaFree(gfsum);
    cudaFree(log_weight);
    cudaFree(log_mk_inverse);
    cudaFree(log_norm_old);
    cudaFree(log_norm);
    cudaFree(log_pk);
    cudaFree(log_nk_inverse);
    cudaFree(log_nk);
    cudaFree(beta_k);
    cudaFree(NkExpBetakU);
    cudaFree(Nk);
    cudaFree(sum_a);
    cudaFree(sum_b);
    cudaFree(factor);

    free(nk_record_cpu);
    free(log_norm_record_cpu);

    fclose(nk_traj_file);
}


void CLASSIC_SITS_INFORMATION::SITS_Record_Ene()
{
    CUDA_LAUNCH(SITS_Record_Ene_CUDA, dim3(1), dim3(1),
        ene_recorded, sits_controller->pw_select.select_energy[0], pe_a, pe_b);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gf()
{
    CUDA_LAUNCH(SITS_Update_gf_CUDA,
        dim3(ceilf((float)k_numbers / 32.0f)), dim3(32),
        k_numbers, gf, ene_recorded, log_nk, beta_k);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gfsum()
{
    CUDA_LAUNCH(SITS_Update_gfsum_CUDA, dim3(1), dim3(1), k_numbers, gfsum, gf);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_pk()
{
    CUDA_LAUNCH(SITS_Update_log_pk_CUDA,
        dim3(ceilf((float)k_numbers / 32.0f)), dim3(32),
        k_numbers, log_pk, gf, gfsum, reset);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_mk_inverse()
{
    CUDA_LAUNCH(SITS_Update_log_mk_inverse_CUDA,
        dim3(ceilf((float)k_numbers / 32.0f)), dim3(32),
        k_numbers, log_weight, log_mk_inverse, log_norm_old, log_norm, log_pk, log_nk);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_nk_inverse()
{
    CUDA_LAUNCH(SITS_Update_log_nk_inverse_CUDA, dim3(1), dim3(1), k_numbers,
        log_nk_inverse, log_mk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_nk()
{
    CUDA_LAUNCH(SITS_Update_nk_CUDA,
        dim3(ceilf((float)k_numbers / 32.0f)), dim3(32),
        k_numbers, log_nk, Nk, log_nk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Fb(float beta_0, int step)
{
    if (!is_initialized || sits_controller->sits_mode == SITS_MODE_OBSERVATION || step % fb_interval != 0)
    {
        return;
    }
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Get_Current_Fb
            (sits_controller->atom_numbers, sits_controller->atom_sys_mark, 
             sits_controller->pw_select.select_energy[0],
            k_numbers, NkExpBetakU,
            beta_k, Nk, beta_0,
            sum_a, sum_b, factor,
            pe_a, pe_b, sits_controller->pwwp_enhance_factor);
        CUDA_LAUNCH(SITS_Get_Current_Fb_Final, dim3(1), dim3(1),
            sum_a, sum_b, beta_0, fb_bias, factor);
        cudaMemcpy(&sits_controller->h_factor, factor, sizeof(float), cudaMemcpyDeviceToHost);
    }
    else
    {
        CUDA_LAUNCH(ESITS_Get_Current_Fb, dim3(1), dim3(1),
            sits_controller->pw_select.select_energy[0], factor, pe_a,
            pe_b, fb_bias);
        cudaMemcpy(&sits_controller->h_factor, factor, sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Common(const float beta)
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Record_Ene();
        SITS_Update_gf();
        SITS_Update_gfsum();
        SITS_Update_log_pk();
        reset = 0;
        record_count++;
    }
}


void CLASSIC_SITS_INFORMATION::SITS_Update_Nk()
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Update_log_mk_inverse();
        SITS_Update_log_nk_inverse();
        SITS_Update_nk();

        record_count = 0;
        reset = 1;

        SITS_Write_Nk_Norm();
    }

}


void CLASSIC_SITS_INFORMATION::SITS_Write_Nk_Norm()
{
    cudaMemcpy(nk_record_cpu, Nk, sizeof(float)*k_numbers, cudaMemcpyDeviceToHost);
    fwrite(nk_record_cpu, sizeof(float), k_numbers, nk_traj_file);

    Open_File_Safely(&nk_rest_file, nk_rest_file_name, "w");
    for (int i = 0; i < k_numbers; ++i)
    {
        fprintf(nk_rest_file, "%e ", nk_record_cpu[i]);
    }
    fclose(nk_rest_file);
}

void SITS_INFORMATION::Initial(CONTROLLER * controller, int atom_numbers_, const char *given_module_name)
{
    if (given_module_name == NULL)
    {
        strcpy(module_name, "SITS");
        strcpy(print_aa_kab_name, "SITS");
        strcpy(print_bias_name, "SITS");
        strcpy(print_fb_name, "SITS");
    }
    else
    {
        strcpy(module_name, given_module_name);
        strcpy(print_aa_kab_name, given_module_name);
        strcpy(print_bias_name, given_module_name);
        strcpy(print_fb_name, given_module_name);
    }
    strcat(print_aa_kab_name, "_AA_kAB");
    strcat(print_bias_name, "_bias");
    strcat(print_fb_name, "_fb");
    if (controller[0].Command_Exist(module_name, "mode"))
    {
        if (controller->Command_Choice(module_name, "mode", "observation"))
        {
            controller[0].printf("START INITIALIZING %s\n    %s mode = observation\n", module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_OBSERVATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "iteration"))
        {
            controller[0].printf("START INITIALIZING %s\n    %s mode = iteration\n", module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_ITERATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "production"))
        {
            controller[0].printf("START INITIALIZING %s\n    %s mode = production\n", module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_PRODUCTION;
        }
        else if (controller->Command_Choice(module_name, "mode", "empirical"))
        {
            controller[0].printf("START INITIALIZING %s\n    %s mode = empirical\n", module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_EMPIRICAL;
        }
        else
        {
            return;
        }
        atom_numbers = atom_numbers_;
        controller[0].printf("\tAtom numbers is %d\n", atom_numbers);
        Memory_Allocate();

        pw_select.Initial();
        pw_select.Add_One_Energy(atom_numbers);
        pw_select.Add_One_Force(atom_numbers);
        pw_select.Add_One_Virial(atom_numbers);

        if (controller[0].Command_Exist(module_name, "cross_enhance_factor"))
        {
            controller->Check_Float(module_name, "cross_enhance_factor", "SITS_INFORMATION::Initial");
            pwwp_enhance_factor = atof(controller[0].Command(module_name, "cross_enhance_factor"));
        }
        else
        {
            pwwp_enhance_factor = 0.5;
        }
        controller[0].printf("\tpwwp enhance factor set to %f\n", pwwp_enhance_factor);


        if (controller[0].Command_Exist(module_name, "atom_in_file") || controller[0].Command_Exist(module_name, "atom_numbers"))
        {
            controller[0].printf("    Set atom atribution information\n");
            int * atom_sys_mark_cpu;
            Malloc_Safely((void**)&atom_sys_mark_cpu, sizeof(int)*atom_numbers);
            if (controller[0].Command_Exist(module_name, "atom_in_file"))
            {
                for (int i = 0; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 1;
                }
                controller->printf("    reading %s_atom_in_file\n", module_name);
                FILE *fr = NULL;
                int temp_atom;
                Open_File_Safely(&fr, controller->Command(module_name, "atom_in_file"), "r");
                while (fscanf(fr, "%d", &temp_atom) != EOF)
                {
                    atom_sys_mark_cpu[temp_atom] = 0;
                }
                fclose(fr);
            }
            else
            {
                controller->Check_Int(module_name, "atom_numbers", "SITS_INFORMATION::Initial");
                int protein_numbers = atoi(controller[0].Command(module_name, "atom_numbers"));
                for (int i = 0; i < protein_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 0;
                }
                for (int i = protein_numbers; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 1;
                }
            }
            cudaMemcpy(atom_sys_mark, atom_sys_mark_cpu, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
            free(atom_sys_mark_cpu);
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "SITS_INFORMATION::Initial",
                "Reason:\n\tAtom information must be given in the form of SITS_atom_in_file or SITS_atom_numbers\n");
        }

        
        classic_sits.Initial(controller, this);

        need_potential = 0;
        h_factor = 1.0f;

        controller->Step_Print_Initial(print_aa_kab_name, "%.2f");
        controller->Step_Print_Initial(print_bias_name, "%.4f");
        controller->Step_Print_Initial(print_fb_name, "%.4f");

        controller[0].printf("END INTIALIZING %s\n\n", module_name);
    }
    else
    {
        is_initialized = 0;
        return;
    }
}

void SITS_INFORMATION::Memory_Allocate()
{
    Cuda_Malloc_Safely((void**)&atom_sys_mark, sizeof(float)*atom_numbers);
}

void SITS_INFORMATION::Clear()
{
    cudaFree(atom_sys_mark);
    pw_select.Clear();
    
    classic_sits.Clear();
}

void SITS_INFORMATION::Get_Energy(LENNARD_JONES_INFORMATION* lj_info, LJ_SOFT_CORE* lj_soft_info, const int atom_numbers,
    const ATOM_GROUP* nl, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, const float pme_beta, float* coulomb_atom_ene, const float* extra_energy)
{
    if (this->is_initialized)
    {
        cudaMemset(pw_select.select_atom_energy[0], 0, sizeof(float) * atom_numbers);
        if (lj_info->is_initialized)
        {
            CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
                dim3((atom_numbers + 1023) / 1024), dim3(1024),
                atom_numbers, uint_crd, lj_info->uint_crd_with_LJ, charge);
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * atom_numbers);
            cudaMemset(lj_info->d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_CUDA<false, true, false, true>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof,
                    lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark, lj_info->cutoff,
                    NULL, NULL, pme_beta, lj_info->d_LJ_energy_atom, pw_select.select_atom_energy[0],
                    NULL, NULL, coulomb_atom_ene, pwwp_enhance_factor);
            Sum_Of_List(lj_info->d_LJ_energy_atom, lj_info->d_LJ_energy_sum, atom_numbers);
            CUDA_LAUNCH(device_add, dim3(1), dim3(1),
                lj_info->d_LJ_energy_sum, lj_info->long_range_factor / lj_info->volume);
            cudaMemcpy(&lj_info->h_LJ_energy_sum, lj_info->d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
        }
        if (lj_soft_info->is_initialized)
        {
            CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
                dim3((this->atom_numbers + 1023) / 1024), dim3(1024),
                this->atom_numbers, uint_crd, lj_soft_info->uint_crd_with_LJ, charge);
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * atom_numbers);
            cudaMemset(lj_soft_info->d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, true, false, true, false>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_soft_info->uint_crd_with_LJ, lj_soft_info->uint_dr_to_dr_cof, atom_sys_mark,
                    lj_soft_info->d_LJ_AA, lj_soft_info->d_LJ_AB, lj_soft_info->d_LJ_BA, lj_soft_info->d_LJ_BB, lj_soft_info->cutoff,
                    NULL, NULL, pme_beta, lj_soft_info->d_LJ_energy_atom, pw_select.select_atom_energy[0],
                    NULL, NULL, coulomb_atom_ene, NULL, NULL, NULL,
                    lj_soft_info->lambda, lj_soft_info->alpha, lj_soft_info->p, lj_soft_info->sigma_6, lj_soft_info->sigma_6_min, pwwp_enhance_factor);
            Sum_Of_List(lj_soft_info->d_LJ_energy_atom, lj_soft_info->d_LJ_energy_sum, atom_numbers);
            CUDA_LAUNCH(device_add, dim3(1), dim3(1),
                lj_soft_info->d_LJ_energy_sum, lj_soft_info->long_range_factor / lj_soft_info->volume);
            cudaMemcpy(&lj_soft_info->h_LJ_energy_sum, lj_soft_info->d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
        }
        Sum_Of_List(pw_select.select_atom_energy[0], pw_select.select_energy[0], atom_numbers);
        if (extra_energy != NULL)
        {
            CUDA_LAUNCH(device_add, dim3(1), dim3(1), pw_select.select_energy[0], extra_energy);
        }
        cudaMemcpy(&h_enhancing_energy, pw_select.select_energy[0], sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void SITS_INFORMATION::Reset_Force_Energy(int md_need_potential)
{
    if (!is_initialized)
        return;
    need_potential = 1;
    cudaMemset(pw_select.select_atom_energy[0], 0, sizeof(float)*atom_numbers);
    cudaMemset(pw_select.select_energy[0], 0, sizeof(float));
    cudaMemset(pw_select.select_force[0], 0, sizeof(VECTOR) * atom_numbers);
    cudaMemset(pw_select.select_atom_virial[0], 0, sizeof(float) * atom_numbers);
    cudaMemset(pw_select.select_virial[0], 0, sizeof(float));
}

void SITS_INFORMATION::Update_And_Enhance(const int step, float* d_total_potential, int need_pressure, float* d_total_virial, VECTOR* frc, float beta0)
{
    if (!is_initialized)
        return;
    Sum_Of_List(pw_select.select_atom_energy[0], pw_select.select_energy[0], atom_numbers);
    if (need_pressure)
    {
        Sum_Of_List(pw_select.select_atom_virial[0], pw_select.select_virial[0], atom_numbers);
    }
    if (sits_mode != SITS_MODE_OBSERVATION && !classic_sits.nk_fix && step % classic_sits.record_interval == 0)
    {
        classic_sits.SITS_Update_Common(beta0);
        if (classic_sits.record_count % classic_sits.update_interval == 0)
        {
            classic_sits.SITS_Update_Nk();
        }
    }
    if (sits_mode != SITS_MODE_OBSERVATION)
    {
        classic_sits.SITS_Update_Fb(beta0, step);
    }
    CUDA_LAUNCH(SITS_For_Enhanced_Force_Protein_Water_CUDA, dim3(1), dim3(128),
        atom_numbers, atom_sys_mark, frc, pw_select.select_force[0],
        d_total_potential, pw_select.select_energy[0], classic_sits.pe_b / classic_sits.pe_a, need_pressure, d_total_virial, pw_select.select_virial[0], h_factor - 1);
}

void SITS_INFORMATION::SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, LENNARD_JONES_INFORMATION* lj_info, 
    VECTOR* md_frc, const ATOM_GROUP* nl, const float cutoff, const float pme_beta, float* atom_energy, const int need_pressure, float* atom_virial, float *coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
            dim3((this->atom_numbers + 1023) / 1024), dim3(1024),
            this->atom_numbers, uint_crd, lj_info->uint_crd_with_LJ, charge);
        if (need_potential && !need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, false, true>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof,
                    lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, pwwp_enhance_factor);
        }
        else if (need_potential && need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_CUDA<true, true, true, true>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof,
                    lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, pwwp_enhance_factor);
        }
        else if (!need_potential && need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, true, true>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof,
                    lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, pwwp_enhance_factor);
        }
        else
        {
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_CUDA<true, false, false, true>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof,
                lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark, cutoff,
                md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, pwwp_enhance_factor);
        }
    }
}


void SITS_INFORMATION::SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, LJ_SOFT_CORE* lj_info, 
    VECTOR* md_frc, const ATOM_GROUP* nl, const float cutoff, const float pme_beta, float* atom_energy, const int need_pressure, float* atom_virial, float* coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        CUDA_LAUNCH(Copy_Crd_And_Charge_To_New_Crd,
            dim3((this->atom_numbers + 1023) / 1024), dim3(1024),
            this->atom_numbers, uint_crd, lj_info->uint_crd_with_LJ, charge);
        if (need_potential && !need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, false, true, false>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof, atom_sys_mark,
                    lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, NULL, NULL, NULL,
                    lj_info->lambda, lj_info->alpha, lj_info->p, lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
        }
        else if (need_potential && need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, true, true, false>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof, atom_sys_mark,
                    lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, NULL, NULL, NULL,
                    lj_info->lambda, lj_info->alpha, lj_info->p, lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
        }
        else if (!need_potential && need_pressure)
        {
            cudaMemset(coulomb_atom_ene, 0, sizeof(float) * this->atom_numbers);
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, true, true, false>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof, atom_sys_mark,
                    lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, NULL, NULL, NULL,
                    lj_info->lambda, lj_info->alpha, lj_info->p, lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
        }
        else
        {
            CUDA_LAUNCH((Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, false, true, false>),
                dim3((atom_numbers + 31) / 32), dim3(32, 32),
                atom_numbers, nl,
                    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof, atom_sys_mark,
                    lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff,
                    md_frc, pw_select.select_force[0], pme_beta, atom_energy, pw_select.select_atom_energy[0],
                    atom_virial, pw_select.select_atom_virial[0], coulomb_atom_ene, NULL, NULL, NULL,
                    lj_info->lambda, lj_info->alpha, lj_info->p, lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
        }
    }
}

void SITS_INFORMATION::Step_Print(CONTROLLER* controller, const float beta0, LENNARD_JONES_INFORMATION* lj_info, LJ_SOFT_CORE* lj_soft_info, const int atom_numbers,
    const UNSIGNED_INT_VECTOR* uint_crd, const ATOM_GROUP* d_nl, const float beta, const float* charge, float* pme_direct_energy, int step, const float* extra_energy)
{
    if (!is_initialized)
        return;
    Get_Energy(lj_info, lj_soft_info, atom_numbers, d_nl, uint_crd, charge, beta, pme_direct_energy, extra_energy);
    controller->Step_Print(print_aa_kab_name, h_enhancing_energy);
    if (lj_info->is_initialized)
    {
        controller->Step_Print("LJ", lj_info->h_LJ_energy_sum, true);
    }
    if (lj_soft_info->is_initialized)
    {
        controller->Step_Print("LJ_soft", lj_soft_info->h_LJ_energy_sum, true);
    }
    if (sits_mode != SITS_MODE_OBSERVATION)
    {
        classic_sits.SITS_Update_Fb(beta0, step + 1);
        float b;
        if (sits_mode == SITS_MODE_EMPIRICAL)
        {
            if (h_enhancing_energy > classic_sits.pe_b)
            {
                controller->Step_Print(print_bias_name, -classic_sits.pe_a * logf(h_enhancing_energy - classic_sits.pe_b + classic_sits.pe_a), true);
            }
            else
            {
                controller->Step_Print(print_bias_name, -h_enhancing_energy, true);
            }
        }
        else
        {
            cudaMemcpy(&b, classic_sits.sum_b, sizeof(float), cudaMemcpyDeviceToHost);
            controller->Step_Print(print_bias_name, -logf(b) / beta0 / classic_sits.pe_a +
                classic_sits.fb_bias * (h_enhancing_energy + classic_sits.pe_b / classic_sits.pe_a), true);
        }
        controller->Step_Print(print_fb_name, h_factor);
    }
}

void SELECT::Initial()
{
    select_atom_energy.clear();
    select_energy.clear();
    select_force.clear();
    select_atom_virial.clear();
    select_virial.clear();
}

int SELECT::Add_One_Energy(int atom_numbers)
{
    float * tmp_atom_energy;
    float * tmp_energy;
    Cuda_Malloc_Safely((void**)&tmp_atom_energy, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&tmp_energy, sizeof(float));
    select_atom_energy.push_back(tmp_atom_energy);
    select_energy.push_back(tmp_energy);
    return select_energy.size() - 1;
}

int SELECT::Add_One_Force(int atom_numbers)
{
    VECTOR * tmp_force;
    Cuda_Malloc_Safely((void**)&tmp_force, sizeof(VECTOR)*atom_numbers);
    select_force.push_back(tmp_force);
    return (select_force.size() - 1);
}

int SELECT::Add_One_Virial(int atom_numbers)
{
    float * tmp_atom_virial;
    float * tmp_virial;
    Cuda_Malloc_Safely((void**)&tmp_atom_virial, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&tmp_virial, sizeof(float)*atom_numbers);
    select_atom_virial.push_back(tmp_atom_virial);
    select_virial.push_back(tmp_virial);
    return select_virial.size() - 1;
}

void SELECT::Clear()
{
    int size = select_atom_energy.size();
    for (int i = 0; i < size; ++i)
    {
        cudaFree(select_atom_energy[i]);
        cudaFree(select_energy[i]);
    }
    size = select_force.size();
    for (int i = 0; i < size; ++i)
    {
        cudaFree(select_force[i]);
        cudaFree(select_force[i]);
    }
    size = select_virial.size();
    for (int i = 0; i < size; ++i)
    {
        cudaFree(select_virial[i]);
        cudaFree(select_virial[i]);
    }
}
