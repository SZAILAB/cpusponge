#include "restrain_cv.h"

static __global__ void restrain_energy(float* cv_value, float weight, float reference, float period, float *energy)
{
    float dCV = cv_value[0] - reference;
    if (period > 0)
    {
        dCV = dCV - floorf(dCV / period + 0.5) * period;
    }
    energy[0] = weight * dCV * dCV;
}

static __global__ void restrain_force_and_energy(int atom_numbers, float *cv_value, VECTOR *crd_grads, float weight, float reference, float period, VECTOR* frc, float *energy)
{
    float dCV = cv_value[0] - reference;
    if (period > 0)
    {
        dCV = dCV - floorf(dCV / period + 0.5) * period;
    }
    //能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        atomicAdd(energy, weight * dCV * dCV);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
}

static __global__ void restrain_force_and_energy_and_virial(int atom_numbers, float *cv_value, VECTOR *crd_grads, VECTOR *box_grads, VECTOR box_length,
    float weight, float reference, float period, VECTOR* frc, float *energy, float *virial)
{
    float dCV = cv_value[0] - reference;
    if (period > 0)
    {
        dCV = dCV - floorf(dCV / period + 0.5) * period;
    }
    //能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        atomicAdd(energy, weight * dCV * dCV);
        atomicAdd(virial, -2 * weight * dCV * (box_grads[0] * box_length));
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
}

void RESTRAIN_CV::Initial(CONTROLLER* controller, COLLECTIVE_VARIABLE_CONTROLLER* cv_controller)
{
    strcpy(this->module_name, "restrain_cv");
    controller->printf("START INITIALIZING RESTRAIN CV:\n");
    cv_list = cv_controller->Ask_For_CV("restrain", 0);
    CV_numbers = cv_list.size();
    if (CV_numbers)
    {
        weight = cv_controller->Ask_For_Float_Parameter("restrain", "weight", cv_list.size(), 1, true, 0, 0, "kcal/mol/CV^2");
        reference = cv_controller->Ask_For_Float_Parameter("restrain", "reference", cv_list.size(), 1, true, 0, 0, "CV");
        period = cv_controller->Ask_For_Float_Parameter("restrain", "period", cv_list.size(), 1, false, 0, 0, "CV");
        start_step = cv_controller->Ask_For_Int_Parameter("restrain", "start_step", cv_list.size(), 1, false, 0, 0);
        max_step = cv_controller->Ask_For_Int_Parameter("restrain", "max_step", cv_list.size(), 1, false, 0, 0);
        reduce_step = cv_controller->Ask_For_Int_Parameter("restrain", "reduce_step", cv_list.size(), 1, false, 0, 0);
        stop_step = cv_controller->Ask_For_Int_Parameter("restrain", "stop_step", cv_list.size(), 1, false, 0, 0);
        for (int i = 0; i < CV_numbers; i++)
        {
            StringMap error_map = { {"i", std::to_string(i)}, 
                                    {"start_step", std::to_string(start_step[i])},
                                    {"max_step", std::to_string(max_step[i])},
                                    {"reduce_step", std::to_string(reduce_step[i])} ,
                                    {"stop_step", std::to_string(stop_step[i])} };
            if (max_step[i] != 0 && max_step[i] < start_step[i])
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format("Reason:\n\tThe max step (%max_step%) of %i%-th CV is smaller than \
the start step (%start_step%)", error_map).c_str());
            }
            if (reduce_step[i] != 0 && reduce_step[i] < max_step[i])
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format("Reason:\n\tThe reducing step (%reduce_step%) of %i%-th CV is smaller than \
the max step (%max_step%)", error_map).c_str());
            }
            if (reduce_step[i] != 0 && stop_step[i] < reduce_step[i])
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format("Reason:\n\tThe stop step (%stop_step%) of %i%-th CV is smaller than \
the reduce step (%reduce_step%)", error_map).c_str());
            }
            if (stop_step[i] != 0 && reduce_step[i] == 0)
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format("Reason:\n\tThe reduce step (%reduce_step%) of %i%-th CV should be non-zero \
when the stop step is not zero (%stop_step%)", error_map).c_str());
            }
        }
        Malloc_Safely((void**)&h_ene, sizeof(float) * CV_numbers);
        Cuda_Malloc_Safely((void**)&d_ene, sizeof(float) * CV_numbers);
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_initialized = 1;
        cudaOccupancyMaxPotentialBlockSize(&cuda_grid_size, &cuda_block_size, restrain_force_and_energy, 0, 0);
        cuda_grid_size /= CV_numbers;
        if (cuda_grid_size < 1)
            cuda_grid_size = 1;
        printf("    device launch parameters: grid size %d and block size %d\n", cuda_grid_size, cuda_block_size);
        controller->printf("END INITIALIZING RESTRAIN CV\n\n");
    }
    else
    {
        controller->printf("RESTRAIN CV IS NOT INITIALIZED\n\n");
    }
}

float RESTRAIN_CV::Get_Energy(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step)
{
    if (!is_initialized)
        return NAN;
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    for (int i = 0; i < CV_numbers; i++)
    {
            if (step < start_step[i] || (stop_step[i] > 0 && step >= stop_step[i]))
                continue;
            float local_weight = weight[i];
            if (step < max_step[i] && max_step[i] > start_step[i])
                local_weight *= (float)(step - start_step[i]) / (max_step[i] - start_step[i]);
            if (reduce_step[i] != 0 && step > reduce_step[i] && reduce_step[i] < stop_step[i])
                local_weight *= (float)(stop_step[i] - step) / (stop_step[i] - reduce_step[i]);
        cv = cv_list[i];
            cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, CV_NEED_GPU_VALUE, step);
            CUDA_LAUNCH_STREAM(restrain_energy, dim3(1), dim3(1), 0, cv->cuda_stream,
                cv->d_value, local_weight, reference[i], period[i], d_ene + i);
    }
    cudaMemcpy(h_ene, d_ene, sizeof(float) * CV_numbers, cudaMemcpyDeviceToHost);
    float ret = 0;
    for (int i = 0; i < CV_numbers; i++)
    {
        ret += h_ene[i];
    }
    return ret;
}

void RESTRAIN_CV::Restraint(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step,
    float *d_ene, float *d_virial, VECTOR *frc, int need_potential, int need_pressure)
{
    if (!is_initialized)
        return;
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    int need = CV_NEED_CRD_GRADS | CV_NEED_GPU_VALUE;
    if (need_pressure)
        need |= CV_NEED_BOX_GRADS;
    for (int i = 0; i < CV_numbers; i++)
    {
        if (step < start_step[i] || (stop_step[i] > 0 && step >= stop_step[i]))
            continue;
        float local_weight = weight[i];
        if (step < max_step[i] && max_step[i] > start_step[i])
            local_weight *= (float)(step - start_step[i]) / (max_step[i] - start_step[i]);
        if (reduce_step[i] != 0 && step > reduce_step[i] && reduce_step[i] < stop_step[i])
            local_weight *= (float)(stop_step[i] - step) / (stop_step[i] - reduce_step[i]);
        cv = cv_list[i];
        cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, need, step);
        if (!need_pressure)
            CUDA_LAUNCH_STREAM(restrain_force_and_energy, cuda_grid_size, cuda_block_size, 0, cv->cuda_stream,
                atom_numbers, cv->d_value, cv->crd_grads, local_weight, reference[i], period[i], frc, d_ene);
        else
            CUDA_LAUNCH_STREAM(restrain_force_and_energy_and_virial, cuda_grid_size, cuda_block_size, 0, cv->cuda_stream,
                atom_numbers, cv->d_value, cv->crd_grads, cv->box_grads, box_length,
                local_weight, reference[i], period[i], frc, d_ene, d_virial);
    }
}
