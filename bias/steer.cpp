#include "steer.h"

static __global__ void steer_energy(float* cv_value, float weight, float *energy)
{
	energy[0] = weight * cv_value[0];
}

static __global__ void steer_force_and_energy(int atom_numbers, float *cv_value, VECTOR *crd_grads, float weight, VECTOR* frc, float *energy)
{
	//能量只第一个线程算
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		atomicAdd(energy, weight * cv_value[0]);
	}
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
	{
		VECTOR force = -weight * crd_grads[i];
		atomicAdd(&frc[i].x, force.x);
		atomicAdd(&frc[i].y, force.y);
		atomicAdd(&frc[i].z, force.z);
	}
}

static __global__ void steer_force_and_energy_and_virial(int atom_numbers, float *cv_value, VECTOR *crd_grads, VECTOR *box_grads, VECTOR box_length,
	float weight, VECTOR* frc, float *energy, float *virial)
{
	//能量只第一个线程算
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		atomicAdd(energy, weight * cv_value[0]);
		atomicAdd(virial, -weight * (box_grads[0] * box_length));
	}
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < atom_numbers; i += blockDim.x * gridDim.x)
	{
		VECTOR force = -weight * crd_grads[i];
		atomicAdd(&frc[i].x, force.x);
		atomicAdd(&frc[i].y, force.y);
		atomicAdd(&frc[i].z, force.z);
	}
}

void STEER_CV::Initial(CONTROLLER* controller, COLLECTIVE_VARIABLE_CONTROLLER* cv_controller)
{
	strcpy(this->module_name, "steer_cv");
	controller->printf("START INITIALIZING STEER CV:\n");
	cv_list = cv_controller->Ask_For_CV("steer", 0);
	CV_numbers = cv_list.size();
	if (CV_numbers)
	{
		weight = cv_controller->Ask_For_Float_Parameter("steer", "weight", cv_list.size(), 1, true, 0, 0, "kcal/mol/CV^2");
		Malloc_Safely((void**)&h_ene, sizeof(float) * CV_numbers);
		Cuda_Malloc_Safely((void**)&d_ene, sizeof(float) * CV_numbers);
		controller->Step_Print_Initial(this->module_name, "%.2f");
		is_initialized = 1;
		cudaOccupancyMaxPotentialBlockSize(&cuda_grid_size, &cuda_block_size, steer_force_and_energy, 0, 0);
		cuda_grid_size /= CV_numbers;
		if (cuda_grid_size < 1)
			cuda_grid_size = 1;
		printf("    device launch parameters: grid size %d and block size %d\n", cuda_grid_size, cuda_block_size);
		controller->printf("END INITIALIZING STEER CV\n\n");
	}
	else
	{
		controller->printf("STEER CV IS NOT INITIALIZED\n\n");
	}
}

float STEER_CV::Get_Energy(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step)
{
	if (!is_initialized)
		return NAN;
	COLLECTIVE_VARIABLE_PROTOTYPE* cv;
	for (int i = 0; i < CV_numbers; i++)
	{
		cv = cv_list[i];
		cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, CV_NEED_GPU_VALUE, step);
		CUDA_LAUNCH_STREAM(steer_energy, dim3(1), dim3(1), 0, cv->cuda_stream,
			cv->d_value, weight[i], d_ene + i);
	}
	cudaMemcpy(h_ene, d_ene, sizeof(float) * CV_numbers, cudaMemcpyDeviceToHost);
	float ret = 0;
	for (int i = 0; i < CV_numbers; i++)
	{
		ret += h_ene[i];
	}
	return ret;
}

void STEER_CV::Steer(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step,
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
		cv = cv_list[i];
		cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, need, step);
		if (!need_pressure)
			CUDA_LAUNCH_STREAM(steer_force_and_energy, cuda_grid_size, cuda_block_size, 0, cv->cuda_stream,
				atom_numbers, cv->d_value, cv->crd_grads, weight[i], frc, d_ene);
		else
			CUDA_LAUNCH_STREAM(steer_force_and_energy_and_virial, cuda_grid_size, cuda_block_size, 0, cv->cuda_stream,
				atom_numbers, cv->d_value, cv->crd_grads, cv->box_grads, box_length,
				weight[i], frc, d_ene, d_virial);
	}
}
