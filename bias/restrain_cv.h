#ifndef RESTRAIN_CV_CUH
#define RESTRAIN_CV_CUH

#include "../collective_variable/collective_variable.h"


// E = \sum weight * (CV - reference) ^ 2
struct RESTRAIN_CV
{
	char module_name[CHAR_LENGTH_MAX];
	int is_initialized = 0;
	int is_controller_printf_initialized = 0;
	int last_modify_date = 20220925;

	int CV_numbers;
	CV_LIST cv_list;
	float* weight;
	float* reference;
	float* period;
        int* start_step;
        int* max_step;
        int* reduce_step;
        int* stop_step;
	float* h_ene, * d_ene;
	int cuda_grid_size, cuda_block_size;
	void Initial(CONTROLLER* controller, COLLECTIVE_VARIABLE_CONTROLLER* manager);
	float Get_Energy(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step);
	void Restraint(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int step,
		float *d_ene, float *d_virial, VECTOR *frc, int need_potential, int need_pressure);
};

#endif //


