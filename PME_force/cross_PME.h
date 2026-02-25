#include "../common.h"
#include "../control.h"
#include "PME_force.h"

struct CROSS_PME
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20230523;

    int charge_perturbated = 0;
    void Initial(CONTROLLER *controller, const int atom_numbers, const int PME_Nall, const float lambda_lj, const char* module_name = NULL);
    float Get_Partial_H_Partial_Lambda(Particle_Mesh_Ewald* pme, const int atom_numbers,
        const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR uint_dr_to_dr_cof, const float* d_charge, const ATOM_GROUP* nl,
        int* d_excluded_list_start, int* d_excluded_list, int* d_excluded_numbers, float* d_direct_ene);

    float* charge_B_A;
    float* d_charge_B_A;
    float* PME_Q_B_A;
    float* d_cross_reciprocal_ene;
    float* d_cross_self_ene;
    float* charge_sum_B_A;
    float* d_cross_correction_atom_energy;
    float* d_cross_correction_ene;
    float dH_dlambda;
    float cross_reciprocal_ene;
    float cross_self_ene;
    float cross_direct_ene;
    float cross_correction_ene;
};