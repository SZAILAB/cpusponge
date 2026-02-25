/*
* Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
*
* NOTICE TO LICENSEE:
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//更加详细的类似备注请见bond模块

#ifndef NB14_CUH
#define NB14_CUH
#include "../common.h"
#include "../control.h"

struct NON_BOND_14
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211222;

    //r = ab原子的距离
    //E_lj_energy = (A/12 * r^-12 - B/6 * r^-6) 
    //E_cf_energy = cf_scale_factor * charge_a * charge_b / r
    //lj_A、lj_B、charge从外部传入，lj_A、lj_B参考LJ，charge参考md_core
    int nb14_numbers = 0;
    int *h_atom_a = NULL;
    int *h_atom_b = NULL;
    int *d_atom_a = NULL;
    int *d_atom_b = NULL;
    float *h_A = NULL;
    float *d_A = NULL;
    float *h_B = NULL;
    float *d_B = NULL;
    float *h_cf_scale_factor = NULL;
    float *d_cf_scale_factor = NULL;

    float *d_nb14_energy = NULL;
    float *d_nb14_cf_energy_sum = NULL;
    float *d_nb14_lj_energy_sum = NULL;
    float h_nb14_cf_energy_sum = 0;
    float h_nb14_lj_energy_sum = 0;

    int threads_per_block = 128;

    void Initial(CONTROLLER *controller, const float *LJ_type_A, const float *LJ_type_B, const int *lj_atom_type, const char *module_name = NULL);
    void Clear();
    void Memory_Allocate();
    void Read_Information_From_AMBERFILE(const char *file_name, CONTROLLER controller, const float *LJ_type_A, const float *LJ_type_B, const int *lj_atom_type);
    void Parameter_Host_To_Device();

    //同时计算原子的力、能量和维里
    void Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, const VECTOR scaler, VECTOR *frc, float *atom_energy, float *atom_virial);

    //获得能量
    float Get_14_LJ_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, int is_download = 1);
    float Get_14_CF_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, const VECTOR scaler, int is_download = 1);

    
};

#endif //NB14_CUH(nb14.h)
