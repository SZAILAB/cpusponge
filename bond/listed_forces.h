/*
* Copyright 2021-2023 Gao's lab, Peking University, CCME. All rights reserved.
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


#ifndef LISTED_FORCE_CUH
#define LISTED_FORCE_CUH
#include "../common.h"
#include "../control.h"

struct LISTED_FORCE
{
    char module_name[CHAR_LENGTH_MAX];
    std::vector<std::string> atom_labels;
    std::vector<std::string> parameter_type;
    std::vector<std::string> parameter_name;
    std::string source_code;
    std::string connected_atoms;
    std::string constrain_distance;
    JIT_Function force_function;
    int item_numbers;
    void** gpu_parameters;
    void** cpu_parameters;
    std::vector<void*> launch_args;
    float* item_energy;
    float* sum_energy;
    void Initialize_Parameters(CONTROLLER* controller, std::string parameter_string);
    void Compile(CONTROLLER* controller);
    void Initial(CONTROLLER* controller, CONECT* connectivity, PAIR_DISTANCE* con_dis);
    void Compute_Force(VECTOR* crd, VECTOR box_length, VECTOR* frc, float* atom_energy, float* atom_virial);
    float Get_Energy(VECTOR* crd, VECTOR box_length);
};

struct LISTED_FORCES
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20230710;

    std::vector<LISTED_FORCE*> forces;
    void Initial(CONTROLLER* controller, CONECT* connectivity, PAIR_DISTANCE* con_dis, const char* module_name = NULL);
    void Compute_Force(VECTOR* crd, VECTOR box_length, VECTOR* frc, float* atom_energy, float* atom_virial);
    void Step_Print(CONTROLLER* controller, VECTOR* crd, VECTOR box_length);
};

#endif
