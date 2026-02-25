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


#ifndef SOFT_WALL_CUH
#define SOFT_WALL_CUH
#include "../common.h"
#include "../control.h"
#include "nvrtc.h"

struct SOFT_WALL
{
    char module_name[CHAR_LENGTH_MAX];
    std::string source_code;
    JIT_Function force_function;
    float* item_energy;
    float* sum_energy;
    void Compile(CONTROLLER* controller);
    void Initial(int atom_numbers);
    void Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc, float* atom_energy);
    float Get_Energy(int atom_numbers, VECTOR* crd);
};

struct SOFT_WALLS
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20231011;

    std::vector<SOFT_WALL*> forces;
    void Initial(CONTROLLER* controller, int atom_numbers, const char* module_name = NULL);
    void Compute_Force(int atom_numbers, VECTOR* crd, VECTOR* frc, float* atom_energy);
    void Step_Print(CONTROLLER* controller, int atom_numbers, VECTOR* crd);
};

#endif
