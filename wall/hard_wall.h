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


#ifndef HARD_WALL_CUH
#define HARD_WALL_CUH
#include "../common.h"
#include "../control.h"


//硬墙，直接反射粒子
struct HARD_WALL
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20231011;
    
    // 坐标小于时反弹的墙
    float x_low = -INFINITY;
    float y_low = -INFINITY;
    float z_low = -INFINITY;
    // 坐标大于时反弹的墙
    float x_high = INFINITY;
    float y_high = INFINITY;
    float z_high = INFINITY;
    /* 墙的移动相关信息暂时未实现
    // 如果想让墙可以动，可以设定质量，保证动量守恒
    // 此处存储的为质量的倒数
    float x_low_mass_inverse;
    float y_low_mass_inverse;
    float z_low_mass_inverse;
    float x_high_mass_inverse;
    float y_high_mass_inverse;
    float z_high_mass_inverse;
    */
    //温度和压强用于墙的移动，但目前暂不实现
    void Initial(CONTROLLER* controller, float temperature, float pressure, bool npt_mode, const char* module_name = NULL);
    void Reflect(int atom_numbers, VECTOR* crd, VECTOR* vel);
};

#endif

