#include "hard_wall.h"

template<bool low_boundary, int xyz>
static __global__ void Hard_Wall_Reflection_Cuda(int atom_numbers, float* crd, float* vel, float boundary)
{
    int atom_i = threadIdx.x + blockDim.x * blockIdx.x;
    if (atom_i < atom_numbers)
    {
        int index = 3 * atom_i + xyz;
        float delta = crd[index] - boundary;
        if (low_boundary)
        {
            if (delta < 0)
            {
                vel[index] = fabsf(vel[index]);
            }
        }
        else
        {
            if (delta > 0)
            {
                vel[index] = -fabsf(vel[index]);
            }
        }
    }
}

void HARD_WALL::Initial(CONTROLLER* controller, float temperature, float pressure, bool npt_mode, const char* module_name)
{
    if (module_name != NULL)
    {
        strcpy(this->module_name, module_name);
    }
    else
    {
        strcpy(this->module_name, "hard_wall");
    }
    controller->printf("START INITIALIZING HARD WALL:\n");
    if (controller->Command_Exist(this->module_name, "x_low"))
    {
        controller->Check_Float(this->module_name, "x_low", "HARD_WALL::Initial");
        x_low = atof(controller->Command(this->module_name, "x_low"));
        controller->printf("    x_low = %f Angstrom\n", x_low);
        is_initialized = 1;
    }
    if (controller->Command_Exist(this->module_name, "y_low"))
    {
        controller->Check_Float(this->module_name, "y_low", "HARD_WALL::Initial");
        y_low = atof(controller->Command(this->module_name, "y_low"));
        controller->printf("    y_low = %f Angstrom\n", y_low);
        is_initialized = 1;
    }
    if (controller->Command_Exist(this->module_name, "z_low"))
    {
        controller->Check_Float(this->module_name, "z_low", "HARD_WALL::Initial");
        z_low = atof(controller->Command(this->module_name, "z_low"));
        controller->printf("    z_low = %f Angstrom\n", z_low);
        is_initialized = 1;
    }
    if (controller->Command_Exist(this->module_name, "x_high"))
    {
        controller->Check_Float(this->module_name, "x_high", "HARD_WALL::Initial");
        x_high = atof(controller->Command(this->module_name, "x_high"));
        controller->printf("    x_high = %f Angstrom\n", x_high);
        is_initialized = 1;
    }
    if (controller->Command_Exist(this->module_name, "y_high"))
    {
        controller->Check_Float(this->module_name, "y_high", "HARD_WALL::Initial");
        y_high = atof(controller->Command(this->module_name, "y_high"));
        controller->printf("    y_high = %f Angstrom\n", y_high);
        is_initialized = 1;
    }
    if (controller->Command_Exist(this->module_name, "z_high"))
    {
        controller->Check_Float(this->module_name, "z_high", "HARD_WALL::Initial");
        z_high = atof(controller->Command(this->module_name, "z_high"));
        controller->printf("    z_high = %f Angstrom\n", z_high);
        is_initialized = 1;
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        if (npt_mode)
        {
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "HARD_WALL::Initial", "Reason:\n\tHard walls can not be used in the NPT mode\n");
        }
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller->printf("END INITIALIZING HARD WALL\n\n");
}

void HARD_WALL::Reflect(int atom_numbers, VECTOR* crd, VECTOR* vel)
{
    if (!this->is_initialized)
        return;
    if (!isinf(this->x_high))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<false, 0>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->x_high);
    }
    if (!isinf(this->y_high))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<false, 1>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->y_high);
    }
    if (!isinf(this->z_high))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<false, 2>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->z_high);
    }
    if (!isinf(this->x_low))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<true, 0>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->x_low);
    }
    if (!isinf(this->y_low))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<true, 1>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->y_low);
    }
    if (!isinf(this->z_low))
    {
        CUDA_LAUNCH((Hard_Wall_Reflection_Cuda<true, 2>), dim3((atom_numbers + 1023) / 1024), dim3(1024),
            atom_numbers, (float*)crd, (float*)vel, this->z_low);
    }
}

