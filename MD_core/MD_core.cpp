#include "MD_core.h"
#include <climits>

#define BOX_TRAJ_COMMAND "box"
#define BOX_TRAJ_DEFAULT_FILENAME "mdbox.txt"
#define TRAJ_COMMAND "crd"
#define TRAJ_DEFAULT_FILENAME "mdcrd.dat"
#define RESTART_COMMAND "rst"
#define RESTART_DEFAULT_FILENAME "restart"
//20210827用于输出速度和力
#define FRC_TRAJ_COMMAND "frc"
#define VEL_TRAJ_COMMAND "vel"

//20230303 用于错误输出原因
#define ATOM_NUMBERS_DISMATCH "Reason:\n\t'atom_numbers' (the number of atoms) is diiferent in different input files\n"
#define ATOM_NUMBERS_MISSING "Reason:\n\tno 'atom_numbers' (the number of atoms) found. No 'mass_in_file' or 'amber_parm7' is provided\n"


//用来将原子的真实坐标转换为unsigned int坐标,注意factor需要乘以0.5（保证越界坐标自然映回box）
static __global__ void Crd_To_Uint_Crd(const int atom_numbers, const VECTOR box_length_inverse, const VECTOR *crd, UNSIGNED_INT_VECTOR *uint_crd)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        UNSIGNED_INT_VECTOR uinti;
        VECTOR temp = crd[atom_i];

        temp.x *= box_length_inverse.x;
        temp.y *= box_length_inverse.y;
        temp.z *= box_length_inverse.z;

        temp.x -= floorf(temp.x);
        temp.y -= floorf(temp.y);
        temp.z -= floorf(temp.z);

        uinti.uint_x = CONSTANT_UINT_MAX_FLOAT * temp.x;
        uinti.uint_y = CONSTANT_UINT_MAX_FLOAT * temp.y;
        uinti.uint_z = CONSTANT_UINT_MAX_FLOAT * temp.z;

        uint_crd[atom_i] = uinti;
    }
}

static __global__ void MD_Iteration_Leap_Frog
(const int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, const float *inverse_mass, const float dt)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        acc[i].x = inverse_mass[i] * frc[i].x;
        acc[i].y = inverse_mass[i] * frc[i].y;
        acc[i].z = inverse_mass[i] * frc[i].z;

        vel[i].x = vel[i].x + dt*acc[i].x;
        vel[i].y = vel[i].y + dt*acc[i].y;
        vel[i].z = vel[i].z + dt*acc[i].z;

        crd[i].x = crd[i].x + dt*vel[i].x;
        crd[i].y = crd[i].y + dt*vel[i].y;
        crd[i].z = crd[i].z + dt*vel[i].z;

    }
}

static __global__ void Scale_Position_To_Center_CUDA(const int atom_numbers, const float scaler, VECTOR* crd, VECTOR half_box_length)
{
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        crd[atom_i] = scaler * (crd[atom_i] - half_box_length) + half_box_length;   
    }
}

static __global__ void MD_Iteration_Leap_Frog_With_Max_Velocity
(const int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, const float *inverse_mass, const float dt, const float max_velocity)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        VECTOR acc_i = inverse_mass[i] * frc[i];
        VECTOR vel_i = vel[i] + dt * acc_i;
        vel_i = Make_Vector_Not_Exceed_Value(vel_i, max_velocity);
        vel[i] = vel_i;
        crd[i] = crd[i] + dt * vel_i;
    }
}

static __global__ void MD_Iteration_Gradient_Descent
(const int atom_numbers, VECTOR *crd, VECTOR *frc, const float *mass_inverse, const float dt, VECTOR *vel, const float momentum_keep)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        vel[i] = momentum_keep * vel[i] + dt * mass_inverse[i] * frc[i];
        crd[i] = crd[i] + dt * vel[i];
    }
}

static __global__ void MD_Iteration_Gradient_Descent_With_Max_Move
(const int atom_numbers, VECTOR *crd, VECTOR *frc, const float *mass_inverse, const float dt, VECTOR *vel, const float momentum_keep, float max_move)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        vel[i] = momentum_keep * vel[i] + dt * mass_inverse[i] * frc[i];
        VECTOR move = dt * vel[i];
        move = Make_Vector_Not_Exceed_Value(move, max_move);
        crd[i] = crd[i] + move;
    }
}

static __global__ void MD_Iteration_Speed_Verlet_1(const int atom_numbers, const float half_dt, const float dt, const VECTOR *acc, VECTOR *vel, VECTOR *crd, VECTOR *frc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        vel[i].x = vel[i].x + half_dt*acc[i].x;
        vel[i].y = vel[i].y + half_dt*acc[i].y;
        vel[i].z = vel[i].z + half_dt*acc[i].z;
        crd[i].x = crd[i].x + dt*vel[i].x;
        crd[i].y = crd[i].y + dt*vel[i].y;
        crd[i].z = crd[i].z + dt*vel[i].z;
    }
}

static __global__ void MD_Iteration_Speed_Verlet_2(const int atom_numbers, const float half_dt, const float *inverse_mass, const VECTOR *frc, VECTOR *vel, VECTOR *acc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        acc[i].x = inverse_mass[i] * frc[i].x;
        acc[i].y = inverse_mass[i] * frc[i].y;
        acc[i].z = inverse_mass[i] * frc[i].z;
        vel[i].x = vel[i].x + half_dt*acc[i].x;
        vel[i].y = vel[i].y + half_dt*acc[i].y;
        vel[i].z = vel[i].z + half_dt*acc[i].z;
    }
}

static __global__ void MD_Iteration_Speed_Verlet_2_With_Max_Velocity(const int atom_numbers, const float half_dt, const float *inverse_mass, const VECTOR *frc, VECTOR *vel, VECTOR *acc, const float max_velocity)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        VECTOR acc_i = inverse_mass[i] * frc[i];
        VECTOR vel_i = vel[i] + half_dt * acc_i;

        vel[i] = Make_Vector_Not_Exceed_Value(vel_i, max_velocity);
        acc[i] = acc_i;
    }
}

static __global__ void Get_Origin(const int residue_numbers, const int *start, const int *end,
    const VECTOR *crd, const float *atom_mass, const float *residue_mass_inverse, VECTOR *center_of_mass)
{
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
    {
        VECTOR origin = { FLT_MAX, FLT_MAX, FLT_MAX };
        for (int atom_i = start[residue_i]; atom_i < end[residue_i]; atom_i += 1)
        {
            VECTOR crd_i = crd[atom_i]; 
            origin.x = fminf(origin.x, crd_i.x);
            origin.y = fminf(origin.y, crd_i.y);
            origin.z = fminf(origin.z, crd_i.z);
        }
        center_of_mass[residue_i] = origin;
    }
}

static __global__ void Map_Center_Of_Mass(const int residue_numbers, const int *start, const int *end,
    const float scaler, const VECTOR *center_of_mass, const VECTOR box_length, VECTOR *crd, int* periodicity)
{
    VECTOR trans_vec;
    VECTOR com;
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
    {
        if (periodicity == NULL || periodicity[residue_i] == 0)
        {
            com = center_of_mass[residue_i];
            trans_vec.x = com.x - floorf(com.x / box_length.x) * box_length.x;
            trans_vec.y = com.y - floorf(com.y / box_length.y) * box_length.y;
            trans_vec.z = com.z - floorf(com.z / box_length.z) * box_length.z;
            trans_vec.x = scaler * trans_vec.x - com.x;
            trans_vec.y = scaler * trans_vec.y - com.y;
            trans_vec.z = scaler * trans_vec.z - com.z;
            for (int atom_i = start[residue_i] + threadIdx.y; atom_i < end[residue_i]; atom_i += blockDim.y)
            {
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
        else
        {
            for (int atom_i = start[residue_i] + threadIdx.y; atom_i < end[residue_i]; atom_i += blockDim.y)
            {
                com = crd[atom_i];
                trans_vec.x = com.x - floorf(com.x / box_length.x) * box_length.x;
                trans_vec.y = com.y - floorf(com.y / box_length.y) * box_length.y;
                trans_vec.z = com.z - floorf(com.z / box_length.z) * box_length.z;
                trans_vec.x = scaler * trans_vec.x - com.x;
                trans_vec.y = scaler * trans_vec.y - com.y;
                trans_vec.z = scaler * trans_vec.z - com.z;
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
    }
}

static __global__ void Map_Center_Of_Mass(const int residue_numbers, const int *start, const int *end,
    const VECTOR scaler, const VECTOR *center_of_mass, const VECTOR box_length, VECTOR *crd, int* periodicity)
{
    VECTOR trans_vec;
    VECTOR com;
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
    {
        if (periodicity == NULL || periodicity[residue_i] == 0)
        {
            com = center_of_mass[residue_i];
            trans_vec.x = com.x - floorf(com.x / box_length.x) * box_length.x;
            trans_vec.y = com.y - floorf(com.y / box_length.y) * box_length.y;
            trans_vec.z = com.z - floorf(com.z / box_length.z) * box_length.z;
            trans_vec.x = scaler.x * trans_vec.x - com.x;
            trans_vec.y = scaler.y * trans_vec.y - com.y;
            trans_vec.z = scaler.z * trans_vec.z - com.z;
            for (int atom_i = start[residue_i] + threadIdx.y; atom_i < end[residue_i]; atom_i += blockDim.y)
            {
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
        else
        {
            for (int atom_i = start[residue_i] + threadIdx.y; atom_i < end[residue_i]; atom_i += blockDim.y)
            {
                com = crd[atom_i];
                trans_vec.x = com.x - floorf(com.x / box_length.x) * box_length.x;
                trans_vec.y = com.y - floorf(com.y / box_length.y) * box_length.y;
                trans_vec.z = com.z - floorf(com.z / box_length.z) * box_length.z;
                trans_vec.x = scaler.x * trans_vec.x - com.x;
                trans_vec.y = scaler.y * trans_vec.y - com.y;
                trans_vec.z = scaler.z * trans_vec.z - com.z;
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
    }
}

static __global__ void Add_Sum_List(int n, float *atom_virial, float *sum_virial)
{
    float temp = 0;
    for (int i = threadIdx.x; i < n; i = i + blockDim.x)
    {
        temp = temp + atom_virial[i];
    }
    atomicAdd(sum_virial, temp);
}

static __global__ void Calculate_Pressure_Cuda(const float V_inverse, const float *ek, const float *virial, float *pressure)
{
    pressure[0] = (ek[0] * 2 + virial[0]) * 0.33333333333333f * V_inverse;
}

static __global__ void MD_Residue_Ek
(const int residue_numbers,const int *start,const int *end,float *ek,
const VECTOR *atom_vel,const float *atom_mass)
{
    int residue_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
    {
        VECTOR momentum = {0.,0.,0.};
        float res_mass = 0.; //待提出，只需要初始时计算一遍
        int s = start[residue_i];
        int e = end[residue_i];
        float mass_lin;
        for (int atom_i = s; atom_i < e; atom_i = atom_i + 1)
        {
            mass_lin = atom_mass[atom_i];

            momentum.x = momentum.x + mass_lin*atom_vel[atom_i].x;
            momentum.y = momentum.y + mass_lin*atom_vel[atom_i].y;
            momentum.z = momentum.z + mass_lin*atom_vel[atom_i].z;
            res_mass = res_mass + mass_lin;
        }
        ek[residue_i] = 0.5*(momentum.x*momentum.x + momentum.y*momentum.y + momentum.z*momentum.z) / res_mass;
    }
}

static __global__ void MD_Atom_Ek
(const int atom_numbers, float *ek, const VECTOR *atom_vel, const float *atom_mass)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR v = atom_vel[atom_i];
        ek[atom_i] = 0.5 * v * v * atom_mass[atom_i];
    }
}

void MD_INFORMATION::system_information::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    steps = 0;
    if (md_info->mode != md_info->RERUN)
    {
        step_limit = 1000;
        if (controller[0].Command_Exist("step_limit"))
        {
            controller->Check_Int("step_limit", "MD_INFORMATION::system_information::Initial");
            step_limit = atoi(controller[0].Command("step_limit"));
        }

        target_temperature = 300.0f;
        if (md_info->mode >= md_info->NVT && controller[0].Command_Exist("target_temperature"))
        {
            controller->Check_Float("target_temperature", "MD_INFORMATION::system_information::Initial");
            target_temperature = atof(controller[0].Command("target_temperature"));
        }

        target_pressure = 1;
        if (md_info->mode == md_info->NPT && controller[0].Command_Exist("target_pressure"))
        {
            controller->Check_Float("target_pressure", "MD_INFORMATION::system_information::Initial");
            target_pressure = atof(controller[0].Command("target_pressure"));
        }
        target_pressure *= CONSTANT_PRES_CONVERTION_INVERSE;
    }
    else
    {
        step_limit = INT_MAX - 1;
        if (controller[0].Command_Exist("frame_limit"))
        {
            controller->Check_Int("frame_limit", "MD_INFORMATION::system_information::Initial");
            step_limit = atoi(controller[0].Command("frame_limit"));
        }
    }
    for (int i = 0; i < md_info->atom_numbers; i++)
    {
        std::set<int> temp;
        connectivity[i] = temp;
    }
    if (md_info->mode != md_info->RERUN)
    {
        controller->Step_Print_Initial("step", "%d");
        controller->Step_Print_Initial("time", "%.3lf");
        controller->Step_Print_Initial("temperature", "%.2f");
    }
    else
    {
        controller->Step_Print_Initial("frame", "%d");
    }
    Cuda_Malloc_Safely((void**)&this->d_virial, sizeof(float));
    Cuda_Malloc_Safely((void**)&this->d_pressure, sizeof(float));
    Cuda_Malloc_Safely((void**)&this->d_temperature, sizeof(float));
    Cuda_Malloc_Safely((void**)&this->d_potential, sizeof(float));
    Cuda_Malloc_Safely((void**)&this->d_sum_of_atom_ek, sizeof(float));
}

void MD_INFORMATION::non_bond_information::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    if (controller[0].Command_Exist("skin"))
    {
        controller->Check_Float("skin", "MD_INFORMATION::non_bond_information::Initial");
        skin = atof(controller[0].Command("skin"));
    }
    else
    {
        skin = 2.0;
    }
    controller->printf("    skin set to %.2f Angstrom\n", skin);

    if (controller[0].Command_Exist("cutoff"))
    {
        controller->Check_Float("cutoff", "MD_INFORMATION::non_bond_information::Initial");
        cutoff = atof(controller[0].Command("cutoff"));
    }
    else
    {
        cutoff = 10.0;
    }
    controller->printf("    cutoff set to %.2f Angstrom\n", cutoff);
    /*===========================
    读取排除表相关信息
    ============================*/
    if (controller[0].Command_Exist("exclude_in_file"))
    {
        FILE *fp = NULL;
        controller->printf("    Start reading excluded list:\n");
        Open_File_Safely(&fp, controller[0].Command("exclude_in_file"), "r");
        
        int atom_numbers = 0;
        int scanf_ret = fscanf(fp, "%d %d", &atom_numbers, &excluded_atom_numbers);
        if (scanf_ret != 2)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of exclude_in_file is not right\n");
        }
        if (md_info->atom_numbers > 0 && md_info->atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::non_bond_information::Initial", ATOM_NUMBERS_DISMATCH);
        }
        else if (md_info->atom_numbers == 0)
        {
            md_info->atom_numbers = atom_numbers;
        }
        controller->printf("        excluded list total length is %d\n", excluded_atom_numbers);

        Cuda_Malloc_Safely((void**)&d_excluded_list_start, sizeof(int)*atom_numbers);
        Cuda_Malloc_Safely((void**)&d_excluded_numbers, sizeof(int)*atom_numbers);
        Cuda_Malloc_Safely((void**)&d_excluded_list, sizeof(int)*excluded_atom_numbers);

        Malloc_Safely((void**)&h_excluded_list_start, sizeof(int)*atom_numbers);
        Malloc_Safely((void**)&h_excluded_numbers, sizeof(int)*atom_numbers);
        Malloc_Safely((void**)&h_excluded_list, sizeof(int)*excluded_atom_numbers);
        int count = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%d", &h_excluded_numbers[i]);
            if (scanf_ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "Reason:\n\tThe format of exclude_in_file is not right\n");
            }
            h_excluded_list_start[i] = count;
            for (int j = 0; j < h_excluded_numbers[i]; j++)
            {
                scanf_ret = fscanf(fp, "%d", &h_excluded_list[count]);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "Reason:\n\tThe format of exclude_in_file is not right\n");
                }
                count++;
            }
        }
        if (count != excluded_atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", 
                "Reason:\n\tThe format of exclude_in_file is not right (excluded_atom_numbers is not right)\n");
        }
        cudaMemcpy(d_excluded_list_start, h_excluded_list_start, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_numbers, h_excluded_numbers, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_list, h_excluded_list, sizeof(int)*excluded_atom_numbers, cudaMemcpyHostToDevice);
        controller->printf("    End reading excluded list\n\n");
        fclose(fp);
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        /*===========================
        从parm中读取排除表相关信息
        ============================*/
        FILE *parm = NULL;
        Open_File_Safely(&parm, controller[0].Command("amber_parm7"), "r");
        controller->printf("    Start reading excluded list from AMBER file:\n");
        while (true)
        {
            char temps[CHAR_LENGTH_MAX];
            char temp_first_str[CHAR_LENGTH_MAX];
            char temp_second_str[CHAR_LENGTH_MAX];
            if (!fgets(temps, CHAR_LENGTH_MAX, parm))
            {
                break;
            }
            if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
            {
                continue;
            }
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "POINTERS") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

                int atom_numbers = 0;
                int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                }
                if (md_info->atom_numbers > 0 && md_info->atom_numbers != atom_numbers)
                {
                    controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::non_bond_information::Initial", ATOM_NUMBERS_DISMATCH);
                }
                else if (md_info->atom_numbers == 0)
                {
                    md_info->atom_numbers = atom_numbers;
                }
                Cuda_Malloc_Safely((void**)&d_excluded_list_start, sizeof(int)*atom_numbers);
                Cuda_Malloc_Safely((void**)&d_excluded_numbers, sizeof(int)*atom_numbers);

                Malloc_Safely((void**)&h_excluded_list_start, sizeof(int)*atom_numbers);
                Malloc_Safely((void**)&h_excluded_numbers, sizeof(int)*atom_numbers);
                for (int i = 0; i < 9; i = i + 1)
                {
                    scanf_ret = fscanf(parm, "%d\n", &excluded_atom_numbers);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                    }
                }
                scanf_ret = fscanf(parm, "%d\n", &excluded_atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                }
                controller->printf("        excluded list total length is %d\n", excluded_atom_numbers);

                Cuda_Malloc_Safely((void**)&d_excluded_list, sizeof(int)*excluded_atom_numbers);
                Malloc_Safely((void**)&h_excluded_list, sizeof(int)*excluded_atom_numbers);
            }

            //read atom_excluded_number for every atom
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "NUMBER_EXCLUDED_ATOMS") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                for (int i = 0; i<md_info->atom_numbers; i = i + 1)
                {
                    int scanf_ret = fscanf(parm, "%d\n", &h_excluded_numbers[i]);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                    }
                }
            }
            //read every atom's excluded atom list
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "EXCLUDED_ATOMS_LIST") == 0)
            {
                int count = 0;
                int lin = 0;
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                for (int i = 0; i<md_info->atom_numbers; i = i + 1)
                {
                    h_excluded_list_start[i] = count;
                    for (int j = 0; j<h_excluded_numbers[i]; j = j + 1)
                    {
                        int scanf_ret = fscanf(parm, "%d\n", &lin);
                        if (scanf_ret != 1)
                        {
                            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                        }
                        if (lin == 0)
                        {
                            h_excluded_numbers[i] = 0;
                            break;
                        }
                        else
                        {
                            h_excluded_list[count] = lin - 1;
                            count = count + 1;
                        }
                    }
                    if (h_excluded_numbers[i] > 0)
                        std::sort(&h_excluded_list[h_excluded_list_start[i]],
                                  &h_excluded_list[h_excluded_list_start[i]] + h_excluded_numbers[i]);
                }
            }
        }

        cudaMemcpy(d_excluded_list_start, h_excluded_list_start, sizeof(int)*md_info->atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_numbers, h_excluded_numbers, sizeof(int)*md_info->atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_list, h_excluded_list, sizeof(int)*excluded_atom_numbers, cudaMemcpyHostToDevice);
        controller->printf("    End reading excluded list from AMBER file\n\n");
        fclose(parm);
    }
    else
    {
        int atom_numbers = md_info->atom_numbers;
        excluded_atom_numbers = 0;
        controller->printf("    Set all atom exclude no atoms as default\n"); 

        Cuda_Malloc_Safely((void**)&d_excluded_list_start, sizeof(int)*atom_numbers);
        Cuda_Malloc_Safely((void**)&d_excluded_numbers, sizeof(int)*atom_numbers);
        Cuda_Malloc_Safely((void**)&d_excluded_list, sizeof(int)*excluded_atom_numbers);

        Malloc_Safely((void**)&h_excluded_list_start, sizeof(int)*atom_numbers);
        Malloc_Safely((void**)&h_excluded_numbers, sizeof(int)*atom_numbers);
        Malloc_Safely((void**)&h_excluded_list, sizeof(int)*excluded_atom_numbers);


        int count = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            h_excluded_numbers[i] = 0;
            h_excluded_list_start[i] = count;
            for (int j = 0; j < h_excluded_numbers[i]; j++)
            {
                h_excluded_list[count] = 0;
                count++;
            }
        }
        cudaMemcpy(d_excluded_list_start, h_excluded_list_start, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_numbers, h_excluded_numbers, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excluded_list, h_excluded_list, sizeof(int)*excluded_atom_numbers, cudaMemcpyHostToDevice);
    }

}

void MD_INFORMATION::periodic_box_condition_information::Initial(CONTROLLER *controller, MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * md_info->sys.box_length;
    this->md_info->MD_Information_Crd_To_Uint_Crd();
    this->pbc = true;
    if (controller->Command_Exist("pbc"))
    {
        this->pbc = controller->Get_Bool("pbc", "MD_INFORMATION::periodic_box_condition_information::Initial");
    }
    this->No_PBC_Check(controller);
}

void MD_INFORMATION::periodic_box_condition_information::No_PBC_Check(CONTROLLER* controller)
{
    if (this->pbc)
        return;
    if (md_info->nb.cutoff < 100)
    {
        controller->Warn("The cutoff for NOPBC is not greater than 100 angstrom, which may be inaccurate");
    }
    if (md_info->sys.box_length.x < 900 || md_info->sys.box_length.y < 900 || md_info->sys.box_length.z < 900)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "The box length of the system should always be greater than 900 angstrom for NOPBC");
    }
    if (md_info->mode == md_info->NPT)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "NPT mode can not be used for NOPBC");
    }
    if (controller->Command_Exist("SITS", "mode"))
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "SITS can not be used for NOPBC now");
    }
    Cuda_Malloc_Safely((void**)&nopbc_crd, sizeof(VECTOR) * (md_info->atom_numbers + md_info->no_direct_interaction_virtual_atom_numbers));
    cudaMemset(nopbc_crd, 0, sizeof(VECTOR) * md_info->atom_numbers);
}

void MD_INFORMATION::Read_Mode(CONTROLLER *controller)
{
    if (controller->Command_Choice("mode", "nvt"))
    {
        controller->printf("    Mode set to NVT\n");
        mode = 1;
    }
    else if (controller->Command_Choice("mode", "npt"))
    {
        controller->printf("    Mode set to NPT\n");
        mode = 2;
    }
    else if (controller->Command_Choice("mode", "minimization"))
    {
        controller->printf("    Mode set to Energy Minimization\n");
        mode = -1;
    }
    else if (controller->Command_Choice("mode", "nve"))
    {
        controller->printf("    Mode set to NVE\n");
        mode = 0;
    }
    else if (controller->Command_Choice("mode", "rerun"))
    {
        controller->printf("    Mode set to RERUN\n");
        mode = -2;
    }
    else
    {
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "MD_INFORMATION::Read_Mode", "Reason:\n\t'mode' is required\n");
    }
}

void MD_INFORMATION::Read_dt(CONTROLLER *controller)
{
    if (mode == RERUN)
    {
        return;
    }
    if (controller[0].Command_Exist("dt"))
    {
        controller->Check_Float("dt", "MD_INFORMATION::Read_dt");
        controller->printf("    dt set to %f ps\n", atof(controller[0].Command("dt")));
        dt = atof(controller[0].Command("dt")) * CONSTANT_TIME_CONVERTION;
        sscanf(controller[0].Command("dt"), "%lf", &sys.dt_in_ps);
    }
    else
    {
        if (mode != MINIMIZATION)
            dt = 0.001;
        else
            dt = 1e-8;
        sys.dt_in_ps = 0.001;
        controller->printf("    dt set to %e ps\n", dt);
        dt *= CONSTANT_TIME_CONVERTION;
    }
    if (mode == MINIMIZATION)
    {
        sys.dt_in_ps = 0;
    }
}

void MD_INFORMATION::trajectory_output::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    current_crd_synchronized_step = -1;
    if (md_info->mode != md_info->RERUN)
    {
        int default_interval = 1000;
        if (controller[0].Command_Exist("write_information_interval"))
        {
            controller->Check_Int("write_information_interval", "MD_INFORMATION::trajectory_output::Initial");
            default_interval = atoi(controller[0].Command("write_information_interval"));
        }
        write_trajectory_interval = default_interval;        
        if (controller[0].Command_Exist("write_trajectory_interval"))
        {
            controller->Check_Int("write_trajectory_interval", "MD_INFORMATION::trajectory_output::Initial");
            write_trajectory_interval = atoi(controller[0].Command("write_trajectory_interval"));
        }
        write_mdout_interval = default_interval;
        if (controller[0].Command_Exist("write_mdout_interval"))
        {
            controller->Check_Int("write_mdout_interval", "MD_INFORMATION::trajectory_output::Initial");
            write_mdout_interval = atoi(controller[0].Command("write_mdout_interval"));
        }
        write_restart_file_interval = md_info->sys.step_limit;
        if (controller[0].Command_Exist("write_restart_file_interval"))
        {
            controller->Check_Int("write_restart_file_interval", "MD_INFORMATION::trajectory_output::Initial");
            write_restart_file_interval = atoi(controller[0].Command("write_restart_file_interval"));
        }
        if (controller->Command_Exist(RESTART_COMMAND))
        {
            strcpy(restart_name, controller->Command(RESTART_COMMAND));
        }
        else
        {
            strcpy(restart_name, RESTART_DEFAULT_FILENAME);
        }
        if (controller->Command_Exist(FRC_TRAJ_COMMAND))
        {
            is_frc_traj = 1;
            Open_File_Safely(&frc_traj, controller->Command(FRC_TRAJ_COMMAND), "wb");
            controller->Set_File_Buffer(frc_traj, sizeof(VECTOR) * md_info->atom_numbers);
        }
        if (controller->Command_Exist(VEL_TRAJ_COMMAND))
        {
            is_vel_traj = 1;
            Open_File_Safely(&vel_traj, controller->Command(VEL_TRAJ_COMMAND), "wb");
            controller->Set_File_Buffer(vel_traj, sizeof(VECTOR) * md_info->atom_numbers);
        }
    }
    else
    {
        write_trajectory_interval = 0;
        write_mdout_interval = 1;
        write_restart_file_interval = 0;
    }

    if (write_trajectory_interval != 0)
    {
        if (controller->Command_Exist(TRAJ_COMMAND))
        {
            Open_File_Safely(&crd_traj, controller->Command(TRAJ_COMMAND), "wb");
        }
        else
        {
            Open_File_Safely(&crd_traj, TRAJ_DEFAULT_FILENAME, "wb");
        }
        controller->Set_File_Buffer(crd_traj, sizeof(VECTOR) * md_info->atom_numbers);
        if (controller->Command_Exist(BOX_TRAJ_COMMAND))
        {
            Open_File_Safely(&box_traj, controller->Command(BOX_TRAJ_COMMAND), "w");
        }
        else
        {
            Open_File_Safely(&box_traj, BOX_TRAJ_DEFAULT_FILENAME, "w");
        }
        controller->Set_File_Buffer(box_traj, sizeof(char) * 50);
    }
}

void MD_INFORMATION::NVE_iteration::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    max_velocity = -1;
    if (controller[0].Command_Exist("velocity_max"))
    {
        controller->Check_Float("velocity_max", "MD_INFORMATION::NVE_iteration::Initial");
        max_velocity = atof(controller[0].Command("velocity_max"));
    }
}
void MD_INFORMATION::residue_information::Read_AMBER_Parm7(const char *file_name, CONTROLLER controller)
{
    FILE *parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    controller.printf("    Start reading residue informataion from AMBER parm7:\n");
    
    while (true)
    {
        char temps[CHAR_LENGTH_MAX];
        char temp_first_str[CHAR_LENGTH_MAX];
        char temp_second_str[CHAR_LENGTH_MAX];
        if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL)
        {
            break;
        }
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        //read in atomnumber atomljtypenumber
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "POINTERS") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            int atom_numbers = 0;
            int scanf_ret = fscanf(parm, "%d", &atom_numbers);
            if (scanf_ret != 1)
            {
                controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Read_AMBER_Parm7", "Reason:\n\tthe format of the amber_parm7 is not right\n");
            }
            if (md_info->atom_numbers > 0 && md_info->atom_numbers != atom_numbers)
            {
                controller.Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::residue_information::Read_AMBER_Parm7", ATOM_NUMBERS_DISMATCH);
            }
            else if (md_info->atom_numbers == 0)
            {
                md_info->atom_numbers = atom_numbers;
            }
            for (int i = 0; i < 10; i = i + 1)
            {
                int lin;
                scanf_ret = fscanf(parm, "%d\n", &lin);
                if (scanf_ret != 1)
                {
                    controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Read_AMBER_Parm7", "Reason:\n\tthe format of the amber_parm7 is not right\n");
                }
            }
            scanf_ret = fscanf(parm, "%d\n", &this->residue_numbers);//NRES
            if (scanf_ret != 1)
            {
                controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Read_AMBER_Parm7", "Reason:\n\tthe format of the amber_parm7 is not right\n");
            }
            controller.printf("        residue_numbers is %d\n", this->residue_numbers);

            Malloc_Safely((void**)&h_mass, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_mass_inverse, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_res_start, sizeof(int)*this->residue_numbers);
            Malloc_Safely((void**)&h_res_end, sizeof(int)*this->residue_numbers);
            Malloc_Safely((void**)&h_momentum, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
            Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

            Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_res_start, sizeof(int)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_res_end, sizeof(int)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_momentum, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&res_ek_energy, sizeof(float)*this->residue_numbers); 
            Cuda_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));
        }//FLAG POINTERS

        //residue range read
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "RESIDUE_POINTER") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            //注意读进来的数的编号要减1
            int *lin_serial;
            Malloc_Safely((void**)&lin_serial, sizeof(int)* this->residue_numbers);
            for (int i = 0; i<this->residue_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%d\n", &lin_serial[i]);
                if (scanf_ret != 1)
                {
                    controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Read_AMBER_Parm7", "Reason:\n\tthe format of the amber_parm7 is not right\n");
                }
            }
            for (int i = 0; i<this->residue_numbers - 1; i = i + 1)
            {
                h_res_start[i] = lin_serial[i] - 1;
                h_res_end[i] = lin_serial[i + 1] - 1;
            }
            h_res_start[this->residue_numbers - 1] = lin_serial[this->residue_numbers - 1] - 1;
            h_res_end[this->residue_numbers - 1] = md_info->atom_numbers + 1 - 1;

            free(lin_serial);
        }
    }//while cycle

    cudaMemcpy(this->d_res_start, h_res_start, sizeof(int)*this->residue_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_res_end, h_res_end, sizeof(int)*this->residue_numbers, cudaMemcpyHostToDevice);

    controller.printf("    End reading residue informataion from AMBER parm7\n\n");

    fclose(parm);
}

void MD_INFORMATION::residue_information::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    if (!(controller[0].Command_Exist("residue_in_file")))
    {
        if (controller[0].Command_Exist("amber_parm7"))
        {
            Read_AMBER_Parm7(controller[0].Command("amber_parm7"), controller[0]);
            is_initialized = 1;
        }
        //对于没有residue输入的模拟，默认每个粒子作为一个residue
        else
        {
            residue_numbers = md_info->atom_numbers;
            controller->printf("    Set default residue list:\n");
            controller->printf("        residue_numbers is %d\n", residue_numbers);
            Malloc_Safely((void**)&h_mass, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_mass_inverse, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_res_start, sizeof(int)*this->residue_numbers);
            Malloc_Safely((void**)&h_res_end, sizeof(int)*this->residue_numbers);
            Malloc_Safely((void**)&h_momentum, sizeof(float)*this->residue_numbers);
            Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
            Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

            Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_res_start, sizeof(int)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_res_end, sizeof(int)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_momentum, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&d_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&res_ek_energy, sizeof(float)*this->residue_numbers);
            Cuda_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));
            int count = 0;
            int temp=1;//每个粒子作为一个residue
            for (int i = 0; i < residue_numbers; i++)
            {
                h_res_start[i] = count;
                count += temp;
                h_res_end[i] = count;
            }
            cudaMemcpy(d_res_start, h_res_start, sizeof(int)* residue_numbers, cudaMemcpyHostToDevice);
            cudaMemcpy(d_res_end, h_res_end, sizeof(int)* residue_numbers, cudaMemcpyHostToDevice);
            controller->printf("    End reading residue list\n\n");
            is_initialized = 1;
        }
    }
    else
    {
        FILE *fp = NULL;
        controller->printf("    Start reading residue list:\n");
        Open_File_Safely(&fp, controller[0].Command("residue_in_file"), "r");
        int atom_numbers = 0;
        int scanf_ret = fscanf(fp, "%d %d", &atom_numbers, &residue_numbers);
        if (scanf_ret != 2)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Initial", "Reason:\n\tthe format of the residue_in_file is not right\n");
        }
        if (md_info->atom_numbers > 0 && md_info->atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::residue_information::Initial", ATOM_NUMBERS_DISMATCH);
        }
        else if (md_info->atom_numbers == 0)
        {
            md_info->atom_numbers = atom_numbers;
        }
        controller->printf("        residue_numbers is %d\n", residue_numbers);
        Malloc_Safely((void**)&h_mass, sizeof(float)*this->residue_numbers);
        Malloc_Safely((void**)&h_mass_inverse, sizeof(float)*this->residue_numbers);
        Malloc_Safely((void**)&h_res_start, sizeof(int)*this->residue_numbers);
        Malloc_Safely((void**)&h_res_end, sizeof(int)*this->residue_numbers);
        Malloc_Safely((void**)&h_momentum, sizeof(float)*this->residue_numbers);
        Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
        Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

        Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&d_res_start, sizeof(int)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&d_res_end, sizeof(int)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&d_momentum, sizeof(float)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&d_center_of_mass, sizeof(VECTOR)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&res_ek_energy, sizeof(float)*this->residue_numbers);
        Cuda_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));

        int count = 0;
        int temp;
        for (int i = 0; i < residue_numbers; i++)
        {
            h_res_start[i] = count;
            scanf_ret = fscanf(fp, "%d", &temp);
            if (scanf_ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::residue_information::Initial", "Reason:\n\tthe format of the residue_in_file is not right\n");
            }
            count += temp;
            h_res_end[i] = count;
        }
        cudaMemcpy(d_res_start, h_res_start, sizeof(int)* residue_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_res_end, h_res_end, sizeof(int)* residue_numbers, cudaMemcpyHostToDevice);
        controller->printf("    End reading residue list\n\n");
        fclose(fp);
        is_initialized = 1;
    }
    if (is_initialized)
    {
        if (md_info->h_mass != NULL)
        {
            for (int i = 0; i < residue_numbers; i++)
            {
                float temp_mass = 0;
                for (int j = h_res_start[i]; j < h_res_end[i]; j++)
                {
                    temp_mass += md_info->h_mass[j];
                }
                this->h_mass[i] = temp_mass;
                if (temp_mass == 0)
                    this->h_mass_inverse[i] = 0;
                else
                    this->h_mass_inverse[i] = 1.0 / temp_mass;
            }
            cudaMemcpy(d_mass_inverse, h_mass_inverse, sizeof(float)* residue_numbers, cudaMemcpyHostToDevice);
            cudaMemcpy(d_mass, h_mass, sizeof(float)* residue_numbers, cudaMemcpyHostToDevice);
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "MD_INFORMATION::residue_information::Initial",
                                           "Reason:\n\tno mass information found");
        }
    }
}

void MD_INFORMATION::Read_Coordinate_And_Velocity(CONTROLLER *controller)
{
    sys.start_time = 0.0;
    if (mode == RERUN)
    {
        if (atom_numbers == 0)
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "MD_INFORMATION::Read_Coordinate_And_Velocity", 
                "Reason:\n\tFor the 'rerun' mode, the number of atoms should be provided by mass_in_file or charge_in_file\n");
        }
        this->atom_numbers = atom_numbers;
        Malloc_Safely((void**)&coordinate, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
        Cuda_Malloc_Safely((void**)&crd, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));   
        Cuda_Malloc_Safely((void**)&last_crd, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
        cudaMemset(last_crd, 0,  sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));         
        Malloc_Safely((void**)&velocity, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
        Cuda_Malloc_Safely((void**)&vel, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
        sys.box_length = { 100.0f, 100.0f, 100.0f };
        rerun.Initial(controller, this);
        rerun.Iteration(rerun.start_frame);
        sys.box_length.x *= rerun.box_length_change_factor.x;
        sys.box_length.y *= rerun.box_length_change_factor.y;
        sys.box_length.z *= rerun.box_length_change_factor.z;
    }
    else if (controller[0].Command_Exist("coordinate_in_file"))
    {
        Read_Coordinate_In_File(controller[0].Command("coordinate_in_file"), controller[0]);
        if (controller[0].Command_Exist("velocity_in_file"))
        {
            FILE *fp = NULL;
            controller->printf("    Start reading velocity_in_file:\n");
            Open_File_Safely(&fp, controller[0].Command("velocity_in_file"), "r");
            
            int atom_numbers = 0;
            char lin[CHAR_LENGTH_MAX];
            char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
            int scanf_ret = sscanf(lin, "%d", &atom_numbers);
            if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Coordinate_And_Velocity", ATOM_NUMBERS_DISMATCH);
            }
            Malloc_Safely((void**)&velocity, sizeof(VECTOR)* (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
            Cuda_Malloc_Safely((void**)&vel, sizeof(VECTOR)* (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
            for (int i = 0; i < atom_numbers; i++)
            {
                scanf_ret = fscanf(fp, "%f %f %f", &velocity[i].x, &velocity[i].y, &velocity[i].z);
                if (scanf_ret != 3)
                {
                    std::string error_reason = "Reason:\n\tthe format of the velocity_in_file (";
                    error_reason += controller->Command("velocity_in_file");
                    error_reason += ") is not right (missing the velocity of atom ";
                    error_reason += i;
                    error_reason += ")\n";
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Coordinate_And_Velocity", error_reason.c_str());
                }
            }
            cudaMemcpy(vel, velocity, sizeof(VECTOR)* atom_numbers, cudaMemcpyHostToDevice);
            controller->printf("    End reading velocity_in_file\n\n");
            fclose(fp);
        }
        else
        {
            controller->printf("    Velocity is set to zero as default\n");
            Malloc_Safely((void**)&velocity, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
            Cuda_Malloc_Safely((void**)&vel, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
            for (int i = 0; i < atom_numbers; i++)
            {
                velocity[i].x = 0;
                velocity[i].y = 0;
                velocity[i].z = 0;
            }
            cudaMemcpy(vel, velocity, sizeof(VECTOR)* atom_numbers, cudaMemcpyHostToDevice);
        }
    }
    else if (controller[0].Command_Exist("amber_rst7"))
    {
        output.amber_irest = 1;
        if (controller[0].Command_Exist("amber_irest"))
        {
            output.amber_irest = controller->Get_Bool("amber_irest", "MD_INFORMATION::Read_Coordinate_And_Velocity");
        }
        Read_Rst7(controller[0].Command("amber_rst7"), output.amber_irest, controller[0]);
    }
    else
    {
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
            "MD_INFORMATION::Read_Coordinate_And_Velocity",
            "Reason:\n\tno coordinate information found");
    }
}

void MD_INFORMATION::Read_Mass(CONTROLLER *controller)
{
    if (controller[0].Command_Exist("mass_in_file"))
    {
        FILE *fp = NULL;
        controller->printf("    Start reading mass:\n");
        Open_File_Safely(&fp, controller[0].Command("mass_in_file"), "r");
        int atom_numbers = 0;
        char lin[CHAR_LENGTH_MAX];
        char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
        int scanf_ret = sscanf(lin, "%d", &atom_numbers);
        if (scanf_ret != 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of mass_in_file is not right\n");
        }
        if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Mass", ATOM_NUMBERS_DISMATCH);
        }
        else if (this->atom_numbers == 0)
        {
            this->atom_numbers = atom_numbers;
        }
        Malloc_Safely((void**)&h_mass, sizeof(float)* atom_numbers);
        Malloc_Safely((void**)&h_mass_inverse, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)* atom_numbers);
        sys.total_mass = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", &h_mass[i]);
            if (scanf_ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of mass_in_file is not right\n");
            }
            sys.total_mass += h_mass[i];
            if (h_mass[i] == 0)
                h_mass_inverse[i] = 0;
            else
                h_mass_inverse[i] = 1.0 / h_mass[i];

        }
        controller->printf("    End reading mass\n\n");
        fclose(fp);
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        FILE *parm = NULL;
        Open_File_Safely(&parm, controller[0].Command("amber_parm7"), "r");
        controller[0].printf("    Start reading mass from AMBER parm7:\n");
        while (true)
        {
            char temps[CHAR_LENGTH_MAX];
            char temp_first_str[CHAR_LENGTH_MAX];
            char temp_second_str[CHAR_LENGTH_MAX];
            if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL)
            {
                break;
            }
            if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
            {
                continue;
            }
            //read in atomnumber atomljtypenumber
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "POINTERS") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

                int atom_numbers = 0;
                int scanf_ret = fscanf(parm, "%d", &atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                }
                if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
                {
                    controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Mass", ATOM_NUMBERS_DISMATCH);
                }
                else if (this->atom_numbers == 0)
                {
                    this->atom_numbers = atom_numbers;
                }
                Malloc_Safely((void**)&h_mass, sizeof(float)* atom_numbers);
                Malloc_Safely((void**)&h_mass_inverse, sizeof(float)* atom_numbers);
                Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)* atom_numbers);
                Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)* atom_numbers);
            }
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "MASS") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                double lin;
                sys.total_mass = 0;
                for (int i = 0; i < this->atom_numbers; i = i + 1)
                {
                    int scanf_ret = fscanf(parm, "%lf\n", &lin);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                    }
                    this->h_mass[i] = (float)lin;
                    if (h_mass[i] == 0)
                        h_mass_inverse[i] = 0;
                    else
                        h_mass_inverse[i] = 1.0f / h_mass[i];
                    sys.total_mass += h_mass[i];
                }
            }
        }
        controller[0].printf("    End reading mass from AMBER parm7\n\n");
        fclose(parm);
    }
    else if (atom_numbers > 0)
    {
        controller[0].printf("    mass is set to 20 as default\n");
        sys.total_mass = 0;
        Malloc_Safely((void**)&h_mass, sizeof(float)* atom_numbers);
        Malloc_Safely((void**)&h_mass_inverse, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)* atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            h_mass[i] = 20;
            h_mass_inverse[i] = 1.0 / h_mass[i];
            sys.total_mass += h_mass[i];
        }
    }
    else
    {
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "MD_INFORMATION::Read_Mass", ATOM_NUMBERS_MISSING);
    }
    if (atom_numbers > 0)
    {
        cudaMemcpy(d_mass, h_mass, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass_inverse, h_mass_inverse, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    }
}

void MD_INFORMATION::Read_Charge(CONTROLLER *controller)
{
    if (controller[0].Command_Exist("charge_in_file"))
    {
        FILE *fp = NULL;
        controller->printf("    Start reading charge:\n");
        Open_File_Safely(&fp, controller[0].Command("charge_in_file"), "r");
        int atom_numbers = 0;
        char lin[CHAR_LENGTH_MAX];
        char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
        int scanf_ret = sscanf(lin, "%d", &atom_numbers);
        if (scanf_ret != 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of charge_in_file is not right\n");
        }
        if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "MD_INFORMATION::Read_Charge", ATOM_NUMBERS_DISMATCH);
        }
        else if (this->atom_numbers == 0)
        {
            this->atom_numbers = atom_numbers;
        }
        Malloc_Safely((void**)&h_charge, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_charge, sizeof(float)* atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", &h_charge[i]);
            if (scanf_ret != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of charge_in_file is not right\n");
            }
        }
        controller->printf("    End reading charge\n\n");
        fclose(fp);
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        FILE *parm = NULL;
        Open_File_Safely(&parm, controller[0].Command("amber_parm7"), "r");
        controller[0].printf("    Start reading charge from AMBER parm7:\n");
        while (true)
        {
            char temps[CHAR_LENGTH_MAX];
            char temp_first_str[CHAR_LENGTH_MAX];
            char temp_second_str[CHAR_LENGTH_MAX];
            if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL)
            {
                break;
            }
            if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
            {
                continue;
            }
            //read in atomnumber atomljtypenumber
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "POINTERS") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

                int atom_numbers = 0;
                int scanf_ret = fscanf(parm, "%d", &atom_numbers);
                if (scanf_ret != 1)
                {
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                }
                if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
                {
                    controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Charge", ATOM_NUMBERS_DISMATCH);
                }
                else if (this->atom_numbers == 0)
                {
                    this->atom_numbers = atom_numbers;
                }
                Malloc_Safely((void**)&h_charge, sizeof(float)* atom_numbers);
                Cuda_Malloc_Safely((void**)&d_charge, sizeof(float)* atom_numbers);
            }
            if (strcmp(temp_first_str, "%FLAG") == 0
                && strcmp(temp_second_str, "CHARGE") == 0)
            {
                char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
                for (int i = 0; i < this->atom_numbers; i = i + 1)
                {
                    int scanf_ret = fscanf(parm, "%f", &h_charge[i]);
                    if (scanf_ret != 1)
                    {
                        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::non_bond_information::Initial", "The format of amber_parm7 is not right\n");
                    }
                }
            }
        }
        controller[0].printf("    End reading charge from AMBER parm7\n\n");
        fclose(parm);
    }
    else if (atom_numbers > 0)
    {
        controller[0].printf("    charge is set to 0 as default\n");
        Malloc_Safely((void**)&h_charge, sizeof(float)* atom_numbers);
        Cuda_Malloc_Safely((void**)&d_charge, sizeof(float)* atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            h_charge[i] = 0;
        }
    }
    else
    {
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "MD_INFORMATION::Read_Charge", ATOM_NUMBERS_MISSING);
    }
    if (atom_numbers > 0)
    {
        cudaMemcpy(d_charge, h_charge, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    }
}

//MD_INFORMATION成员函数
void MD_INFORMATION::Initial(CONTROLLER *controller)
{
    controller->printf("START INITIALIZING MD CORE:\n");
    atom_numbers = 0; //初始化，使得能够进行所有原子数目是否相等的判断
    
    strcpy(md_name, controller[0].Command("md_name"));
    Read_Mode(controller);
    Read_dt(controller);

    Read_Mass(controller);
    Read_Charge(controller);
    Atom_Information_Initial();

    Read_Coordinate_And_Velocity(controller);

    sys.Initial(controller, this);
    nb.Initial(controller, this);
    
    output.Initial(controller, this);

    nve.Initial(controller, this);
    
    min.Initial(controller, this);

    res.Initial(controller, this);

    mol.md_info = this;

    pbc.Initial(controller, this);
    
    is_initialized = 1;
    controller->printf("    structure last modify date is %d\n", last_modify_date);
    controller->printf("END INITIALIZING MD CORE\n\n");
}

void MD_INFORMATION::Atom_Information_Initial()
{
    Malloc_Safely((void**)&this->force, sizeof(VECTOR) * (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Malloc_Safely((void**)&this->h_atom_energy, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&this->h_atom_virial, sizeof(double)* atom_numbers);
    Cuda_Malloc_Safely((void**)&this->acc, sizeof(VECTOR) * atom_numbers);
    Cuda_Malloc_Safely((void**)&this->frc, sizeof(VECTOR) * (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&this->uint_crd, sizeof(UNSIGNED_INT_VECTOR) * (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&this->d_atom_energy, sizeof(float) * atom_numbers);
    Cuda_Malloc_Safely((void**)&this->d_atom_virial, sizeof(float) * atom_numbers);
    Cuda_Malloc_Safely((void**)&this->d_atom_ek, sizeof(float) * atom_numbers);
    CUDA_LAUNCH(Reset_List, dim3(static_cast<unsigned int>(ceilf((float)3. * this->atom_numbers / 32))),
        dim3(32), 3 * this->atom_numbers, (float*)this->acc, 0.);
    CUDA_LAUNCH(Reset_List, dim3(static_cast<unsigned int>(ceilf((float)3. * this->atom_numbers / 32))),
        dim3(32), 3 * this->atom_numbers, (float*)this->frc, 0.);
    sys.freedom = 3 * atom_numbers; //最大自由度，后面减
}

void MD_INFORMATION::Read_Coordinate_In_File(const char* file_name, CONTROLLER controller)
{
    FILE *fp =NULL;
    controller.printf("    Start reading coordinate_in_file:\n");
    Open_File_Safely(&fp, file_name, "r");
    char lin[CHAR_LENGTH_MAX];
    char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
    int atom_numbers = 0;
    int scanf_ret = sscanf(lin, "%d %lf", &atom_numbers, &sys.start_time);
    if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
    {
        controller.Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Coordinate_In_File", ATOM_NUMBERS_DISMATCH);
    }
    else if (this->atom_numbers == 0)
    {
        this->atom_numbers = atom_numbers;
    }
    if (scanf_ret == 0)
    {
        std::string error_reason = "Reason:\n\tthe format of the coordinate_in_file (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Coordinate_In_File", error_reason.c_str());
    }
    else  if (scanf_ret == 1)
    {
        sys.start_time = 0;
    }

    controller.printf("        atom_numbers is %d\n", this->atom_numbers);
    controller.printf("        system start_time is %lf\n", this->sys.start_time);
    Malloc_Safely((void**)&coordinate, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&crd, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&last_crd, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    cudaMemset(last_crd, 0,  sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));

    for (int i = 0; i < atom_numbers; i++)
    {
        scanf_ret = fscanf(fp, "%f %f %f", &coordinate[i].x, &coordinate[i].y, &coordinate[i].z);
        if (scanf_ret != 3)
        {
            std::string error_reason = "Reason:\n\tthe format of the coordinate_in_file (";
            error_reason += file_name;
            error_reason += ") is not right\n";
            controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Coordinate_In_File", error_reason.c_str());
        }
    }
    scanf_ret = fscanf(fp, "%f %f %f", &sys.box_length.x, &sys.box_length.y, &sys.box_length.z);
    if (scanf_ret != 3)
    {
        std::string error_reason = "Reason:\n\tthe format of the coordinate_in_file (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Coordinate_In_File", error_reason.c_str());
    }
    controller.printf("        box_length is\n            x: %f\n            y: %f\n            z: %f\n", sys.box_length.x, sys.box_length.y, sys.box_length.z);
    cudaMemcpy(crd, coordinate, sizeof(VECTOR)* atom_numbers, cudaMemcpyHostToDevice);
    controller.printf("    End reading coordinate_in_file\n\n");
    fclose(fp);
}
void MD_INFORMATION::Read_Rst7(const char* file_name, int irest, CONTROLLER controller)
{
    FILE *fin = NULL;
    Open_File_Safely(&fin, file_name, "r");
    controller.printf("    Start reading AMBER rst7:\n");
    char lin[CHAR_LENGTH_MAX];
    int atom_numbers = 0;
    char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    int has_vel = 0;
    int scanf_ret = sscanf(lin, "%d %lf", &atom_numbers, &sys.start_time);
    if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
    {
        controller.Throw_SPONGE_Error(spongeErrorConflictingCommand, "MD_INFORMATION::Read_Rst7", ATOM_NUMBERS_DISMATCH);
    }
    else if (this->atom_numbers == 0)
    {
        this->atom_numbers = atom_numbers;
    }
    if (scanf_ret == 0)
    {
        std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
        error_reason += file_name;
        error_reason += ") is not right\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Rst7", error_reason.c_str());
    }
    else  if (scanf_ret == 2)
    {
        has_vel = 1;
    }
    else
    {
        sys.start_time = 0;
    }

    Malloc_Safely((void**)&coordinate, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&crd, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&last_crd, sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    cudaMemset(last_crd, 0,  sizeof(VECTOR)*(this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Malloc_Safely((void**)&velocity, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Cuda_Malloc_Safely((void**)&vel, sizeof(VECTOR) * (this->atom_numbers + no_direct_interaction_virtual_atom_numbers));

    controller.printf("        atom_numbers is %d\n", this->atom_numbers);
    controller.printf("        system start time is %lf\n", this->sys.start_time);

    if (has_vel == 0 || irest == 0)
    {
        controller.printf("        All velocity will be set to 0\n");
    }


    for (int i = 0; i < this->atom_numbers; i = i + 1)
    {
        scanf_ret = fscanf(fin, "%f %f %f",
            &this->coordinate[i].x,
            &this->coordinate[i].y,
            &this->coordinate[i].z);
        if (scanf_ret != 3)
        {
            std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
            error_reason += file_name;
            error_reason += ") is not right (missing the coordinate of atom ";
            error_reason += i;
            error_reason += ")\n";
            controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Rst7", error_reason.c_str());
        }
    }
    if (has_vel)
    {
        for (int i = 0; i < this->atom_numbers; i = i + 1)
        {
            scanf_ret = fscanf(fin, "%f %f %f",
                &this->velocity[i].x,
                &this->velocity[i].y,
                &this->velocity[i].z);
            if (scanf_ret != 3)
            {
                std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
                error_reason += file_name;
                error_reason += ") is not right (missing the velocity of atom ";
                error_reason += i;
                error_reason += ")\n";
                controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Rst7", error_reason.c_str());
            }
        }
    }
    if (irest == 0 || !has_vel)
    {
        for (int i = 0; i < this->atom_numbers; i = i + 1)
        {
            this->velocity[i].x = 0.0;
            this->velocity[i].y = 0.0;
            this->velocity[i].z = 0.0;
        }
    }
    scanf_ret = fscanf(fin, "%f %f %f", &this->sys.box_length.x, &this->sys.box_length.y, &this->sys.box_length.z);
    if (scanf_ret != 3)
    {
        std::string error_reason = "Reason:\n\tthe format of the amber_rst7 (";
        error_reason += file_name;
        error_reason += ") is not right (missing the box information)\n";
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat, "MD_INFORMATION::Read_Rst7", error_reason.c_str());
    }
    controller.printf("        system size is %f %f %f\n", this->sys.box_length.x, this->sys.box_length.y, this->sys.box_length.z);
    cudaMemcpy(this->crd, this->coordinate, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->vel, this->velocity, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyHostToDevice);
    fclose(fin);
    controller.printf("    End reading AMBER rst7\n\n");
}

void MD_INFORMATION::trajectory_output::Append_Crd_Traj_File(FILE *fp)
{
    if (md_info->is_initialized)
    {
        md_info->Crd_Vel_Device_To_Host();
        if (fp == NULL)
        {
            fp = crd_traj;
        }
        fwrite(&md_info->coordinate[0].x, sizeof(VECTOR), md_info->atom_numbers, fp);
    }
}

// 20210827用于输出速度和力
void MD_INFORMATION::trajectory_output::Append_Frc_Traj_File(FILE *fp)
{
    if (md_info->is_initialized)
    {
        cudaMemcpy(md_info->force, md_info->frc, sizeof(VECTOR)*md_info->atom_numbers, cudaMemcpyDeviceToHost);
        if (fp == NULL)//默认的frc输出位置
        {
            fp = frc_traj;
            if (fp != NULL)
            {
                fwrite(&md_info->force[0].x, sizeof(VECTOR), md_info->atom_numbers, fp);
            }
        }
        else
        {
            fwrite(&md_info->force[0].x, sizeof(VECTOR), md_info->atom_numbers, fp);
        }
    }
}
void MD_INFORMATION::trajectory_output::Append_Vel_Traj_File(FILE *fp)
{
    if (md_info->is_initialized)
    {
        cudaMemcpy(md_info->velocity, md_info->vel, sizeof(VECTOR)*md_info->atom_numbers, cudaMemcpyDeviceToHost);
        if (fp == NULL)//默认的vel输出位置
        {
            fp = vel_traj;
            if (fp != NULL)
            {
                fwrite(&md_info->velocity[0].x, sizeof(VECTOR), md_info->atom_numbers, fp);
            }
        }
        else
        {
            fwrite(&md_info->velocity[0].x, sizeof(VECTOR), md_info->atom_numbers, fp);
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_Box_Traj_File(FILE *fp)
{
    if (md_info->is_initialized)
    {
        if (fp == NULL)
        {
            fp = box_traj;
        }
        fprintf(fp, "%f %f %f %.0f %.0f %.0f\n", md_info->sys.box_length.x, md_info->sys.box_length.y, md_info->sys.box_length.z, 90.0f, 90.0f, 90.0f);
    }
}

void MD_INFORMATION::trajectory_output::Export_Restart_File(const char *rst7_name)
{
    if (!md_info->is_initialized)
        return;

    char filename[CHAR_LENGTH_MAX];
    if (rst7_name == NULL)
        strcpy(filename, restart_name);
    else
        strcpy(filename, rst7_name);
    md_info->Crd_Vel_Device_To_Host();
    if (amber_irest >= 0)
    {
        const char *sys_name = md_info->md_name;
        FILE *lin = NULL;
        Open_File_Safely(&lin, filename, "w");
        fprintf(lin, "%s\n", sys_name);
        fprintf(lin, "%8d %.3lf\n", md_info->atom_numbers, md_info->sys.Get_Current_Time());
        int s = 0;
        for (int i = 0; i < md_info->atom_numbers; i = i + 1)
        {
            fprintf(lin, "%12.7f%12.7f%12.7f", md_info->coordinate[i].x, md_info->coordinate[i].y, md_info->coordinate[i].z);
            s = s + 1;
            if (s == 2)
            {
                s = 0;
                fprintf(lin, "\n");
            }
        }
        if (s == 1)
        {
            s = 0;
            fprintf(lin, "\n");
        }
        for (int i = 0; i < md_info->atom_numbers; i = i + 1)
        {
            fprintf(lin, "%12.7f%12.7f%12.7f", md_info->velocity[i].x, md_info->velocity[i].y, md_info->velocity[i].z);
            s = s + 1;
            if (s == 2)
            {
                s = 0;
                fprintf(lin, "\n");
            }
        }
        if (s == 1)
        {
            s = 0;
            fprintf(lin, "\n");
        }
        fprintf(lin, "%12.7f%12.7f%12.7f", (float)md_info->sys.box_length.x, (float)md_info->sys.box_length.y, (float)md_info->sys.box_length.z);
        fprintf(lin, "%12.7f%12.7f%12.7f", (float)90., (float)90., (float)90.);
        fclose(lin);
    }
    else
    {
        FILE *lin = NULL;
        FILE *lin2 = NULL;
        std::string buffer;
        buffer = std::string(filename) + "_coordinate.txt";
        Open_File_Safely(&lin, buffer.c_str(), "w");
        buffer = std::string(filename) + "_velocity.txt";
        Open_File_Safely(&lin2, buffer.c_str(), "w");
        fprintf(lin, "%d %.3lf\n", md_info->atom_numbers, md_info->sys.Get_Current_Time());
        fprintf(lin2, "%d %.3lf\n", md_info->atom_numbers, md_info->sys.Get_Current_Time());
        for (int i = 0; i < md_info->atom_numbers; i++)
        {
            fprintf(lin, "%12.7f %12.7f %12.7f\n", md_info->coordinate[i].x, md_info->coordinate[i].y, md_info->coordinate[i].z);
            fprintf(lin2, "%12.7f %12.7f %12.7f\n", md_info->velocity[i].x, md_info->velocity[i].y, md_info->velocity[i].z);
        }
        fprintf(lin, "%12.7f %12.7f %12.7f %12.7f %12.7f %12.7f", md_info->sys.box_length.x, md_info->sys.box_length.y, md_info->sys.box_length.z, 90.0f, 90.0f, 90.0f);
        fclose(lin);
        fclose(lin2);
    }
}


void MD_INFORMATION::Update_Volume(double factor)
{
    sys.box_length = factor * sys.box_length;
    pbc.uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * sys.box_length;
    MD_Information_Crd_To_Uint_Crd();
}

void MD_INFORMATION::Update_Box_Length(VECTOR factor)
{
    sys.box_length.x = factor.x * sys.box_length.x;
    sys.box_length.y = factor.y * sys.box_length.y;
    sys.box_length.z = factor.z * sys.box_length.z;
    pbc.uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * sys.box_length;
    MD_Information_Crd_To_Uint_Crd();
}

float MD_INFORMATION::system_information::Get_Density()
{
    density = total_mass * 1e24f / 6.023e23f / Get_Volume();
    return density;
}

double MD_INFORMATION::system_information::Get_Current_Time()
{
    current_time = start_time + (double)dt_in_ps * steps;
    return current_time;
}

float MD_INFORMATION::system_information::Get_Volume()
{
    volume = box_length.x * box_length.y * box_length.z;
    return volume;
}

void MD_INFORMATION::MD_Information_Crd_To_Uint_Crd()
{
    CUDA_LAUNCH(Crd_To_Uint_Crd,
        dim3((atom_numbers + no_direct_interaction_virtual_atom_numbers + 127) / 128),
        dim3(128),
        this->atom_numbers + no_direct_interaction_virtual_atom_numbers,
        1.0f / sys.box_length,
        crd,
        uint_crd);
}

void MD_INFORMATION::NVE_iteration::Leap_Frog()
{
    if (max_velocity <= 0)
    {
        CUDA_LAUNCH(MD_Iteration_Leap_Frog,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 128))),
            dim3(128),
            md_info->atom_numbers,
            md_info->vel,
            md_info->crd,
            md_info->frc,
            md_info->acc,
            md_info->d_mass_inverse,
            md_info->dt);
    }
    else
    {
        CUDA_LAUNCH(MD_Iteration_Leap_Frog_With_Max_Velocity,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 128))),
            dim3(128),
            md_info->atom_numbers,
            md_info->vel,
            md_info->crd,
            md_info->frc,
            md_info->acc,
            md_info->d_mass_inverse,
            md_info->dt,
            max_velocity);
    }
}

void MD_INFORMATION::MINIMIZATION_iteration::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    if (md_info->mode == MINIMIZATION)
    {
        controller->printf("    Start initializing minimization:\n");
        max_move = 0.1f;
        if (controller[0].Command_Exist("minimization_max_move"))
        {
            controller->Check_Float("minimization", "max_move", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            max_move = atof(controller[0].Command("minimization_max_move"));
        }
        controller->printf("        minimization max move is %f A\n", max_move);

        momentum_keep = 0;
        if (controller[0].Command_Exist("minimization_momentum_keep"))
        {
            controller->Check_Float("minimization", "momentum_keep", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            momentum_keep = atof(controller[0].Command("minimization_momentum_keep"));
        }
        controller->printf("        minimization momentum keep is %f\n", momentum_keep);

        dynamic_dt = 1;
        if (controller[0].Command_Exist("minimization_dynamic_dt"))
        {
            controller->Check_Int("minimization", "dynamic_dt", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            dynamic_dt = atoi(controller[0].Command("minimization_dynamic_dt"));
        }
        controller->printf("        minimization dynamic dt is %d\n", dynamic_dt);

        if (dynamic_dt)
        {
            dt_decreasing_rate = 0.01;
            if (controller[0].Command_Exist("minimization_dt_decreasing_rate"))
            {
                controller->Check_Float("minimization", "dt_decreasing_rate", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                dt_decreasing_rate = atof(controller[0].Command("minimization_dt_decreasing_rate"));
            }
            controller->printf("        minimization dt decreasing rate is %f\n", dt_decreasing_rate);

            dt_increasing_rate = 1.01 + rand() / RAND_MAX;
            if (controller[0].Command_Exist("minimization_dt_increasing_rate"))
            {
                controller->Check_Float("minimization", "dt_increasing_rate", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                dt_increasing_rate = atof(controller[0].Command("minimization_dt_increasing_rate"));
            }
            controller->printf("        minimization dt increasing rate is %f\n", dt_increasing_rate);
    
           dt_factor = 1e-4 + 1e-2 * rand() / RAND_MAX;
            if (controller[0].Command_Exist("minimization_dt_factor"))
            {
                controller->Check_Float("minimization", "dt_factor", "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                dt_factor = atof(controller[0].Command("minimization_dt_factor"));
            }
            controller->printf("        minimization dt factor is %f\n", dt_factor);
            last_decrease_step = 0;
            Cuda_Malloc_Safely((void**)&d_max_force, sizeof(float));
        }
        controller->printf("    End initializing minimization\n\n");
    }
}

static __global__ void find_max_frc(int atom_numbers,  VECTOR *frc,  float *d_max_force)
{
    __shared__ float maxf[1024];
    float temp_f, temp_max = 0;
    VECTOR temp_frc;
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
    {
        temp_frc = frc[i];
        temp_f = norm3df(temp_frc.x, temp_frc.y, temp_frc.z);
        if (temp_f > temp_max)
        {
            temp_max = temp_f;
        }
    }
    maxf[threadIdx.x] = temp_max;
    __syncthreads();
    int delta = 1024;
    for (int i = 0; i < 10; i += 1)
    {
        delta >>= 1;
        if (threadIdx.x < delta)
        {
            maxf[threadIdx.x] = fmaxf(maxf[threadIdx.x], maxf[threadIdx.x + delta]);
        }
    }
    d_max_force[0] = maxf[0];
}

void MD_INFORMATION::MINIMIZATION_iteration::Gradient_Descent()
{
    if (dynamic_dt)
    {
        if (md_info->sys.steps != 1)
        {
            if (last_decrease_step == 0)
            {
                if (last_potential > md_info->sys.h_potential)
                {
                    md_info->dt *= dt_increasing_rate;
                }
                else
                {
                    if (md_info->dt > 1e-8f)
                    {
                        md_info->dt *= dt_decreasing_rate;
                    }
                    else
                    {
                        last_decrease_step = md_info->sys.steps;
                    }
                }
            }
            else
            {
                CUDA_LAUNCH(find_max_frc, dim3(1), dim3(1024), md_info->atom_numbers, md_info->frc, d_max_force);
                cudaMemcpy(&h_max_force, d_max_force, sizeof(float), cudaMemcpyDeviceToHost);
                md_info->dt = sqrtf(dt_factor / h_max_force);
                if (md_info->sys.steps - last_decrease_step > 100)
                {
                    last_decrease_step = 0;
                }
            }
        }
        
        if (isnan(md_info->sys.h_potential) || (last_potential != 0 && (md_info->sys.h_potential - last_potential) / fabsf(last_potential) > 10))
        {
            md_info->sys.h_potential = last_potential;
            cudaMemcpy(md_info->crd, md_info->last_crd, sizeof(VECTOR) * md_info->atom_numbers, cudaMemcpyDeviceToDevice);
            cudaMemset(md_info->vel, 0, sizeof(VECTOR) * md_info->atom_numbers);
            if (md_info->dt > 1e-8)
            {
                md_info->dt *= dt_decreasing_rate;
            }
            else
            {
                last_decrease_step = md_info->sys.steps;
            }
        }
        else
        {
            last_potential = md_info->sys.h_potential;
            cudaMemcpy(md_info->last_crd, md_info->crd, sizeof(VECTOR) * md_info->atom_numbers, cudaMemcpyDeviceToDevice);
        }
    }

    if (max_move <= 0)
    {
        CUDA_LAUNCH(MD_Iteration_Gradient_Descent,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 128))),
            dim3(128),
            md_info->atom_numbers,
            md_info->crd,
            md_info->frc,
            md_info->d_mass_inverse,
            md_info->dt,
            md_info->vel,
            momentum_keep);
    }
    else
    {
        CUDA_LAUNCH(MD_Iteration_Gradient_Descent_With_Max_Move,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 128))),
            dim3(128),
            md_info->atom_numbers,
            md_info->crd,
            md_info->frc,
            md_info->d_mass_inverse,
            md_info->dt,
            md_info->vel,
            momentum_keep,
            max_move);
    }
}

void MD_INFORMATION::MINIMIZATION_iteration::Check_Nan()
{
    if (dynamic_dt)
    {
        float temperature = md_info->res.Get_Residue_Temperature();
        if (isnan(temperature) || (last_temperature > 0.1 && temperature / last_temperature > 10) || (last_temperature < 0.01 && temperature > 0.1))
        {
            md_info->sys.h_potential = last_potential;
            cudaMemcpy(md_info->crd, md_info->last_crd, sizeof(VECTOR) * md_info->atom_numbers, cudaMemcpyDeviceToDevice);
            cudaMemset(md_info->vel, 0, sizeof(VECTOR) * md_info->atom_numbers);
            if (md_info->dt > 1e-8)
            {
                md_info->dt *= dt_decreasing_rate;
            }
            else
            {
                last_decrease_step = md_info->sys.steps;
            }
        }
        last_temperature = temperature;
    }
}

void MD_INFORMATION::RERUN_information::Initial(CONTROLLER *controller, MD_INFORMATION *md_info)
{
    this->md_info = md_info;
    if (md_info->mode == RERUN)
    {
        controller->printf("    Start initializing rerun:\n");
        if (!Open_File_Safely(&traj_file, controller->Command(TRAJ_COMMAND), "rb"))
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                "MD_INFORMATION::RERUN_information::Initial",
                "Reason:\n\tno trajectory information found (command 'crd' required)");
        }
        else
        {
            controller->printf("        Open rerun coordinate trajectory\n");
        }
        controller->Set_File_Buffer(traj_file, sizeof(VECTOR) * md_info->atom_numbers);
        if (!Open_File_Safely(&box_file, controller->Command(BOX_TRAJ_COMMAND), "r"))
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                "MD_INFORMATION::RERUN_information::Initial",
                "Reason:\n\tno box information found (command 'box' required)");
        }
        else
        {
            controller->printf("        Open rerun box trajectory\n");
        }
        controller->Set_File_Buffer(box_file, sizeof(char) * 50);
        start_frame = 0;
        if (controller->Command_Exist("rerun_start"))
        {
            controller->Check_Float("rerun_start", "MD_INFORMATION::RERUN_information::Initial");
            start_frame = atoi(controller->Command("rerun_start"));
        }
        strip_frame = 0;
        if (controller->Command_Exist("rerun_strip"))
        {
            controller->Check_Float("rerun_strip", "MD_INFORMATION::RERUN_information::Initial");
            strip_frame = atoi(controller->Command("rerun_strip"));
        }
        md_info->sys.step_limit = INT_MAX;
        controller->printf("    End initializing rerun\n\n");
    }
}

void MD_INFORMATION::RERUN_information::Iteration(int strip)
{
    int n;
    int scanf_box;
    if (strip < 0)
    {
        strip = this->strip_frame;
    }
    for (int i = 0; i < strip; i++)
    {
        n = fread(this->md_info->coordinate, sizeof(VECTOR), this->md_info->atom_numbers, traj_file);
        if (md_info->pbc.pbc)
        {
            scanf_box = fscanf(box_file, "%f %f %f %*f %*f %*f",
                &box_length_change_factor.x,
                &box_length_change_factor.y,
                &box_length_change_factor.z);
        }
        else
            box_length_change_factor = md_info->sys.box_length;
        md_info->sys.steps += 1;
    }
    n = fread(this->md_info->coordinate, sizeof(VECTOR), this->md_info->atom_numbers, traj_file);
    if (md_info->pbc.pbc)
    {
        scanf_box = fscanf(box_file, "%f %f %f %*f %*f %*f",
            &box_length_change_factor.x,
            &box_length_change_factor.y,
            &box_length_change_factor.z);
    }
    else
    {
        box_length_change_factor = md_info->sys.box_length;
        scanf_box = 3;
    }
    if (n != this->md_info->atom_numbers || scanf_box != 3)
    {
        md_info->sys.step_limit = md_info->sys.steps;
    }
    box_length_change_factor = box_length_change_factor / md_info->sys.box_length;  
    cudaMemcpy(this->md_info->crd, this->md_info->coordinate, sizeof(VECTOR)* this->md_info->atom_numbers, cudaMemcpyHostToDevice);
}

void MD_INFORMATION::NVE_iteration::Velocity_Verlet_1()
{
    CUDA_LAUNCH(MD_Iteration_Speed_Verlet_1,
        dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 32))),
        dim3(32),
        md_info->atom_numbers,
        0.5 * md_info->dt,
        md_info->dt,
        md_info->acc,
        md_info->vel,
        md_info->crd,
        md_info->frc);
}

void MD_INFORMATION::NVE_iteration::Velocity_Verlet_2()
{
    if (max_velocity <= 0)
    {
        CUDA_LAUNCH(MD_Iteration_Speed_Verlet_2,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 32))),
            dim3(32),
            md_info->atom_numbers,
            0.5 * md_info->dt,
            md_info->d_mass_inverse,
            md_info->frc,
            md_info->vel,
            md_info->acc);
    }
    else
    {
        CUDA_LAUNCH(MD_Iteration_Speed_Verlet_2_With_Max_Velocity,
            dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 32))),
            dim3(32),
            md_info->atom_numbers,
            0.5 * md_info->dt,
            md_info->d_mass_inverse,
            md_info->frc,
            md_info->vel,
            md_info->acc,
            max_velocity);
    }
}


float MD_INFORMATION::system_information::Get_Total_Atom_Ek(int is_download)
{
    CUDA_LAUNCH(MD_Atom_Ek,
        dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 32.))),
        dim3(32),
        md_info->atom_numbers,
        md_info->d_atom_ek,
        md_info->vel,
        md_info->d_mass);
    CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), md_info->atom_numbers, md_info->d_atom_ek, d_sum_of_atom_ek);
    if (is_download)
    {
        cudaMemcpy(&h_sum_of_atom_ek, d_sum_of_atom_ek, sizeof(float), cudaMemcpyDeviceToHost);
        return h_sum_of_atom_ek;
    }
    else
    {
        return 0;
    }    
}

float MD_INFORMATION::system_information::Get_Atom_Temperature()
{
    h_temperature = Get_Total_Atom_Ek() * 2. / CONSTANT_kB / freedom;
    return h_temperature;
}

float MD_INFORMATION::residue_information::Get_Total_Residue_Ek(int is_download)
{
    CUDA_LAUNCH(MD_Residue_Ek,
        dim3(static_cast<unsigned int>(ceilf((float)residue_numbers / 32.))),
        dim3(32),
        residue_numbers,
        d_res_start,
        d_res_end,
        res_ek_energy,
        md_info->vel,
        md_info->d_mass);
    CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), residue_numbers, res_ek_energy, sigma_of_res_ek);
    if (is_download)
    {
        cudaMemcpy(h_sigma_of_res_ek, sigma_of_res_ek, sizeof(float), cudaMemcpyDeviceToHost);
        return h_sigma_of_res_ek[0];
    }
    else
    {
        return 0;
    }
}

float MD_INFORMATION::residue_information::Get_Residue_Temperature()
{
    h_temperature = Get_Total_Residue_Ek() * 2. / CONSTANT_kB / residue_numbers / 3;
    return h_temperature;
}

void MD_INFORMATION::residue_information::Residue_Crd_Map(VECTOR scaler)
{
    CUDA_LAUNCH(Get_Origin,
        dim3((residue_numbers + 1023) / 1024),
        dim3(1024),
        residue_numbers,
        d_res_start,
        d_res_end,
        md_info->crd,
        md_info->d_mass,
        d_mass_inverse,
        d_center_of_mass);
    CUDA_LAUNCH(Map_Center_Of_Mass,
        dim3((residue_numbers + 63) / 64),
        dim3(64, 16),
        residue_numbers,
        d_res_start,
        d_res_end,
        scaler,
        d_center_of_mass,
        md_info->sys.box_length,
        md_info->crd,
        (int*)NULL);
}


void MD_INFORMATION::MD_Reset_Atom_Energy_And_Virial_And_Force()
{
    need_potential = 0;
    cudaMemset(d_atom_energy, 0, sizeof(float)* atom_numbers);
    cudaMemset(sys.d_potential, 0, sizeof(float));

    need_pressure = 0;
    cudaMemset(d_atom_virial, 0, sizeof(float)* atom_numbers);
    cudaMemset(sys.d_virial, 0, sizeof(float));

    cudaMemset(frc, 0, sizeof(VECTOR)* atom_numbers);
}


void MD_INFORMATION::Calculate_Pressure_And_Potential_If_Needed(int is_download)
{
    if (need_pressure > 0)
    {
        sys.Get_Pressure(is_download);
    }
    if (need_potential > 0)
    {
        sys.Get_Potential(is_download);
    }
}

void MD_INFORMATION::Scale_Position_To_Center(float scaler)
{
    CUDA_LAUNCH(Scale_Position_To_Center_CUDA,
        dim3((atom_numbers + 1023) / 1024),
        dim3(1024),
        atom_numbers,
        scaler,
        crd,
        0.5f * scaler * sys.box_length);
}

float MD_INFORMATION::system_information::Get_Pressure(int is_download)
{
    //计算动能
    CUDA_LAUNCH(MD_Atom_Ek,
        dim3(static_cast<unsigned int>(ceilf((float)md_info->atom_numbers / 32.))),
        dim3(32),
        md_info->atom_numbers,
        md_info->d_atom_ek,
        md_info->vel,
        md_info->d_mass);
    CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), md_info->atom_numbers, md_info->d_atom_ek, d_sum_of_atom_ek);

    //计算维里
    CUDA_LAUNCH(Add_Sum_List, dim3(1), dim3(1024), md_info->atom_numbers, md_info->d_atom_virial, d_virial);

    //合并起来
    CUDA_LAUNCH(Calculate_Pressure_Cuda, dim3(1), dim3(1), 1.0 / Get_Volume(), d_sum_of_atom_ek, d_virial, d_pressure);

    if (is_download)
    {
        cudaMemcpy(&h_pressure, d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        return h_pressure;
    }
    else
    {
        return 0;
    }
}

float MD_INFORMATION::system_information::Get_Potential(int is_download)
{

    CUDA_LAUNCH(Add_Sum_List, dim3(1), dim3(1024), md_info->atom_numbers, md_info->d_atom_energy, d_potential);

    if (is_download)
    {
        cudaMemcpy(&h_potential, d_potential, sizeof(float), cudaMemcpyDeviceToHost);
        return h_potential;
    }
    else
    {
        return 0;
    }
}

void MD_INFORMATION::MD_Information_Frc_Device_To_Host()
{
    cudaMemcpy(this->force, this->frc, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyDeviceToHost);
}

void MD_INFORMATION::MD_Information_Frc_Host_To_Device()
{
    cudaMemcpy(this->frc, this->force, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyHostToDevice);
}

void MD_INFORMATION::Crd_Vel_Device_To_Host(int forced)
{
    if (output.current_crd_synchronized_step != sys.steps || forced)
    {    
        output.current_crd_synchronized_step = sys.steps;
        if (pbc.pbc)
            cudaMemcpy(this->coordinate, this->crd, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(this->coordinate, this->pbc.nopbc_crd, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyDeviceToHost);
        cudaMemcpy(this->velocity, this->vel, sizeof(VECTOR)*this->atom_numbers, cudaMemcpyDeviceToHost);
    }
}

void MD_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (mode == RERUN)
    {
        controller->Step_Print("frame", this->sys.steps);
    }
    else
    {
        controller->Step_Print("step", this->sys.steps);
        controller->Step_Print("time", this->sys.Get_Current_Time());
        controller->Step_Print("temperature", this->res.Get_Residue_Temperature());
    }

}

void MD_INFORMATION::Clear()
{

}

static void Get_Atom_Group_From_Edges(const int atom_numbers, const int* edges,
    const int* first_edge, const int* edge_next, CPP_ATOM_GROUP& mol_atoms, int* belongs)
{

    std::deque<int> queue;
    std::vector<int> visited(atom_numbers, 0);
    int atom;
    int edge_count;
    for (int i = 0; i < atom_numbers; i++)
    {
        if (!visited[i])
        {
            std::vector<int> atoms;
            visited[i] = 1;
            queue.push_back(i);
            while (!queue.empty())
            {
                atom = queue[0];
                belongs[atom] = mol_atoms.size();
                atoms.push_back(atom);
                queue.pop_front();
                edge_count = first_edge[atom];
                while (edge_count != -1)
                {
                    atom = edges[edge_count];
                    if (!visited[atom])
                    {
                        queue.push_back(atom);
                        visited[atom] = 1;
                    }
                    edge_count = edge_next[edge_count];
                }
            }
            mol_atoms.push_back(atoms);
        }
    }
}

static void Get_Molecule_Atoms(CONTROLLER* controller, int atom_numbers, CONECT connectivity,
    CPP_ATOM_GROUP& mol_atoms, std::vector<int>& molecule_belongings)
{
    //分子拓扑是一个无向图，邻接表进行描述
    int edge_numbers = 0;
    for (int i = 0; i < atom_numbers; i++)
    {
        edge_numbers += connectivity[i].size();
    }
    edge_numbers *= 2;
    int* first_edge = NULL; //每个原子的第一个边（链表的头）
    int* edges = NULL;  //每个边的序号
    int* edge_next = NULL; //每个原子的边（链表结构）
    Malloc_Safely((void**)&first_edge, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&edges, sizeof(int) * edge_numbers);
    Malloc_Safely((void**)&edge_next, sizeof(int) * edge_numbers);
    //初始化链表
    for (int i = 0; i < atom_numbers; i++)
    {
        first_edge[i] = -1;
    }
    int atom_i, atom_j, edge_count = 0;
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        std::set<int> conect_i = connectivity[atom_i];
        for (auto iter = conect_i.begin(); iter != conect_i.end(); iter++)
        {
            atom_j = *iter;
            edge_next[edge_count] = first_edge[atom_i];
            first_edge[atom_i] = edge_count;
            edges[edge_count] = atom_j;
            edge_count++;
        }
    }
    if (controller->Command_Exist("make_output_whole"))
    {
        std::string temp = string_strip(controller->Original_Command("make_output_whole"));
        for (std::string aword : string_split(temp, " "))
        {
            std::vector<std::string> atomij = string_split(string_strip(aword), "-");
            if (atomij.size() != 2 || !is_str_int(atomij[0].c_str()) || !is_str_int(atomij[1].c_str()))
            {
                controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "Move_Crd_Nearest_From_Exclusions_Host",
                    "Reason:\n\t'make_output_whole' should provide atoms in the format of atom_i-atom_j");
            }
            atom_i = atoi(atomij[0].c_str());
            atom_j = atoi(atomij[1].c_str());
            edge_next[edge_count] = first_edge[atom_i];
            first_edge[atom_i] = edge_count;
            edges[edge_count] = atom_j;
            edge_count++;
            edge_next[edge_count] = first_edge[atom_j];
            first_edge[atom_j] = edge_count;
            edges[edge_count] = atom_i;
            edge_count++;
        }
    }
    Get_Atom_Group_From_Edges(atom_numbers, edges, first_edge, edge_next, mol_atoms, &molecule_belongings[0]);
    free(first_edge);
    free(edges);
    free(edge_next);
}

static std::vector<int> Check_Periodic_Molecules(CPP_ATOM_GROUP mol_atoms, const VECTOR* crd, const VECTOR box_length)
{
    std::vector<int> periodic_mols;
    std::vector<int> atoms;
    VECTOR abosolute_dr;
    for (int i = 0; i < mol_atoms.size(); i++)
    {
        atoms = mol_atoms[i];
        abosolute_dr.x = 0;
        abosolute_dr.y = 0;
        abosolute_dr.z = 0;
        for (int j = 1; j < atoms.size(); j++)
        {
            int atom = atoms[j];
            int atom_front = atoms[j - 1];
            abosolute_dr = abosolute_dr + Get_Periodic_Displacement(crd[atom], crd[atom_front], box_length);
        }
        periodic_mols.push_back(abosolute_dr.x * abosolute_dr.x >= box_length.x * box_length.x
            || abosolute_dr.y * abosolute_dr.y >= box_length.y * box_length.y
            || abosolute_dr.z * abosolute_dr.z >= box_length.z * box_length.z);
    }
    return periodic_mols;
}

static void Move_Crd_Nearest_From_Connectivity(CPP_ATOM_GROUP mol_atoms, VECTOR* crd, 
    const VECTOR box_length, std::vector<int> periodic_molecules)
{
    std::vector<int> atoms;
    for (int i = 0; i < mol_atoms.size(); i++)
    {
        if (periodic_molecules[i])
        {
            continue;
        }
        atoms = mol_atoms[i];
        for (int j = 1; j < atoms.size(); j++)
        {
            int atom = atoms[j];
            int atom_front = atoms[j - 1];
            int map_x = floorf((crd[atom_front].x - crd[atom].x) / box_length.x + 0.5);
            crd[atom].x = crd[atom].x + map_x * box_length.x;
            map_x = floorf((crd[atom_front].y - crd[atom].y) / box_length.y + 0.5);
            crd[atom].y = crd[atom].y + map_x * box_length.y;
            map_x = floorf((crd[atom_front].z - crd[atom].z) / box_length.z + 0.5);
            crd[atom].z = crd[atom].z + map_x * box_length.z;
        }
    }
}

void MD_INFORMATION::molecule_information::Initial(CONTROLLER *controller)
{
    if (!md_info->pbc.pbc)
        return;
    //分子拓扑是一个无向图，邻接表进行描述，通过排除表形成
    CPP_ATOM_GROUP mol_atoms;
    std::vector<int> molecule_belongings(md_info->atom_numbers, 0);
    Get_Molecule_Atoms(controller, md_info->atom_numbers, md_info->sys.connectivity,
        mol_atoms, molecule_belongings);
    molecule_numbers = mol_atoms.size();
    std::vector<int> h_periodicity = Check_Periodic_Molecules(mol_atoms, md_info->coordinate, md_info->sys.box_length);
    Cuda_Malloc_And_Copy_Safely((void**)&d_periodicity, &h_periodicity[0],
        sizeof(int) * molecule_numbers);
    Move_Crd_Nearest_From_Connectivity(mol_atoms, md_info->coordinate, md_info->sys.box_length, h_periodicity);
    cudaMemcpy(md_info->crd, md_info->coordinate, sizeof(VECTOR) * md_info->atom_numbers, cudaMemcpyHostToDevice);

    Malloc_Safely((void**)&h_mass, sizeof(float)*molecule_numbers);
    Malloc_Safely((void**)&h_mass_inverse, sizeof(float)*molecule_numbers);
    Malloc_Safely((void**)&h_atom_start, sizeof(int)*molecule_numbers);
    Malloc_Safely((void**)&h_atom_end, sizeof(int)*molecule_numbers);
    Malloc_Safely((void**)&h_residue_start, sizeof(int)*molecule_numbers);
    Malloc_Safely((void**)&h_residue_end, sizeof(int)*molecule_numbers);
    Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR)*molecule_numbers);

    Cuda_Malloc_Safely((void**)&d_mass, sizeof(float)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_start, sizeof(int)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_end, sizeof(int)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_residue_start, sizeof(int)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_residue_end, sizeof(int)*molecule_numbers);
    Cuda_Malloc_Safely((void**)&d_center_of_mass, sizeof(VECTOR)*molecule_numbers);

    int molecule_j = 0;
    h_atom_start[0] = 0;
    //该判断基于一个分子的所有原子一定在列表里是连续的
    for (int i = 0; i < md_info->atom_numbers; i++)
    {
        if (molecule_belongings[i] != molecule_j)
        {
            if (molecule_belongings[i] < molecule_j)
            {
                char error_reason[CHAR_LENGTH_MAX];
                sprintf(error_reason, "Reason:\n\tthe indexes of atoms in the same one molecule should be continous, and atom #%d is not right\n", i);
                controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "MD_INFORMATION::molecule_information::Initial", error_reason);
            }
            h_atom_end[molecule_j] = i;
            molecule_j += 1;
            if (molecule_j < molecule_numbers)
                h_atom_start[molecule_j] = i;
        }
    }
    h_atom_end[molecule_numbers - 1] = md_info->atom_numbers;
    
    molecule_j = 0;
    h_residue_start[0] = 0;
    //该判断基于一个分子的所有残基一定在列表里是连续的，且原子在残基里也是连续的
    for (int i = 0; i < md_info->res.residue_numbers; i++)
    {
        if (md_info->res.h_res_start[i] == h_atom_end[molecule_j])
        {
            h_residue_end[molecule_j] = i;
            molecule_j += 1;
            if (molecule_j < molecule_numbers)
                h_residue_start[molecule_j] = i;
        }
    }
    h_residue_end[molecule_numbers - 1] = md_info->res.residue_numbers;

    for (int i = 0; i < molecule_numbers; i++)
    {
        h_mass[i] = 0;
        for (molecule_j = h_atom_start[i]; molecule_j < h_atom_end[i]; molecule_j++)
        {
            h_mass[i] += md_info->h_mass[molecule_j];
        }
        h_mass_inverse[i] = 1.0f / h_mass[i];
    }

    cudaMemcpy(d_mass, h_mass, sizeof(float)*molecule_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_inverse, h_mass_inverse, sizeof(float)*molecule_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_start, h_atom_start, sizeof(int)*molecule_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_end, h_atom_end, sizeof(int)*molecule_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residue_start, h_residue_start, sizeof(int)*molecule_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residue_end, h_residue_end, sizeof(int)*molecule_numbers, cudaMemcpyHostToDevice);
}

static __global__ void NoPBC_Mapping(const int atom_numbers, VECTOR* crd, VECTOR* last_crd,
 VECTOR* nopbc_crd, VECTOR box_length)
{
    int atom_i = threadIdx.x + blockDim.x * blockIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR crd_i = crd[atom_i];
        nopbc_crd[atom_i] = nopbc_crd[atom_i] + Get_Periodic_Displacement(crd_i, last_crd[atom_i], box_length);
        crd_i.x = crd_i.x - floorf(crd_i.x / box_length.x) * box_length.x;
        crd_i.y = crd_i.y - floorf(crd_i.y / box_length.y) * box_length.y;
        crd_i.z = crd_i.z - floorf(crd_i.z / box_length.z) * box_length.z;         
        last_crd[atom_i] = crd_i;
        crd[atom_i] = crd_i;
    }
}

void MD_INFORMATION::molecule_information::Molecule_Crd_Map(float scaler)
{
    if (!md_info->pbc.pbc)
    {
        CUDA_LAUNCH(NoPBC_Mapping,
            dim3((md_info->atom_numbers + 1023) / 1024),
            dim3(1024),
            md_info->atom_numbers,
            md_info->crd,
            md_info->last_crd,
            md_info->pbc.nopbc_crd,
            md_info->sys.box_length);
    }
    else
    {
        //为了有一个分子有很多残基，而其他分子都很小这种情况的并行，先求残基的质心
        CUDA_LAUNCH(Get_Origin,
            dim3((md_info->res.residue_numbers + 1023) / 1024),
            dim3(1024),
            md_info->res.residue_numbers,
            md_info->res.d_res_start,
            md_info->res.d_res_end,
            md_info->crd,
            md_info->d_mass,
            md_info->res.d_mass_inverse,
            md_info->res.d_center_of_mass);
        //再用残基的质心求分子的质心
        CUDA_LAUNCH(Get_Origin,
            dim3((molecule_numbers + 1023) / 1024),
            dim3(1024),
            molecule_numbers,
            d_residue_start,
            d_residue_end,
            md_info->res.d_center_of_mass,
            md_info->res.d_mass,
            d_mass_inverse,
            d_center_of_mass);

        CUDA_LAUNCH(Map_Center_Of_Mass,
            dim3((molecule_numbers + 63) / 64),
            dim3(64, 16),
            molecule_numbers,
            d_atom_start,
            d_atom_end,
            scaler,
            d_center_of_mass,
            md_info->sys.box_length,
            md_info->crd,
            d_periodicity);
    }
}


void MD_INFORMATION::molecule_information::Molecule_Crd_Map(VECTOR scaler)
{
    if (!md_info->pbc.pbc)
    {
        CUDA_LAUNCH(NoPBC_Mapping,
            dim3((md_info->atom_numbers + 1023) / 1024),
            dim3(1024),
            md_info->atom_numbers,
            md_info->crd,
            md_info->last_crd,
            md_info->pbc.nopbc_crd,
            md_info->sys.box_length);
    }
    else
    {
        //为了有一个分子有很多残基，而其他分子都很小这种情况的并行，先求残基的质心
        CUDA_LAUNCH(Get_Origin,
            dim3((md_info->res.residue_numbers + 1023) / 1024),
            dim3(1024),
            md_info->res.residue_numbers,
            md_info->res.d_res_start,
            md_info->res.d_res_end,
            md_info->crd,
            md_info->d_mass,
            md_info->res.d_mass_inverse,
            md_info->res.d_center_of_mass);
        //再用残基的质心求分子的质心
        CUDA_LAUNCH(Get_Origin,
            dim3((molecule_numbers + 1023) / 1024),
            dim3(1024),
            molecule_numbers,
            d_residue_start,
            d_residue_end,
            md_info->res.d_center_of_mass,
            md_info->res.d_mass,
            d_mass_inverse,
            d_center_of_mass);
        CUDA_LAUNCH(Map_Center_Of_Mass,
            dim3((molecule_numbers + 63) / 64),
            dim3(64, 16),
            molecule_numbers,
            d_atom_start,
            d_atom_end,
            scaler,
            d_center_of_mass,
            md_info->sys.box_length,
            md_info->crd,
            d_periodicity);
    }
}
