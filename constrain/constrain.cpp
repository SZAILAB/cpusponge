#include "constrain.h"


void CONSTRAIN::Initial_Constrain(CONTROLLER *controller, const int atom_numbers, const float dt, const VECTOR box_length, const float exp_gamma, const int is_Minimization, float *atom_mass, int *system_freedom)
{
    //从传入的参数复制基本信息
    this->atom_numbers = atom_numbers;
    this->dt = dt;
    this->dt_inverse = 1.0 / dt;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    this->volume = box_length.x * box_length.y * box_length.z;


    //确定使用的朗之万热浴，并给定exp_gamma
    this->v_factor = exp_gamma;
    this->x_factor = 0.5*(1. + exp_gamma);

    if (is_Minimization)
    {
        this->v_factor = 0.0f;
    }

    int extra_numbers = 0;
    FILE *fp = NULL;
    //读文件第一个数确认constrain数量，为分配内存做准备
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"), "r");
        int scanf_ret = fscanf(fp, "%d", &extra_numbers);
    }

    constrain_pair_numbers = bond_constrain_pair_numbers + extra_numbers;
    system_freedom[0] -= constrain_pair_numbers;
    controller[0].printf("    constrain pair number is %d\n", constrain_pair_numbers);

    Malloc_Safely((void**)&h_constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers);
    Cuda_Malloc_Safely((void**)&constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers);
    for (int i = 0; i < bond_constrain_pair_numbers; i = i + 1)
    {
        h_constrain_pair[i] = h_bond_pair[i];
        h_constrain_pair[i].constrain_k = h_constrain_pair[i].constrain_k / this->x_factor;
    }
    //读文件存入
    if (fp != NULL)
    {
        int atom_i, atom_j;
        int count = bond_constrain_pair_numbers;
        for (int i = 0; i < extra_numbers; i = i + 1)
        {
            int scanf_ret = fscanf(fp, "%d %d %f", &atom_i, &atom_j, &h_constrain_pair[count].constant_r);
            h_constrain_pair[count].atom_i_serial = atom_i;
            h_constrain_pair[count].atom_j_serial = atom_j;
            h_constrain_pair[count].constrain_k = atom_mass[atom_i] * atom_mass[atom_j] / (atom_mass[atom_i] + atom_mass[atom_j]) / this->x_factor;
            count += 1;
        }
        fclose(fp);
        fp = NULL;
    }

    //上传GPU
    cudaMemcpy(constrain_pair, h_constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers, cudaMemcpyHostToDevice);


    //清空初始化时使用的临时变量
    if (h_bond_pair != NULL)
    {
        free(h_bond_pair);
        h_bond_pair = NULL;
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING CONSTRAIN\n\n");
    is_initialized = 1;
}


void CONSTRAIN::Initial_List(CONTROLLER* controller, CONECT connectivity, PAIR_DISTANCE con_dis, float* atom_mass, const char* module_name)
{
    controller[0].printf("START INITIALIZING CONSTRAIN:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "constrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    constrain_mass = 3.3f;
    if (controller[0].Command_Exist(this->module_name,"in_file"))
        constrain_mass = 0.0f;
    if (controller[0].Command_Exist(this->module_name, "mass"))
    {
        controller->Check_Float(this->module_name, "mass", "CONSTRAIN::Add_HBond_To_Constrain_Pair");
        constrain_mass = atof(controller[0].Command(this->module_name, "mass"));
    }
    //预先分配一个足够大的CONSTRAIN_PAIR用于临时存储
    Malloc_Safely((void**)&h_bond_pair, sizeof(CONSTRAIN_PAIR)* con_dis.size());
    int s = 0;
    float mass_a, mass_b;
    int atom_a, atom_b;
    float r0;
    for (auto &i : con_dis)
    {
        atom_a = i.first.first;
        atom_b = i.first.second;
        r0 = i.second;
        mass_a = atom_mass[atom_a];
        mass_b = atom_mass[atom_b];
        if ((mass_a < constrain_mass && mass_a > 0) || (mass_b < constrain_mass && mass_b > 0))
        {
            h_bond_pair[s].atom_i_serial = atom_a;
            h_bond_pair[s].atom_j_serial = atom_b;
            h_bond_pair[s].constant_r = r0;
            h_bond_pair[s].constrain_k = mass_a * mass_b / (mass_a + mass_b);
            s = s + 1;
        }
    }
    bond_constrain_pair_numbers = s;
    if (controller->Command_Exist(this->module_name, "angle") && controller->Get_Bool(this->module_name, "angle", "CONSTRAIN::Add_HAngle_To_Constrain_Pair"))
    {
        controller->Throw_SPONGE_Error(spongeErrorNotImplemented, "CONSTRAIN::Initial_List", "Reason:\n\tConstraints for angle is not supported from v1.4\n");
    }
}

void CONSTRAIN::Update_Volume(VECTOR box_length)
{
    if (is_initialized)
    {
        uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
        volume = box_length.x * box_length.y * box_length.z;
    }
}

void CONSTRAIN::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(constrain_pair);
        constrain_pair = NULL;

        free(h_constrain_pair);
        h_constrain_pair = NULL;
    }
}
