#include "listed_forces.h"

static LISTED_FORCE* Read_One_Force(CONTROLLER* controller, std::string section, Configuration_Reader* cfg)
{
    LISTED_FORCE* force = new LISTED_FORCE;
    strcpy(force->module_name, section.c_str());
    controller->printf("    reading the listed force named %s\n", force->module_name);
    if (!cfg->Key_Exist(section, "potential"))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "Read_One_Force (listed_forces.cpp)",
            string_format("Reason:\n\tThe potential of the listed force %FORCE% is required ([[ potential ]])\n", 
                { {"FORCE", section} }).c_str());
    }
    force->source_code = cfg->Get_Value(section, "potential");
    if (!cfg->Key_Exist(section, "parameters"))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "Read_One_Force (listed_forces.cpp)",
            string_format("Reason:\n\tThe parameters of the listed force %FORCE% is required ([[ potential ]])\n",
                { {"FORCE", section} }).c_str());
    }
    force->Initialize_Parameters(controller, cfg->Get_Value(section, "parameters"));
    if (cfg->Key_Exist(section, "connected_atoms"))
    {
        controller->printf("        parsing connected atoms of %s\n", force->module_name);
        force->connected_atoms = cfg->Get_Value(section, "connected_atoms");
        if (force->connected_atoms.size() != 2 || force->connected_atoms[0] == force->connected_atoms[1])
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "Read_One_Force (listed_forces.cpp)",
                "Reason:\n\tConnected atoms should be 2 different char");
        }
    }
    if (cfg->Key_Exist(section, "constrain_distance"))
    {
        controller->printf("        parsing constrain distance of %s\n", force->module_name);
        force->constrain_distance = cfg->Get_Value(section, "constrain_distance");
    }
    force->Compile(controller);
    controller->printf("    end reading the listed force named %s\n", force->module_name);
    return force;
}

void LISTED_FORCES::Initial(CONTROLLER* controller, CONECT* connectivity, PAIR_DISTANCE* con_dis, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "listed_forces");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
#ifdef SPONGE_CPU_PHASE1
        controller->Throw_SPONGE_Error(
            spongeErrorNotImplemented,
            "LISTED_FORCES::Initial",
            "Reason:\n\tFeature GATE-JIT-LISTED is gated in CPU Phase 1.\n\tIssue: TBD");
#endif
        controller->printf("START INITIALIZING LISTED FORCES:\n");
        Configuration_Reader cfg;
        cfg.Open(controller->Command(this->module_name, "in_file"));
        cfg.Close();
        if (!cfg.error_reason.empty())
        {
            cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "LISTED_FORCES::Initial", cfg.error_reason.c_str());
        }
        for (std::string section : cfg.sections)
        {
            forces.push_back(Read_One_Force(controller, section, &cfg));
        }
        for (auto s : cfg.value_unused)
        {
            std::string error_reason = string_format("Reason:\n\t[[ %s% ]] should not be one of the keys of the listed force [[[ %a% ]]]",
                { {"s", s.second}, {"a", s.first} });
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "LISTED_FORCES::Initial", error_reason.c_str());
        }
    }
    if (forces.size() != 0)
    {
        is_initialized = 1;
        for (auto force : forces)
        {
            force->Initial(controller, connectivity, con_dis);
        }
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        for (auto force : forces)
        {
            controller->Step_Print_Initial(force->module_name, "%.2f");
        }
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING LISTED FORCES\n\n");
    }
    else
    {
        controller->printf("LISTED FORCES IS NOT INITIALIZED\n\n");
    }
}

void LISTED_FORCES::Compute_Force(VECTOR* crd, VECTOR box_length, VECTOR* frc, float* atom_energy, float* atom_virial)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            force->Compute_Force(crd, box_length, frc, atom_energy, atom_virial);
        }
    }
}

void LISTED_FORCES::Step_Print(CONTROLLER* controller, VECTOR* crd, VECTOR box_length)
{
    if (is_initialized)
    {
        for (auto force : forces)
        {
            controller->Step_Print(force->module_name, force->Get_Energy(crd, box_length), true);
        }
    }
}

void LISTED_FORCE::Initialize_Parameters(CONTROLLER* controller, std::string parameter_string)
{
    std::vector<std::string> parameters_with_type = string_split(string_strip(parameter_string), ",");
    for (std::string parameter_with_type : parameters_with_type)
    {
        std::vector<std::string> parameter_and_type = string_split(string_strip(parameter_with_type), " ");
        if (parameter_and_type[0] != "int" && parameter_and_type[0] != "float")
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                "LISTED_FORCE::Initialize_Parameters",
                "Reason:\n\tOnly 'int' or 'float' parameter is acceptable\n");
        }
        this->parameter_type.push_back(parameter_and_type[0]);
        this->parameter_name.push_back(parameter_and_type[1]);
    }
    for (int i = 0; i < parameter_type.size(); i++)
    {
        size_t pos = parameter_name[i].rfind("atom_", 0);
        if (pos == 0 && parameter_name[i].size() == 6 && parameter_type[i] == "int")
        {
            atom_labels.push_back(std::string(1, parameter_name[i][5]));
        }
    }
    if (atom_labels.size() > 6 || atom_labels.size() < 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
            "LISTED_FORCE::Initialize_Parameters",
            "Reason:\n\tthe supported numer of atoms in the listed force is 1 to 6\n");
    }
    else
    {
        std::string print_hint;
        for (int i = 0; i < atom_labels.size(); i++)
        {
            print_hint += atom_labels[i];
            if (i != atom_labels.size() - 1)
            {
                print_hint += ", ";
            }
        }
        controller->printf("            %d labels of atoms (%s) found in the parameters\n",
            atom_labels.size(), print_hint.c_str());
    }
}

void LISTED_FORCE::Compile(CONTROLLER* controller)
{
    std::set<std::string> needed_bond;
    std::vector<StringVector> needed_bonds(2);
    std::set<std::string> needed_angle;
    std::vector<StringVector> needed_angles(3);
    std::set<std::string> needed_dihedral;
    std::vector<StringVector> needed_dihedrals(4);
    std::string bond_pair;
    std::string angle_pair;
    std::string dihedral_pair;
    std::string temp_pair;
    for (int i = 0; i < atom_labels.size(); i++)
    {
        for (int j = 0; j < atom_labels.size(); j++)
        {
            if (i == j)
                continue;
            bond_pair.clear();
            bond_pair += atom_labels[i];
            bond_pair += atom_labels[j];
            if (source_code.find("r_" + bond_pair) != source_code.npos)
            {
                needed_bond.insert(bond_pair);
            }
            for (int k = 0; k < atom_labels.size(); k++)
            {
                if (bond_pair.find(atom_labels[k]) != source_code.npos)
                    continue;
                angle_pair = bond_pair + atom_labels[k];
                if (source_code.find("theta_" + angle_pair) != source_code.npos)
                {
                    needed_angle.insert(angle_pair);
                    needed_bond.insert(bond_pair);
                    temp_pair.clear();
                    temp_pair += atom_labels[k];
                    temp_pair += atom_labels[j];
                    needed_bond.insert(temp_pair);
                }
                for (int l = 0; l < atom_labels.size(); l++)
                {
                    if (angle_pair.find(atom_labels[l]) != source_code.npos)
                        continue;
                    dihedral_pair = angle_pair + atom_labels[l];
                    if (source_code.find("phi_" + dihedral_pair) != source_code.npos)
                    {
                        needed_dihedral.insert(dihedral_pair);
                        needed_bond.insert(bond_pair);
                        temp_pair.clear();
                        temp_pair += atom_labels[j];
                        temp_pair += atom_labels[k];
                        needed_bond.insert(temp_pair);
                        temp_pair.clear();
                        temp_pair += atom_labels[k];
                        temp_pair += atom_labels[l];
                        needed_bond.insert(temp_pair);
                    }
                }
            }
        }
    }
    for (auto s : needed_bond)
    {
        needed_bonds[0].push_back(std::string(1, s[0]));
        needed_bonds[1].push_back(std::string(1, s[1]));
    }
    for (auto s : needed_angle)
    {
        needed_angles[0].push_back(std::string(1, s[0]));
        needed_angles[1].push_back(std::string(1, s[1]));
        needed_angles[2].push_back(std::string(1, s[2]));
    }
    for (auto s : needed_dihedral)
    {
        needed_dihedrals[0].push_back(std::string(1, s[0]));
        needed_dihedrals[1].push_back(std::string(1, s[1]));
        needed_dihedrals[2].push_back(std::string(1, s[2]));
        needed_dihedrals[3].push_back(std::string(1, s[3]));
    }
    controller->printf("            %d distance(s), %d angle(s), %d dihedral(s) needed for %s\n", needed_bond.size(), needed_angle.size(), needed_dihedral.size(), this->module_name);
    std::string full_source_code = R"JIT(#include "common.h"
extern "C" __global__ __launch_bounds__(1024) void listed_force_energy_and_virial(%PARM_ARGS%,
VECTOR* crd, VECTOR box_length, VECTOR *frc, float *atom_ene, float *atom_virial, int only_energy, int listed_force_item_numbers)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < listed_force_item_numbers)
    {
        %TEMP_DEC%
        %PARM_DEC%
        %CRD_DEC%
        %BOND_DEC%
        %ANGLE_DEC%
        %DIHEDRAL_DEC%
        %SOURCE%
        if (!only_energy)
        {
            %OUTPUT%
        }
        else
        {
            atomicAdd(atom_ene + tid, E.val);
        }
    }
}
)JIT";
    std::string sadv = std::to_string(atom_labels.size() * 3 + 3);
    std::string sadf = string_format("SADfloat<%N%>", { {"N",  sadv} });
    sadv = string_format("SADvector<%N%>", { {"N",  sadv} });
    std::string endl = "\n        ";
    std::string TEMP_DEC = sadf + " E;\n";
    if (!needed_dihedral.empty())
    {
        TEMP_DEC += sadv + " temp1, temp2;\n";
    }
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ", { parameter_type, parameter_name });
    std::string PARM_DEC = string_join("const %0% %1% = %1%_list[tid];", endl, { parameter_type, parameter_name });
    std::string CRD_DEC = sadv + " box_length_with_grads(box_length, 0, 1, 2);" + endl;
    CRD_DEC += string_join(
        string_format("%sadv% r_%0%(crd[atom_%0%], 3 * %INDEX% + 3, 3 * %INDEX% + 4, 3 * %INDEX% + 5);",
            { {"sadv", sadv} }),
        endl, { atom_labels });
    std::string BOND_DEC = string_join(
        string_format(R"JIT(%sadv% dr_%0%%1% = Get_Periodic_Displacement(r_%0%, r_%1%, box_length_with_grads);
        %sadf% r_%0%%1% = sqrtf(dr_%0%%1% * dr_%0%%1%);)JIT",
            { {"sadv", sadv}, { "sadf", sadf }}),
        endl, needed_bonds);
    std::string ANGLE_DEC = string_join(
        string_format(R"JIT(%sadf% %theta% = 1.0f / (%r1% * %r1%) / (%r2% * %r2%);
        %theta% = sqrtf(%theta%) * (%r1% * %r2%);
        if (%theta% > 0.999999f) %theta% = 0.999999f;
        else if (%theta% < -0.999999f) %theta% = -0.999999f;
        %theta% = acosf(%theta%);)JIT",
            { { "sadf", sadf }, {"theta", "theta_%0%%1%%2%"}, {"r1", "dr_%0%%1%"}, {"r2", "dr_%2%%1%"}}),
        endl, needed_angles);
    std::string DIHEDRAL_DEC = string_join(
        string_format(R"JIT(temp1 = %r1% ^ %r2%;
        temp2 = %r2% ^ %r3%;
        %sadf% %phi% = temp1 * temp2 / sqrtf((temp1 * temp1) * (temp2 * temp2));
        if (%phi% > 0.999999f) %phi% = 0.999999f;
        else if (%phi% < -0.999999f) %phi% = -0.999999f;
        %phi% = acosf(%phi%);
        if (temp1 * %r3% < 0.0f) %phi% = -%phi%;)JIT",
            { {"sadf", sadf }, {"r1", "dr_%0%%1%"}, {"r2", "dr_%1%%2%"}, 
              {"r3", "dr_%2%%3%"}, {"phi", "phi_%0%%1%%2%%3%"}}),
        endl, needed_dihedrals);
    std::string OUTPUT = string_format("atomicAdd(atom_ene + atom_%0%, E.val);\n\
            atomicAdd(atom_virial + atom_%0%, -E.dval[0] * box_length.x - E.dval[1] * box_length.y - E.dval[2] * box_length.z);\n",
        { {"0", atom_labels[0]} });
    OUTPUT += string_join(R"JIT(
            atomicAdd((float*)(frc + atom_%0%), -E.dval[3 * %INDEX% + 3]);
            atomicAdd((float*)(frc + atom_%0%) + 1, -E.dval[3 * %INDEX% + 4]);
            atomicAdd((float*)(frc + atom_%0%) + 2, -E.dval[3 * %INDEX% + 5]);)JIT", endl, { atom_labels });
    full_source_code = string_format(full_source_code, 
        {
            {"PARM_ARGS", PARM_ARGS},
            {"TEMP_DEC", TEMP_DEC},
            {"PARM_DEC", PARM_DEC},
            {"CRD_DEC", CRD_DEC},
            {"BOND_DEC", BOND_DEC},
            {"ANGLE_DEC", ANGLE_DEC},
            {"DIHEDRAL_DEC", DIHEDRAL_DEC},
            {"SOURCE", source_code},
            {"OUTPUT", OUTPUT}
        });
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        force_function.error_reason += "\nSource:\n" + full_source_code;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "LISTED_FORCE::Compile", force_function.error_reason.c_str());
    }
}

void LISTED_FORCE::Initial(CONTROLLER* controller, CONECT* connectivity, PAIR_DISTANCE* con_dis)
{
    FILE* fp;
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        std::string error_reason = std::string("Reason:\n\tlisted force '") + this->module_name + "' is defined, but " + this->module_name + "_in_file is not provided\n";
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "LISTED_FORCE::Initial", error_reason.c_str());
    }
    controller->printf("    Initializing %s\n", this->module_name);
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"), "r");
    if (fscanf(fp, "%d", &item_numbers) != 1)
    {
        std::string error_reason = std::string("Reason:\n\tFail to read the number of items of the listed force '") + this->module_name + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "LISTED_FORCE::Initial", error_reason.c_str());
    }
    Malloc_Safely((void**)&cpu_parameters, sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters, sizeof(void*) * parameter_name.size());
    launch_args = std::vector<void*>(parameter_name.size() + 7);
    Cuda_Malloc_Safely((void**)&item_energy, sizeof(float) * item_numbers);
    Cuda_Malloc_Safely((void**)&sum_energy, sizeof(float));
    for (int j = 0; j < parameter_name.size(); j++)
    {
        if (parameter_type[j] == "int")
        {
            Malloc_Safely((void**)cpu_parameters + j, sizeof(int) * item_numbers);
        }
        else
        {
            Malloc_Safely((void**)cpu_parameters + j, sizeof(float) * item_numbers);
        }
        launch_args[j] = gpu_parameters + j;
    }
    for (int i = 0; i < item_numbers; i++)
    {
        for (int j = 0; j < parameter_name.size(); j++)
        {
            int scanf_ret = 0;
            if (parameter_type[j] == "int")
            {
                scanf_ret = fscanf(fp, "%d", ((int*)cpu_parameters[j]) + i);
            }
            else
            {
                scanf_ret = fscanf(fp, "%f", ((float*)cpu_parameters[j]) + i);
            }
            if (scanf_ret == 0)
            {
                std::string error_reason = std::string("Reason:\n\tFail to read the parameters of the listed force '") + this->module_name + "'\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "LISTED_FORCE::Initial", error_reason.c_str());
            }
        }
    }
    fclose(fp);
    int connected_a = -1, connected_b = -1, constran_id = -1;
    for (int j = 0; j < parameter_name.size(); j++)
    {
        if (parameter_type[j] == "int")
        {
            Cuda_Malloc_And_Copy_Safely((void**)gpu_parameters + j, cpu_parameters[j], sizeof(int) * item_numbers);
        }
        else
        {
            Cuda_Malloc_And_Copy_Safely((void**)gpu_parameters + j, cpu_parameters[j], sizeof(float) * item_numbers);
        }
        if (connected_atoms.size() > 0)
        {
            std::string atom_("atom_");
            if (atom_ + connected_atoms[0] == parameter_name[j])
            {
                connected_a = j;
            }
            else if (atom_ + connected_atoms[1] == parameter_name[j])
            {
                connected_b = j;
            }
        }
        if (this->constrain_distance.size() > 0)
        {
            if (this->constrain_distance == parameter_name[j])
            {
                constran_id = j;
            }
        }
    }
    if (connected_atoms.size() > 0)
    {
        if (connected_a >= 0 && connected_b >= 0)
        {
            for (int i = 0; i < item_numbers; i++)
            {
                int atom_a = ((int*)cpu_parameters[connected_a])[i];
                int atom_b = ((int*)cpu_parameters[connected_b])[i];
                connectivity[0][atom_a].insert(atom_b);
                connectivity[0][atom_b].insert(atom_a);
                if (constran_id >= 0)
                {
                    if (atom_a < atom_b)
                    {
                        con_dis[0][std::pair<int, int>(atom_a, atom_b)] = ((float*)cpu_parameters[constran_id])[i];
                    }
                    else
                    {
                        con_dis[0][std::pair<int, int>(atom_b, atom_a)] = ((float*)cpu_parameters[constran_id])[i];
                    }
                }
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "LISTED_FORCE::Initial", "Reason:\n\tthe name of the connected atoms is not right\n");
        }
    }
    if (this->constrain_distance.size() > 0 && constran_id < 0)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "LISTED_FORCE::Initial", "Reason:\n\tthe name of the constrain distance is not right\n");
    }
}

void LISTED_FORCE::Compute_Force(VECTOR* crd, VECTOR box_length, VECTOR* frc, float* atom_energy, float* atom_virial)
{
    int FALSE = 0;
    launch_args[parameter_name.size()] = &crd;
    launch_args[parameter_name.size() + 1] = &box_length;
    launch_args[parameter_name.size() + 2] = &frc;
    launch_args[parameter_name.size() + 3] = &atom_energy;
    launch_args[parameter_name.size() + 4] = &atom_virial;
    launch_args[parameter_name.size() + 5] = &FALSE;
    launch_args[parameter_name.size() + 6] = &item_numbers;
    CUresult res = force_function({ (item_numbers + 1023u) / 1024u, 1u, 1u }, { 1024u, 1u, 1u }, NULL, 0, launch_args);
}

float LISTED_FORCE::Get_Energy(VECTOR* crd, VECTOR box_length)
{
    cudaMemset(this->item_energy, 0, sizeof(float) * item_numbers);
    int TRUE = 1;
    float* NULLPTR = NULL;
    launch_args[parameter_name.size()] = &crd;
    launch_args[parameter_name.size() + 1] = &box_length;
    launch_args[parameter_name.size() + 2] = &NULLPTR;
    launch_args[parameter_name.size() + 3] = &item_energy;
    launch_args[parameter_name.size() + 4] = &NULLPTR;
    launch_args[parameter_name.size() + 5] = &TRUE;
    launch_args[parameter_name.size() + 6] = &item_numbers;

    CUresult res = force_function({ (item_numbers + 1023u) / 1024u, 1u, 1u }, { 1024u, 1u, 1u }, NULL, 0, launch_args);
    Sum_Of_List(item_energy, sum_energy, item_numbers);
    float h_energy = NAN;
    if (res == CUDA_SUCCESS)
    {
        cudaMemcpy(&h_energy, sum_energy, sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_energy;
}
