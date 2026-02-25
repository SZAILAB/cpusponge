#include "andersen_barostat.h"



void ANDERSEN_BAROSTAT_INFORMATION::Initial(CONTROLLER *controller, float target_pressure, VECTOR box_length, const char *module_name)
{
    controller->printf("START INITIALIZING ANDERSEN BAROSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "andersen_barostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("    The target pressure is %.2f bar\n", target_pressure*CONSTANT_PRES_CONVERTION);

    V0 = box_length.x * box_length.y * box_length.z;
    new_V = V0;

    float taup = 1.0f;
    if (controller[0].Command_Exist(this->module_name, "tau"))
    {
        controller->Check_Float(this->module_name, "tau", "ANDERSEN_BAROSTAT_INFORMATION::Initial");
        taup = atof(controller[0].Command(this->module_name, "tau"));
    }
    controller->printf("    The time constant tau is %f ps\n", taup);
    

    float compressibility = 4.5e-5f;
    if (controller[0].Command_Exist(this->module_name, "compressibility"))
    {
        controller->Check_Float(this->module_name, "compressibility", "ANDERSEN_BAROSTAT_INFORMATION::Initial");
        compressibility = atof(controller[0].Command(this->module_name, "compressibility"));
    }
    controller->printf("    The compressibility constant is %f bar^-1\n", compressibility);
    
    update_interval = 10;
    if (controller->Command_Exist(this->module_name, "update_interval"))
    {
        controller->Check_Int(this->module_name, "update_interval", "ANDERSEN_BAROSTAT_INFORMATION::Initial");
        update_interval = atoi(controller->Command(this->module_name, "update_interval"));
    }
    controller->printf("    The update interval is %d\n", update_interval);

    h_mass_inverse = taup * taup / V0 / compressibility;
    controller->printf("    The piston mass is %f bar ps^2/A^3\n", h_mass_inverse);

    taup *= CONSTANT_TIME_CONVERTION;
    compressibility *= CONSTANT_PRES_CONVERTION;
    h_mass_inverse = V0 * compressibility / taup / taup;


    dV_dt = 0;
    if (controller[0].Command_Exist(this->module_name, "dV_dt"))
    {
        controller->Check_Float(this->module_name, "dV_dt", "ANDERSEN_BAROSTAT_INFORMATION::Initial");
        dV_dt = atof(controller[0].Command(this->module_name, "dV_dt"));
    }
    controller->printf("    The initial dV_dt is %f A^3/(1/20.455 fs)\n", dV_dt);

    int random_seed = rand();
    if (controller[0].Command_Exist(this->module_name, "seed"))
    {
        controller->Check_Int(this->module_name, "seed", "ANDERSEN_BAROSTAT_INFORMATION::Initial");
        random_seed = atoi(controller[0].Command(this->module_name, "seed"));
    }
    controller->printf("    The random seed is %d\n", random_seed);
    generator = std::default_random_engine(random_seed);
    distribution = std::normal_distribution<double>(0.0, 1.0);

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("density", "%.4f");
        controller->Step_Print_Initial("pressure", "%.2f");
        controller->Step_Print_Initial("dV_dt", "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }

    controller->printf("END INITIALIZING ANDERSEN BAROSTAT\n\n");
}

void ANDERSEN_BAROSTAT_INFORMATION::Control_Velocity_Of_Box(float dt, float target_temperature)
{
    if (!is_initialized)
        return;
    float gamma_ln = 1.0f / CONSTANT_TIME_CONVERTION;
    gamma_ln = exp(-gamma_ln * dt);
    dV_dt = gamma_ln * dV_dt +
        sqrt((1 - gamma_ln * gamma_ln) * target_temperature * CONSTANT_kB * h_mass_inverse) * distribution(generator);
}

void ANDERSEN_BAROSTAT_INFORMATION::Ask_For_Calculate_Pressure(int steps, int* need_pressure)
{
    if (is_initialized && steps % update_interval == 0)
    {
        *need_pressure += 1;
    }
}
