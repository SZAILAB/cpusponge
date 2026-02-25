#include "Berendsen_barostat.h"



void BERENDSEN_BAROSTAT_INFORMATION::Initial(CONTROLLER *controller, float target_pressure, VECTOR box_length, const char *module_name)
{
    controller->printf("START INITIALIZING BERENDSEN BAROSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "berendsen_barostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("    The target pressure is %.2f bar\n", target_pressure*CONSTANT_PRES_CONVERTION);

    VECTOR boxlength = box_length;
    V0 = boxlength.x * boxlength.y * boxlength.z;
    newV = V0;

    dt = 1e-3f;
    if (controller[0].Command_Exist("dt"))
        dt = atof(controller[0].Command("dt"));
    controller->printf("    The dt is %f ps\n", dt);

    taup = 1.0f;
    if (controller[0].Command_Exist(this->module_name, "tau"))
    {
        controller->Check_Float(this->module_name, "tau", "BERENDSEN_BAROSTAT_INFORMATION::Initial");
        taup = atof(controller[0].Command(this->module_name, "tau"));
    }
    controller->printf("    The time constant tau is %f ps\n", taup);

    compressibility = 4.5e-5f;
    if (controller[0].Command_Exist(this->module_name, "compressibility"))
    {
        controller->Check_Float(this->module_name, "compressibility", "BERENDSEN_BAROSTAT_INFORMATION::Initial");
        compressibility = atof(controller[0].Command(this->module_name, "compressibility"));
    }
    controller->printf("    The compressibility constant is %f bar^-1\n", compressibility);
    compressibility *= CONSTANT_PRES_CONVERTION;

    stochastic_term = 0;
    if (controller[0].Command_Exist(this->module_name, "stochastic_term"))
    {
        stochastic_term = controller->Get_Bool(this->module_name, "stochastic_term", "BERENDSEN_BAROSTAT_INFORMATION::Initial");
    }
    controller->printf("    The stochastic term is %d\n", stochastic_term);

    update_interval = 10;
    if (stochastic_term == 1)
        update_interval = 1;
    if (controller[0].Command_Exist(this->module_name, "update_interval"))
    {
        controller->Check_Int(this->module_name, "update_interval", "BERENDSEN_BAROSTAT_INFORMATION::Initial");
        update_interval = atoi(controller[0].Command(this->module_name, "update_interval"));
    }
    controller->printf("    The update_interval is %d\n", update_interval);



    if (stochastic_term)
    {
        if (update_interval != 1)
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "BERENDSEN_BAROSTAT_INFORMATION::Initial",
                "Reason:\n\tthe update interval should be 1 when using stochastic term\n");
        }
        int seed = time(NULL);
        if (controller[0].Command_Exist(this->module_name, "seed"))
        {
            controller->Check_Int(this->module_name, "seed", "BERENDSEN_BAROSTAT_INFORMATION::Initial");
            seed = atoi(controller[0].Command(this->module_name, "seed"));
        }
        controller->printf("    The random seed is %d\n", seed);
        e.seed(seed);
        std::normal_distribution<float> temp(0, update_interval * dt * CONSTANT_TIME_CONVERTION);
        n = temp;
    }

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("density", "%.4f");
        controller->Step_Print_Initial("pressure", "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }

    controller->printf("END INITIALIZING BERENDSEN BAROSTAT\n\n");
}

void BERENDSEN_BAROSTAT_INFORMATION::Ask_For_Calculate_Pressure(int steps, int *need_pressure)
{
    if (is_initialized && steps % update_interval == 0)
    {
        *need_pressure += 1;
    }
}
