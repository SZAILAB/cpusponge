#include "plugin.h"

std::map<std::string, std::function<void(COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*)>> SPONGE_PLUGIN::cv_init_functions;
std::map<std::string, std::function<void(int, UNSIGNED_INT_VECTOR*, VECTOR, VECTOR*, VECTOR, int, int)>> SPONGE_PLUGIN::cv_compute_functions;

void SPONGE_PLUGIN::Initial(MD_INFORMATION* md_info, CONTROLLER* controller,
    COLLECTIVE_VARIABLE_CONTROLLER* cv_controller, NEIGHBOR_LIST* neighbor_list)
{
    if (controller->Command_Exist("plugin"))
    {
#ifdef SPONGE_CPU_PHASE1
        const bool phase2_mainpath_mode =
            controller->Command_Exist("phase2_plugin_runtime_mainpath_mode") &&
            controller->Get_Bool("phase2_plugin_runtime_mainpath_mode", "SPONGE_PLUGIN::Initial");
        if (!phase2_mainpath_mode)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorNotImplemented,
                "SPONGE_PLUGIN::Initial",
                "Reason:\n\tFeature GATE-PLUGIN-RUNTIME is gated in CPU Phase 1.\n"
                "\tSet phase2_plugin_runtime_mainpath_mode = 1 for scoped main-path runtime coverage.\n"
                "\tIssue: PHASE2-REENABLE-PLUGIN-RUNTIME");
        }
        controller->printf("PHASE2-PLUGIN-RUNTIME-MAINPATH: scoped runtime mode enabled\n");
#endif
        controller->printf("START INITIALIZING SPONGE PLUGIN:\n");
        plugin_numbers = 0;
        //strtok is not thread safe, and may raise some strange error, so I use c++ string instead of c char* . 
        std::string command(controller->Original_Command("plugin"));
        auto lastPos = command.find_first_not_of(" ", 0);
        auto pos = command.find_first_of(" ", lastPos);
        while (pos != std::string::npos || lastPos != std::string::npos)
        {
            plugin_numbers += 1;
            lastPos = command.find_first_not_of(" ", pos);
            pos = command.find_first_of(" ", lastPos);
        }
        controller->printf("%d plugin(s) to load\n", plugin_numbers);
        Malloc_Safely((void**)&plugin_handles, sizeof(HMODULE) * plugin_numbers);
        Malloc_Safely((void**)&after_init_funcs, sizeof(RuntimeFunction) * plugin_numbers);
        Malloc_Safely((void**)&force_funcs, sizeof(RuntimeFunction) * plugin_numbers);
        Malloc_Safely((void**)&print_funcs, sizeof(RuntimeFunction) * plugin_numbers);
        int count = 0;
        std::string plugin_name, plugin_version, version_check_error;
        char plugin_path[CHAR_LENGTH_MAX];
        NameFunction name_func, version_func;
        VersionCheckFunction version_check_func;
        lastPos = command.find_first_not_of(" ", 0);
        pos = command.find_first_of(" ", lastPos);
        while (pos != std::string::npos || lastPos != std::string::npos)
        {
            int funcs_loaded = 1;
            sscanf(command.substr(lastPos, pos - lastPos).c_str(), "%s", plugin_path);
            plugin_handles[count] = dlopen(plugin_path, RTLD_LAZY | RTLD_GLOBAL);
            if (plugin_handles[count] == NULL)
            {
                std::string error_reason = "Reason:\n\tOpen Dynamic Library from ";
                error_reason += plugin_path;
                error_reason += " failed\n";
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            name_func = (NameFunction)dlsym(plugin_handles[count], "Name");
            if (name_func == NULL)
            {
                std::string error_reason = "Reason:\n\tFind the name of the plugin from ";
                error_reason += plugin_path;
                error_reason += " failed\n";
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            plugin_name = name_func();
            version_func = (NameFunction)dlsym(plugin_handles[count], "Version");
            if (version_func == NULL)
            {
                std::string error_reason = "Reason:\n\tFind the version of the plugin from ";
                error_reason += plugin_path;
                error_reason += " (" + plugin_name + ") failed\n";
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            plugin_version = version_func();
            version_check_func = (VersionCheckFunction)dlsym(plugin_handles[count], "Version_Check");
            if (version_check_func == NULL)
            {
                std::string error_reason = "Reason:\n\tFind the version check function of the plugin from ";
                error_reason += plugin_path;
                error_reason += " (" + plugin_name + " version: " + plugin_version + ") failed\n";
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            version_check_error = version_check_func(controller->last_modify_date);
            if (!version_check_error.empty())
            {
                std::string error_reason = "Reason:\n\tThe version check of the plugin from ";
                error_reason += plugin_path;
                error_reason += " (" + plugin_name + " version: " + plugin_version + ") failed\n" + version_check_error;
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            controller->printf("Plugin %d:\n    name: %s\n    version: %s\n    path: %s\n    functions loaded: ", plugin_numbers, plugin_name.c_str(), plugin_version.c_str(), plugin_path);
            InitialFunction func = (InitialFunction)dlsym(plugin_handles[count], "Initial");
            if (func == NULL)
            {
                std::string error_reason = "Reason:\n\tFind the initial function of the plugin from ";
                error_reason += plugin_path;
                error_reason += " (" + plugin_name + " version: " + plugin_version + ") failed\n";
                error_reason += dlerror();
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "SPONGE_PLUGIN::Initial", error_reason.c_str());
            }
            controller->printf(" Initial");
            after_init_funcs[after_init_func_numbers] = (RuntimeFunction)dlsym(plugin_handles[count], "After_Initial");
            if (after_init_funcs[after_init_func_numbers] != NULL)
            {
                funcs_loaded += 1;
                after_init_func_numbers += 1;
                controller->printf(" After_Initial");
            }
            force_funcs[force_func_numbers] = (RuntimeFunction)dlsym(plugin_handles[count], "Calculate_Force");
            if (force_funcs[force_func_numbers] != NULL)
            {
                funcs_loaded += 1;
                force_func_numbers += 1;
                controller->printf(" Calculate_Force");
            }
            print_funcs[print_func_numbers] = (RuntimeFunction)dlsym(plugin_handles[count], "Mdout_Print");
            if (print_funcs[print_func_numbers] != NULL)
            {
                funcs_loaded += 1;
                print_func_numbers += 1;
                controller->printf(" Mdout_Print");
            }
            controller->printf(" (%d in total)\n", funcs_loaded);
            func(md_info, controller, neighbor_list, cv_controller, CV_MAP, CV_INSTANCE_MAP);
            count += 1;
            lastPos = command.find_first_not_of(" ", pos);
            pos = command.find_first_of(" ", lastPos);
        }
        controller->printf("END INITIALIZING SPONGE PLUGIN\n\n");
    }
}

void SPONGE_PLUGIN::After_Initial()
{
    for (int i = 0; i < after_init_func_numbers; i++)
    {
        after_init_funcs[i]();
    }
}

void SPONGE_PLUGIN::Calculate_Force()
{
    for (int i = 0; i < force_func_numbers; i++)
    {
        force_funcs[i]();
    }
}

void SPONGE_PLUGIN::Mdout_Print()
{
    for (int i = 0; i < print_func_numbers; i++)
    {
        print_funcs[i]();
    }
}
