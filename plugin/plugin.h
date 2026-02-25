#ifndef PLUGINS_CUH
#define PLUGINS_CUH

#include "../common.h"
#include "../control.h"
#include "../MD_core/MD_core.h"
#include "../collective_variable/collective_variable.h"
#include "../neighbor_list/neighbor_list.h"

#ifdef _WIN32
#include <Windows.h>
#define dlopen(filename, mode) LoadLibrary(filename)
#define dlsym(handle, function) GetProcAddress(handle, function);
#define dlerror() GetLastError()
#define PLUGIN_API extern "C" __declspec(dllexport)
#elif __linux__
#include <dlfcn.h>
#define PLUGIN_API extern "C"
typedef void* HMODULE;
#endif

typedef std::vector<std::vector<std::string>> CVRegisterString;
typedef CVRegisterString (*CVRegisterFunction)();
typedef void (*cv_init_func) (COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*);
typedef void (*cv_compute_func) (int, UNSIGNED_INT_VECTOR*, VECTOR, VECTOR*, VECTOR, int, int);
typedef std::string (*NameFunction)();
typedef std::string (*VersionCheckFunction)(int);
typedef void (*InitialFunction)(MD_INFORMATION* md_info, CONTROLLER* controller, NEIGHBOR_LIST* neighbor_list,
    COLLECTIVE_VARIABLE_CONTROLLER* cv_controller, CV_MAP_TYPE*, CV_INSTANCE_TYPE*);
typedef void (*RuntimeFunction)();

struct SPONGE_PLUGIN
{
    static std::map<std::string, std::function<void(COLLECTIVE_VARIABLE_CONTROLLER*, int, const char*)>> cv_init_functions;
    static std::map<std::string, std::function<void(int, UNSIGNED_INT_VECTOR*, VECTOR, VECTOR*, VECTOR, int, int)>> cv_compute_functions;
    int plugin_numbers = 0;
    HMODULE* plugin_handles;
    int after_init_func_numbers = 0;
    RuntimeFunction* after_init_funcs;
    int force_func_numbers = 0;
    RuntimeFunction* force_funcs;
    int print_func_numbers = 0;
    RuntimeFunction* print_funcs;
    void Initial(MD_INFORMATION* md_info, CONTROLLER* controller,
        COLLECTIVE_VARIABLE_CONTROLLER* cv_controller, NEIGHBOR_LIST* neighbor_list);
    void After_Initial();
    void Calculate_Force();
    void Mdout_Print();
};

#endif

