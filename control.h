/*
* Copyright 2021-2023 Gao's lab, Peking University, CCME. All rights reserved.
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


#ifndef CONTROL_CUH
#define CONTROL_CUH

#include <chrono>
#include "common.h"
#include "stdarg.h"

typedef std::map<std::string, std::string> StringMap;
typedef std::map<std::string, int> CheckMap;
typedef std::vector<std::string> StringVector;

//用于记录时间
struct TIME_RECORDER
{
private:
    std::chrono::steady_clock::time_point start_timestamp;
    std::chrono::steady_clock::time_point end_timestamp;
public:
    double time = 0;
    void Start();
    void Stop();
    void Clear();
};

struct CONTROLLER
{
    //自身信息
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20231219;

    void Initial(int argc, char **argv, const char* subpackage_hint);
    void Clear();

    //文件输出缓冲大小
    int buffer_frame;
    //全局主机随机种子（影响所有使用rand()初始化的模块）
    unsigned int random_seed = 0;
    //设置文件缓冲区大小
    void Set_File_Buffer(FILE* file, size_t one_frame_size);

    //输入控制部分
    StringMap original_commands;  //由文件读入的原始命令
    StringMap commands;   //由文件读入的命令（去除空格）
    CheckMap command_check;   //检查输入是否都使用了
    CheckMap choice_check;   //检查选项是否都使用了
    void Get_Command(char *line, char *prefix); //内部解析argument时专用，设置命令，不外部调用
    void Set_Command(const char *Flag, const char *Value, int Check = 1, const char *prefix = NULL); //内部解析argument时专用，设置命令，不外部调用
    void Arguments_Parse(int argc, char **argv);  //对终端输入进行分析
    void Commands_From_In_File(int argc, char **argv, const char* subpackage_hint); //对mdin输入进行分析并打印日志信息
    void Default_Set();  //对最基本的功能进行默认设置
    int working_device = 0; //使用的设备
    void Init_Cuda();   //对cuda初始化
    //本部分的上面的内容最好不要外部调用

    void Input_Check(); //检查所有输入是否都被使用了（防止错误的输入未被检查到）

    bool Command_Exist(const char *key);   //判断文件读入的命令中是否有key
    bool Command_Exist(const char *prefix, const char *key);   //判断文件读入的命令中是否有key
    //判断是否存在key且值为value。未设置时返回no_set_return_value，可控制比对是否对大小写敏感
    bool Command_Choice(const char *key, const char *value, bool case_sensitive = 0);
    //判断是否存在prefix_key且值为value。未设置时返回no_set_return_value，可控制比对是否对大小写敏感
    bool Command_Choice(const char *prefix, const char *key, const char *value, bool case_sensitive = 0);
    const char * Command(const char *key);   //获得文件读入的命令key对应的value
    const char * Command(const char *prefix, const char *key);   //获得文件读入的命令key对应的value
    const char * Original_Command(const char *key);   //获得文件读入的命令key对应的value
    const char * Original_Command(const char *prefix, const char *key);   //获得文件读入的命令key对应的value

    //计时部分
    TIME_RECORDER core_time; //计时器
    float simulation_speed; //模拟运行速度（纳秒/天）

    //输出控制部分
    FILE *mdinfo = NULL;  //屏幕信息打印文件
    FILE *mdout = NULL;
    StringMap outputs_content;  //记录每步输出数值
    StringMap outputs_format; //记录每步输出的格式
    StringVector outputs_key; //记录每部输出的表头
    //本部分的上面的内容最好不要外部调用

    float printf_sum = 0;
    void printf(const char *fmt, ...);  //重载printf，使得printf能够同时打印到mdinfo和屏幕
    void Step_Print_Initial(const char *head, const char *format); //其他模块初始化时调用，获得对应的表头和格式
    void Step_Print(const char* head, const float* pointer, const bool add_to_total = false); //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char* head, const float pointer, const bool add_to_total = false);  //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char* head, const double pointer, const bool add_to_total = false);  //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char *head, const int pointer);    //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char* head, const char *pointer);    //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Print_First_Line_To_Mdout(FILE *mdout = NULL);             //模拟开始前的操作，将表头打印到mdout，并在屏幕打印一个分割线
    void Print_To_Screen_And_Mdout(FILE *mdout = NULL);             //模拟开始每步的调用，使得其他部分的结果打印到屏幕和mdout

    //错误处理
    void Check_Error(float potential);
    void Check_Int(const char* command, const char* error_by);
    void Check_Float(const char* command, const char* error_by);
    bool Get_Bool(const char* command, const char* error_by);
    void Check_Int(const char* prefix, const char* command, const char* error_by);
    void Check_Float(const char* prefix, const char* command, const char* error_by);
    bool Get_Bool(const char* prefix, const char* command, const char* error_by);
    void Throw_SPONGE_Error(const int error_number, const char* error_by = NULL, const char* extra_error_string = NULL);

    //警告
    StringVector warnings;
    bool warn_of_initialization;
    void Warn(const char* warning);
};
//判断两个字符串是否相等（无视大小写）
bool is_str_equal(const char* a_str, const char *b_str, int case_sensitive = 0);
//判断字符串是否是int
bool is_str_int(const char* str);
//判断字符串是否是float
bool is_str_float(const char* str);
//判断字符串是否是bool
bool is_str_bool(const char* str);
//字符串去掉前后空格
std::string string_strip(std::string string);
//字符串分割
std::vector<std::string> string_split(std::string string, std::string separators);
//字符串替换
std::string string_replace(std::string string, std::string old, std::string new_);
//字符串格式化输出
std::string string_format(std::string, std::map<std::string, std::string>);
//字符串格式化输出
std::string string_join(std::string pattern, std::string separator, std::vector<std::vector<std::string>> string_vectors);

//用于读取复杂的配置文件
struct Configuration_Reader
{
    std::string error_reason;
    FILE* f;
    std::vector<std::string> sections;
    std::map<std::string, std::vector<std::string>> keys;
    std::map<std::pair<std::string, std::string>, std::string> values;
    std::set<std::pair<std::string, std::string>> value_unused;
    void Open(std::string filename);
    bool Section_Exist(std::string section);
    bool Key_Exist(std::string section, std::string key);
    std::string Get_Value(std::string section, std::string key);
    void Close();
};

//用于即时编译函数
struct JIT_Function
{
    std::string error_reason;
    CUfunction function;
    void Compile(std::string source);
    CUresult operator() (dim3 blocks, dim3 threads, cudaStream_t stream, unsigned int shared_memory_size, std::vector<void*> args);
};

//用于安全的显存和内存分配，以及打开文件
bool Malloc_Safely(void** address, size_t size);
bool Cuda_Malloc_Safely(void** address, size_t size);
bool Open_File_Safely(FILE** file, const char* file_name, const char* open_type);
bool Cuda_Malloc_And_Copy_Safely(void** d_address, void* h_address, size_t size, const char* var_name = NULL);

//SPONGE错误类型
enum spongeError
{
    spongeSuccess = 0,
    // 1000以下的错误留给cudaError
    // 未实现的功能
    spongeErrorNotImplemented = 1001,
    // 文件格式（编码、换行符）问题 或 数据格式不正确
    spongeErrorBadFileFormat,
    // 冲突的命令
    spongeErrorConflictingCommand,
    // 缺失的命令
    spongeErrorMissingCommand,
    // 类型错误的命令
    spongeErrorTypeErrorCommand,
    // 值错误的命令
    spongeErrorValueErrorCommand,
    // 模拟崩溃
    spongeErrorSimulationBreakDown,
    // 内存分配失败
    spongeErrorMallocFailed,
    // 越界
    spongeErrorOverflow,
    // 打开文件失败
    spongeErrorOpenFileFailed,
};


#endif //CONTROL_CUH(control.h)
