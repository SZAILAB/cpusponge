#include "control.h"

#define SPONGE_VERSION "v1.4 2024-01-01"

#define MDIN_DEFAULT_FILENAME "mdin.txt"
#define MDOUT_DEFAULT_FILENAME "mdout.txt"
#define MDINFO_DEFAULT_FILENAME "mdinfo.txt"

#define MDIN_COMMAND "mdin"
#define MDOUT_COMMAND "mdout"
#define MDINFO_COMMAND "mdinfo"


bool is_str_equal(const char* a_str, const char* b_str, int case_sensitive)
{
    int i = 0;
    char a;
    char b;
    while (true)
    {
        if (a_str[i] == 0 && b_str[i] == 0)
        {
            return 1;
        }
        else if (a_str[i] == 0 || b_str[i] == 0)
        {
            return 0;
        }
        else
        {
            a = a_str[i];
            b = b_str[i];
            if (!case_sensitive)
            {
                if (a >= 65 && a <= 90)
                {
                    a = a - 65 + 97;
                }
                if (b >= 65 && b <= 90)
                {
                    b = b - 65 + 97;
                }
            }
            if (a != b)
            {
                return 0;
            }
        }
        i = i + 1;
    }
}

bool is_str_int(const char* str)
{
    bool hasNum = false;
    for (int index = 0; *str != '\0'; str++, index++)
    {
        switch (*str)
        {
        case '0':case'1':case'2':case'3':case'4':case'5':
        case'6':case'7':case'8':case'9':
            hasNum = true;
            break;
        case '-':case '+':
            if (index != 0)
            {
                return false;
            }
            break;
        default:
            return false;
        }
    }
    return hasNum;
}

bool is_str_float(const char* str)
{
    bool isE = false, isPoint = false, numBefore = false, numBehind = false, hasNum = false;
    for (int index = 0; *str != '\0'; str++, index++)
    {
        switch (*str)
        {
        case '0':case'1':case'2':case'3':case'4':case'5':
        case'6':case'7':case'8':case'9':
            hasNum = true;
            if (isE)
            {
                numBehind = true;
            }
            else
            {
                numBefore = true;
            }
            break;
        case '+':case '-':
            if (index != 0)
            {
                return false;
            }
            break;
        case 'e':case 'E':
            if (isE || !numBefore)
            {
                return false;
            }
            else
            {
                isPoint = true;
                index = -1;
                isE = true;
            }
            break;
        case '.':
            if (isPoint)
            {
                return false;
            }
            else
            {
                isPoint = true;
            }
            break;
        default:
            return false;
        }
    }
    if (!numBefore)
    {
        return false;
    }
    else if (isE && !numBehind)
    {
        return false;
    }
    return hasNum;
}

std::string string_strip(std::string string)
{
    string.erase(string.find_last_not_of("\n") + 1);
    string.erase(0, string.find_first_not_of(" "));
    string.erase(string.find_last_not_of(" ") + 1);
    return string;
}

std::vector<std::string> string_split(std::string string, std::string separators)
{
    std::vector<std::string> result;
    if (string.size() == 0)
        return result;
    size_t last_pos = string.find_first_not_of(separators, 0);
    size_t pos = string.find_first_of(separators, last_pos);
    while (pos != string.npos)
    {
        result.push_back(string.substr(last_pos, pos - last_pos));
        last_pos = string.find_first_not_of(separators, pos);
        pos = string.find_first_of(separators, last_pos);
    }
    result.push_back(string.substr(last_pos, pos - last_pos));
    return result;
}

std::string string_replace(std::string string, std::string old, std::string new_)
{
    std::string result = string;
    size_t pos = result.find(old, 0);
    while (pos != result.npos)
    {
        result.replace(pos, old.length(), new_);
        pos += new_.length();
        pos = result.find(old, pos);
    }
    return result;
}

std::string string_format(std::string string, std::map<std::string, std::string> dict)
{
    std::string result = string;
    for (auto& pair : dict)
    {
        result = string_replace(result, "%" + pair.first + "%", pair.second);
    }
    return result;
}

std::string string_join(std::string pattern, std::string separator, std::vector<std::vector<std::string>> string_vectors)
{
    std::string result;
    int n = string_vectors.size();
    int l = string_vectors[0].size();
    for (int i = 0; i < l; i++)
    {
        std::string one_result = pattern;
        for (int j = 0; j < n; j++)
        {
            one_result = string_replace(one_result, "%" + std::to_string(j) + "%", string_vectors[j][i]);
        }
        one_result = string_replace(one_result, "%INDEX%", std::to_string(i));
        result += one_result;
        if (i != l - 1)
            result += separator;
    }
    return result;
}

bool CONTROLLER::Command_Exist(const char* key)
{
    const char* temp = strstr(key, "in_file");
    command_check[key] = 0;
    if (temp != NULL && strcmp(temp, "in_file") == 0)
    {
        if (commands.count(key))
        {
            return true;
        }
        else if (Command_Exist("default_in_file_prefix"))
        {
            std::string buffer, buffer2;
            buffer = key;
            buffer = buffer.substr(0, strlen(key) - strlen(temp) - 1);
            buffer2 = Command("default_in_file_prefix") + ("_" + buffer + ".txt");
            FILE* ftemp = fopen(buffer2.c_str(), "r");
            if (ftemp != NULL)
            {
                commands[key] = buffer2;
                fclose(ftemp);
                return true;
            }
            return false;

        }
        else
        {
            return false;
        }
    }
    else
    {
        return (bool)commands.count(key);
    }
}

bool CONTROLLER::Command_Exist(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Exist(temp);
}

bool CONTROLLER::Command_Choice(const char* key, const char* value, bool case_sensitive)
{
    if (commands.count(key))
    {
        if (is_str_equal(commands[key].c_str(), value, case_sensitive))
        {
            command_check[key] = 0;
            choice_check[key] = 1;
            return true;
        }
        else
        {
            command_check[key] = 0;
            if (choice_check[key] != 1)
                choice_check[key] = 2;
            return false;
        }
    }
    else
    {
        choice_check[key] = 3;
        return false;
    }
}

bool CONTROLLER::Command_Choice(const char* prefix, const char* key, const char* value, bool case_sensitive)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Choice(temp, value, case_sensitive);
}

const char* CONTROLLER::Command(const char* key)
{
    command_check[key] = 0;
    return commands[key].c_str();
}

const char* CONTROLLER::Command(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return commands[temp].c_str();
}

const char* CONTROLLER::Original_Command(const char* key)
{
    command_check[key] = 0;
    return original_commands[key].c_str();
}

const char* CONTROLLER::Original_Command(const char* prefix, const char* key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return original_commands[temp].c_str();
}

static int judge_if_flag(const char* str)
{
    if (strlen(str) <= 1)
        return 0;
    if (str[0] != '-')
        return 0;
    if (str[1] >= '0' && str[1] <= '9')
        return 0;
    return 1;
}
void CONTROLLER::Arguments_Parse(int argc, char** argv)
{
    char temp1[CHAR_LENGTH_MAX];
    char temp2[CHAR_LENGTH_MAX];
    char temp3[CHAR_LENGTH_MAX];
    int j = 1;
    for (int i = 1; i < argc; i++)
    {
        temp1[0] = 0;
        strcat(temp1, argv[i]);
        if (judge_if_flag(temp1))
        {
            temp2[0] = ' ';
            temp2[1] = 0;
            j = 1;
            while (i + j < argc)
            {
                strcpy(temp3, argv[i + j]);
                if (!judge_if_flag(temp3))
                {
                    strcat(temp2, " ");
                    strcat(temp2, temp3);
                    j++;
                }
                else
                    break;
            }
            Set_Command(temp1 + 1, temp2);
        }
    }
}

void CONTROLLER::Get_Command(char* line, char* prefix)
{

    if ((prefix[0] == '#' && prefix[1] == '#') || prefix[0] == ' ' || prefix[0] == '\t')
    {
        return;
    }
    char Flag[CHAR_LENGTH_MAX];
    char Value[CHAR_LENGTH_MAX];
    char* flag = strtok(line, "=");
    char* command = strtok(NULL, "=");

    if (flag == NULL || command == NULL)
    {
        return;
    }

    sscanf(flag, "%s", Flag);
    strcpy(Value, command);
    //fprintf(stdout, "%s|\n%s|\n%s|\n\n", Flag, Value, prefix); //debug用
    Set_Command(Flag, Value, 1, prefix);

}

static int read_one_line(FILE* In_File, char* line, char* ender)
{
    int line_count = 0;
    int ender_count = 0;
    char c;
    while ((c = getc(In_File)) != EOF)
    {
        if (line_count == 0 && (c == '\t' || c == ' '))
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            line[line_count] = c;
            line_count += 1;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
            break;
        }
    }
    while ((c = getc(In_File)) != EOF)
    {
        if (c == ' ' || c == '\t')
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            fseek(In_File, -1, SEEK_CUR);
            break;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
        }
    }
    line[line_count] = 0;
    ender[ender_count] = 0;
    if (line_count == 0 && ender_count == 0)
    {
        return EOF;
    }
    return 1;
}

void CONTROLLER::Commands_From_In_File(int argc, char** argv, const char* subpackage_hint)
{
    FILE* In_File = NULL;
    if (!Command_Exist(MDIN_COMMAND))
    {
        In_File = fopen(MDIN_DEFAULT_FILENAME, "r");
        if (In_File == NULL)
        {
            commands["md_name"] = "Default SPONGE MD Task Name";
        }
    }
    else
        Open_File_Safely(&In_File, Command(MDIN_COMMAND), "r");

    if (In_File != NULL)
    {
        char line[CHAR_LENGTH_MAX];
        char prefix[CHAR_LENGTH_MAX] = { 0 };
        char ender[CHAR_LENGTH_MAX];
        char* get_ret = fgets(line, CHAR_LENGTH_MAX, In_File);
        line[strlen(line) - 1] = 0;
        commands["md_name"] = line;
        int while_count = 0;
        while (true)
        {
            while_count += 1;
            if (while_count > 100000)
            {
                Throw_SPONGE_Error(spongeErrorBadFileFormat, "CONTROLLER::Commands_From_In_File",
                    "Possible reasons : \n\t1.The coding of the format is not ASCII\n\t2.The file is created in one OS but used in another OS(Windows / Unix / MacOS)");
            }
            if (read_one_line(In_File, line, ender) == EOF)
            {
                break;
            }
            if (line[0] == '#')
            {
                if (line[1] == '#')
                {
                    if (strchr(ender, '{') != NULL)
                    {
                        int scanf_ret = sscanf(line, "%s", prefix);
                    }
                    if (strchr(ender, '}') != NULL)
                    {
                        prefix[0] = 0;
                    }
                }
                if (strchr(ender, '\n') == NULL)
                {
                    int scanf_ret = fscanf(In_File, "%*[^\n]%*[\n]");
                    fseek(In_File, -1, SEEK_CUR);
                }
            }
            else if (strchr(ender, '{') != NULL)
            {
                int scanf_ret = sscanf(line, "%s", prefix);
            }
            else
            {
                Get_Command(line, prefix);
                line[0] = 0;
            }
            if (strchr(ender, '}') != NULL)
            {
                prefix[0] = 0;
            }
        }
    }

    if (Command_Exist(MDINFO_COMMAND))
    {
        Open_File_Safely(&mdinfo, Command(MDINFO_COMMAND), "w");
    }
    else
    {
        Open_File_Safely(&mdinfo, MDINFO_DEFAULT_FILENAME, "w");
    }
    setvbuf(mdinfo, NULL, _IONBF, 0);
    if (Command_Exist(MDOUT_COMMAND))
    {
        Open_File_Safely(&mdout, Command(MDOUT_COMMAND), "w");
    }
    else
    {
        Open_File_Safely(&mdout, MDOUT_DEFAULT_FILENAME, "w");
    }
    printf("SPONGE Version:\n    %s\n\n", SPONGE_VERSION);
    printf("Sub-package:\n    %s\n\n", subpackage_hint);
    printf("Citation:\n    %s\n", "Huang, Y. - P., Xia, Y., Yang, L., Wei, J., Yang, Y.I.and Gao, Y.Q. (2022), SPONGE: A GPU - Accelerated Molecular Dynamics Package with Enhanced Sampling and AI - Driven Algorithms.Chin.J.Chem., 40 : 160 - 168. https ://doi.org/10.1002/cjoc.202100456\n\n");
    printf("MD TASK NAME:\n    %s\n\n", commands["md_name"].c_str());
    int scanf_ret = fprintf(mdinfo, "Terminal Commands:\n    ");
    for (int i = 0; i < argc; i++)
    {
        scanf_ret = fprintf(mdinfo, "%s ", argv[i]);
    }
    scanf_ret = fprintf(mdinfo, "\n\n");
    if (In_File != NULL)
    {
        scanf_ret = fprintf(mdinfo, "Mdin File:\n");
        fseek(In_File, 0, SEEK_SET);
        char temp[CHAR_LENGTH_MAX];
        while (fgets(temp, CHAR_LENGTH_MAX, In_File) != NULL)
        {
            scanf_ret = fprintf(mdinfo, "    %s", temp);
        }
        scanf_ret = fprintf(mdinfo, "\n\n");
        fclose(In_File);
    }

}
void CONTROLLER::Set_Command(const char* Flag, const char* Value, int Check, const char* prefix)
{
    if (prefix && strcmp(prefix, "comments") == 0)
        return;
    char temp[CHAR_LENGTH_MAX] = { 0 }, temp2[CHAR_LENGTH_MAX];
    if (prefix && prefix[0] != 0 && strcmp(prefix, "main") != 0)
    {
        strcpy(temp, prefix);
        strcat(temp, "_");
    }
    strcat(temp, Flag);
    if (commands.count(temp))
    {
        sprintf(temp2, "Reason:\n\t'%s' is set more than once\n", temp);
        Throw_SPONGE_Error(spongeErrorConflictingCommand, "CONTROLLER::Set_Command", temp2);
    }
    strcpy(temp2, Value);
    char* real_value = strtok(temp2, "#");
    original_commands[temp] = real_value;
    if (sscanf(real_value, "%s", temp2))
        commands[temp] = temp2;
    else
        commands[temp] = "";

    command_check[temp] = Check;
}

void CONTROLLER::Default_Set()
{
    random_seed = (unsigned int)time(NULL);
    if (Command_Exist("random_seed"))
    {
        Check_Int("random_seed", "CONTROLLER::Default_Set");
        const int parsed_seed = atoi(Command("random_seed"));
        if (parsed_seed < 0)
        {
            Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "CONTROLLER::Default_Set",
                "Reason:\n\trandom_seed must be a non-negative integer\n");
        }
        random_seed = (unsigned int)parsed_seed;
    }
    srand(random_seed);
    printf("    random seed is %u\n", random_seed);

    buffer_frame = 1000;
    if (Command_Exist("buffer_frame"))
    {
        Check_Int("buffer_frame", "CONTROLLER::Default_Set");
        buffer_frame = atoi(Command("buffer_frame"));
    }
}

void CONTROLLER::Init_Cuda()
{
#ifdef SPONGE_CPU_PHASE1
    printf("    Start initializing CUDA (CPU Phase-1 compatibility mode)\n");
    working_device = 0;
    printf("    End initializing CUDA (stubbed)\n");
    return;
#endif
    printf("    Start initializing CUDA\n");
    if (cuInit(0) != CUDA_SUCCESS)
    {
        Throw_SPONGE_Error(spongeErrorMallocFailed, "CONTROLLER::Init_Cuda", "Reason:\n\tFail to initialize cuda");
    }

    int count;
    working_device = atoi(Command("device"));
    cudaGetDeviceCount(&count);
#ifdef CUDA_VERSION
    int cuda_major_version = CUDA_VERSION / 1000;
    int cuda_minor_version = (CUDA_VERSION - 1000 * cuda_major_version) / 10;
    printf("        Compiled by CUDA %d.%d\n", cuda_major_version, cuda_minor_version);
#else
    printf("        Compiled by unknown CUDA version\n");
#endif
    printf("        %d device found:\n", count);

    cudaDeviceProp prop;
    float GlobalMem;
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        GlobalMem = (float)prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f;
        printf("            Device %d:\n                Name: %s\n                Memory: %.1f GB\n                Compute Capability: %d%d\n",
            i, prop.name, GlobalMem, prop.major, prop.minor);
    }
    if (count <= working_device)
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason, "Reason:\n\tthe available device count %d is less than the setting working_device %d.", count, working_device);
        this->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "CONTROLLER::Init_Cuda", error_reason);
    }
    printf("        Set Device %d\n", working_device);
    cudaSetDevice(working_device);

    cudaGetDeviceProperties(&prop, working_device);
    int cuda_arch_bin = prop.major * 10 + prop.minor;
    if (cuda_arch_bin < CUDA_ARCH_BIN)
    {
        char error_reason[CHAR_LENGTH_MAX];

        int ret = sprintf(error_reason, "Reason:\n\tthe compute compacity (%d) of the working GPU device (%s) is less than the minimum required compute compacity (%d) for the compiled SPONGE version\n",
            cuda_arch_bin, prop.name, CUDA_ARCH_BIN);

        Throw_SPONGE_Error(cudaErrorInvalidDevice, "CONTROLLER::Init_Cuda", error_reason);
    }

    printf("    End initializing CUDA\n");
}

void CONTROLLER::Input_Check()
{
    if (!(Command_Exist("dont_check_input") && Get_Bool("dont_check_input", "CONTROLLER::Input_Check")))
    {
        int no_warning = 0;
        for (CheckMap::iterator iter = command_check.begin(); iter != command_check.end(); iter++)
        {
            if (iter->second == 1)
            {
                printf("Warning: '%s' is set, but never used.\n", iter->first.c_str());
                no_warning += 1;
            }
        }
        for (CheckMap::iterator iter = choice_check.begin(); iter != choice_check.end(); iter++)
        {
            if (iter->second == 2)
            {
                char error_reason[CHAR_LENGTH_MAX];
                sprintf(error_reason, "Reason:\n\tthe value '%s' of command '%s' matches none of the choices.\n", this->commands[iter->first].c_str(), iter->first.c_str());
                this->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "CONTROLLER::Input_Check", error_reason);
            }
            else if (iter->second == 3)
            {
                printf("Warning: command '%s' is not set.\n", iter->first.c_str());
                no_warning += 1;
            }
        }
        for (int i = 0; i < warnings.size(); i++)
        {
            printf("Warning: %s\n", warnings[i].c_str());
            no_warning += 1;
        }
        if (no_warning)
        {
            printf("\nWarning: inputs raised %d warning(s). If You know WHAT YOU ARE DOING, press any key to continue. Set dont_check_input = 1 to disable this warning.\n", no_warning);
            getchar();
        }
    }
}

void CONTROLLER::printf(const char* fmt, ...)
{
    va_list argp;

    va_start(argp, fmt);
    vfprintf(stdout, fmt, argp);
    va_end(argp);

    if (mdinfo != NULL)
    {
        va_start(argp, fmt);
        vfprintf(mdinfo, fmt, argp);
        va_end(argp);
    }
}

void CONTROLLER::Step_Print_Initial(const char* head, const char* format)
{
    outputs_format.insert(std::pair<std::string, std::string>(head, format));
    outputs_content.insert(std::pair<std::string, std::string>(head, "****"));
    outputs_key.push_back(head);
}

void CONTROLLER::Step_Print(const char* head, const float* pointer, const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer[0]);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += pointer[0];
        }
    }

}

void CONTROLLER::Step_Print(const char* head, const float pointer, const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += pointer;
        }
    }
}

void CONTROLLER::Step_Print(const char* head, const double pointer, const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += (float)pointer;
        }
    }
}

void CONTROLLER::Step_Print(const char* head, const int pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

void CONTROLLER::Step_Print(const char* head, const char* pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

void CONTROLLER::Print_First_Line_To_Mdout(FILE* mdout)
{
    if (mdout == NULL)
    {
        mdout = this->mdout;
    }
    for (int i = 0; i < outputs_key.size(); i++)
    {
        fprintf(mdout, "%15s ", outputs_key[i].c_str());
    }
    fprintf(mdout, "\n");
    printf("------------------------------------------------------------------------------------------------------------\n");
}

void CONTROLLER::Check_Error(float energy)
{
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != 0)
    {
        Throw_SPONGE_Error(cuda_error, "CONTROLLER::Check_Error",
            "Possible reasons:\n\t1. the energy of the system is not fully minimized\n\t2. bad dt (too large)\n\t3. bad thermostat/barostat parameters\n\t4. bad force field parameters\n");
    }
    if (isnan(energy) || isinf(energy) || isnan(printf_sum) || isinf(printf_sum))
    {
        Throw_SPONGE_Error(spongeErrorSimulationBreakDown, "CONTROLLER::Check_Error",
            "Possible reasons:\n\t1. the energy of the system is not fully minimized\n\t2. bad dt (too large)\n\t3. bad thermostat/barostat parameters\n\t4. bad force field parameters\n");
    }
    printf_sum = 0;
}

void CONTROLLER::Check_Int(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (!is_str_int(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason, "Reason:\n\t the value '%s' of the command '%s' is not an int\n", str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by, error_reason);
    }
}

void CONTROLLER::Check_Int(const char* prefix, const char* command, const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (!is_str_int(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason, "Reason:\n\t the value '%s' of the command '%s' is not an int\n", str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by, error_reason);
    }
}

void CONTROLLER::Check_Float(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (!is_str_float(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason, "Reason:\n\t the value '%s' of the command '%s' is not a float\n", str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by, error_reason);
    }
}

void CONTROLLER::Check_Float(const char* prefix, const char* command, const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (!is_str_float(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(error_reason, "Reason:\n\t the value '%s' of the command '%s' is not a float\n", str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by, error_reason);
    }
}

bool CONTROLLER::Get_Bool(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (is_str_equal(str, "true"))
    {
        return true;
    }
    else if (is_str_equal(str, "false"))
    {
        return false;
    }
    else
    {
        Check_Int(command, error_by);
        return atoi(str);
    }
}

bool CONTROLLER::Get_Bool(const char* prefix, const char* command, const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (is_str_equal(str, "true"))
    {
        return true;
    }
    else if (is_str_equal(str, "false"))
    {
        return false;
    }
    else
    {
        Check_Int(prefix, command, error_by);
        return atoi(str);
    }
}


void CONTROLLER::Print_To_Screen_And_Mdout(FILE* mdout)
{

    if (mdout == NULL)
    {
        mdout = this->mdout;
    }
    int line_numbers = 0;
    for (int i = 0; i < outputs_key.size(); i++)
    {
        line_numbers++;
        fprintf(stdout, "%15s = %15s, ", outputs_key[i].c_str(), outputs_content[outputs_key[i]].c_str());
        fprintf(mdout, "%15s ", outputs_content[outputs_key[i]].c_str());
        outputs_content[outputs_key[i]] = "****";
        if (line_numbers % 3 == 0)
            fprintf(stdout, "\n");
    }
    if (line_numbers % 3 != 0)
        fprintf(stdout, "\n");
    fprintf(stdout, "------------------------------------------------------------------------------------------------------------\n");
    fprintf(mdout, "\n");
}

void CONTROLLER::Initial(int argc, char** argv, const char* subpackage_hint)
{
    if (argc == 2 && is_str_equal(argv[1], "-v", 1))
    {
        std::cout << SPONGE_VERSION << std::endl << subpackage_hint << std::endl;
        exit(0);
    }
    warn_of_initialization = 1;
    Arguments_Parse(argc, argv);
    Commands_From_In_File(argc, argv, subpackage_hint);
    printf("START INITIALIZING CONTROLLER\n");
    Default_Set();
    Init_Cuda();
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        printf("    structure last modify date is %d\n", last_modify_date);
    }
    Command_Exist("end_pause");
    printf("END INITIALIZING CONTROLLER\n\n");
}

void CONTROLLER::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        original_commands.clear();
        commands.clear();
        command_check.clear();

        fclose(mdinfo);
        fclose(mdout);
        outputs_content.clear();
        outputs_format.clear();
        outputs_key.clear();
    }
}

void CONTROLLER::Set_File_Buffer(FILE* file, size_t one_frame_size)
{
    if (setvbuf(file, NULL, _IOFBF, one_frame_size * buffer_frame) != 0)
    {
        std::string error_reason = string_format("Reason:\n\tthe trajectory file will be written every %buffer_frame% and SPONGE failed to allocate a memory for this. \
Please use the command 'buffer_frame = xxx' to decrease the buffer size", { {"buffer_frame", std::to_string(buffer_frame)}});
        Throw_SPONGE_Error(spongeErrorMallocFailed, "CONTROLLER::Set_File_Buffer", error_reason.c_str());
    }
}

void TIME_RECORDER::Start()
{
    start_timestamp = std::chrono::steady_clock::now();
}


void TIME_RECORDER::Stop()
{
    end_timestamp = std::chrono::steady_clock::now();
    time += std::chrono::duration<double>(end_timestamp - start_timestamp).count();
}

void TIME_RECORDER::Clear()
{
    time = 0;
    start_timestamp = std::chrono::steady_clock::time_point{};
    end_timestamp = std::chrono::steady_clock::time_point{};
}

void CONTROLLER::Throw_SPONGE_Error(const int error_number, const char* error_by, const char* extra_error_string)
{
    if (error_number == 0)
        return;
    std::string error_name;
    std::string error_reason;
    std::string error_by_;
    std::string extra_error_string_;
    if (error_number <= 1000)
    {
        error_name = cudaGetErrorName((cudaError)error_number);
        error_reason = cudaGetErrorString((cudaError)error_number);
    }
    else
    {
        switch (error_number)
        {
        case spongeErrorNotImplemented:
        {
            error_name = "spongeErrorNotImplemented";
            error_reason = "The function has not been implemented in SPONGE yet";
            break;
        }
        case spongeErrorBadFileFormat:
        {
            error_name = "spongeErrorBadFileFormat";
            error_reason = "The format of the file is bad";
            break;
        }
        case spongeErrorConflictingCommand:
        {
            error_name = "spongeErrorConflictingCommand";
            error_reason = "Some commands are conflicting";
            break;
        }
        case spongeErrorMissingCommand:
        {
            error_name = "spongeErrorMissingCommand";
            error_reason = "Missing required command(s)";
            break;
        }
        case spongeErrorTypeErrorCommand:
        {
            error_name = "spongeErrorMissingCommand";
            error_reason = "The type of the command is wrong";
            break;
        }
        case spongeErrorValueErrorCommand:
        {
            error_name = "spongeErrorValueErrorCommand";
            error_reason = "The value of the command is wrong";
            break;
        }
        case spongeErrorSimulationBreakDown:
        {
            error_name = "spongeErrorSimulationBrokenDown";
            error_reason = "The system was broken down";
            break;
        }
        case spongeErrorMallocFailed:
        {
            error_name = "spongeErrorMallocFailed";
            error_reason = "Fail to allocate memory";
            break;
        }
        case spongeErrorOverflow:
        {
            error_name = "spongeErrorOverflow";
            error_reason = "Boundary was overflowed";
            break;
        }
        case spongeErrorOpenFileFailed:
        {
            error_name = "spongeErrorOpenFileFailed";
            error_reason = "Fail to open file";
            break;
        }
        default:
        {
            error_name = "spongeErrorUnclassified";
            error_reason = "Unclassified Error";
        }
        }
    }
    if (error_by != NULL)
    {
        error_by_ = std::string(" raised by ") + error_by;
    }
    else
    {
        error_by_ = std::string("");
    }
    if (extra_error_string != NULL)
    {
        extra_error_string_ = extra_error_string;
        if (extra_error_string_.back() != '\n')
        {
            extra_error_string_ += "\n";
        }
    }
    else
    {
        extra_error_string_ = "";
    }
    printf("\n%s%s\n%s\n%s", error_name.c_str(), error_by_.c_str(), error_reason.c_str(), extra_error_string_.c_str());
    fcloseall();
    exit(error_number);     
}

void CONTROLLER::Warn(const char* warning)
{
    if (warn_of_initialization)
    {
        warnings.push_back(warning);
    }
    else
    {
        printf("Warning: %s\n", warning);
    }
}

bool Malloc_Safely(void** address, size_t size)
{
    address[0] = NULL;
    address[0] = (void*)malloc(size);
    if (address[0] != NULL)
    {
        return true;
    }
    else
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed, "Malloc_Safely");
#endif
        return false;
    }
}
bool Cuda_Malloc_Safely(void** address, size_t size)
{
    cudaError_t cuda_error = cudaMalloc(&address[0], size);
    if (cuda_error == 0)
    {
        return true;
    }
    else
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed, "Cuda_Malloc_Safely");
#endif
        return false;
    }
}

bool Open_File_Safely(FILE** file, const char* file_name, const char* open_type)
{
    file[0] = NULL;
    file[0] = fopen(file_name, open_type);
    if (file[0] == NULL)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        std::string output = "Open_File_Safely(";
        output += file_name;
        output += ")";
        controller.Throw_SPONGE_Error(spongeErrorOpenFileFailed, output.c_str());
#endif
        return false;
    }
    else
    {
        std::string open_type_str = open_type;
        if (open_type_str == "r")
        {
            fclose(file[0]);
            file[0] = fopen(file_name, "rb");
            fseek(file[0], -2, SEEK_END);
            if (ftell(file[0]) < 2)
            {
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(spongeErrorOpenFileFailed, output.c_str(), 
                    string_format("%FNAME% is an empty file", { {"FNAME", file_name}}).c_str());
#endif
                return false;
            }
            bool hasCR = false;
            bool hasLF = false;
            fseek(file[0], 0, SEEK_SET);
            char ch;
            while ((ch = fgetc(file[0])) != EOF)
            {
                if (ch == '\r')
                {
                    hasCR = true;
                }
                else if (ch == '\n')
                {
                    hasLF = true;
                    break;
                }
            }
            fclose(file[0]);
            file[0] = fopen(file_name, open_type);
            if (hasCR && hasLF)
            {
#ifndef _WIN32
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(spongeErrorOpenFileFailed, output.c_str(),
                    string_format("%FNAME% is a file from Windows, but SPONGE you use is on Linux\n\
The shell commands like 'dos2unix %FNAME%' or 'sed -i 's/\\r$//' %FNAME%' may help you convert the file format", {{"FNAME", file_name}}).c_str());
#endif
                return false;
#endif 
            }
            else if (hasLF)
            {
#ifdef _WIN32
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(spongeErrorOpenFileFailed, output.c_str(),
                    string_format("%FNAME% is a file from Linux, but SPONGE you use is on Windows\n", { {"FNAME", file_name} }).c_str());
#endif
                return false;
#endif
            }
        }
        return true;
    }
}
bool Cuda_Malloc_And_Copy_Safely(void** d_address, void* h_address, size_t size, const char* var_name)
{
    cudaError_t cuda_error;
    if (var_name == NULL)
        var_name = "unnamed var";
    cuda_error = cudaMalloc(&d_address[0], size);
    if (cuda_error != 0)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed, "Cuda_Malloc_And_Copy_Safely");
#endif
        return false;
    }
    cuda_error = cudaMemcpy(d_address[0], h_address, size, cudaMemcpyHostToDevice);
    if (cuda_error != 0)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed, "Cuda_Malloc_And_Copy_Safely");
#endif
        return false;
    }
    return true;
}

static __inline__ void* skip_space_lines(char* buffer, FILE* fp)
{
    do {
        if (fgets(buffer, CHAR_LENGTH_MAX, fp) == NULL)
        {
            return NULL;
        }
    } while (strlen(buffer) < 2);
    return buffer;
}

void Configuration_Reader::Open(std::string filename)
{
    Open_File_Safely(&f, filename.c_str(), "r");
    char buffer[CHAR_LENGTH_MAX], buffer2[CHAR_LENGTH_MAX];
    while (1)
    {
        if (skip_space_lines(buffer, f) == NULL)
        {
            break;
        }
        if (sscanf(buffer, "[[[%s]]]", buffer2) != 1)
        {
            error_reason = "Fail to read a new section '[[[ SECTION ]]]'";
        }
        std::string section = string_strip(buffer2);
        sections.push_back(section);
        while (1)
        {
            if (skip_space_lines(buffer, f) == NULL)
            {
                error_reason = string_format("Fail to read the end of the section '[[ end ]]' for %SECTION%", { {"SECTION", section} });
                break;
            }
            if (sscanf(buffer, "[[%s]]", buffer2) != 1)
            {
                error_reason = string_format("Fail to read a new key '[[ KEYWORD ]]' for %SECTION%", { {"SECTION", section} });
            }
            std::string key = string_strip(buffer2);
            if (key == "end")
            {
                break;
            }
            keys[section].push_back(key);
            auto pos = ftell(f);
            std::string value;
            while (skip_space_lines(buffer, f) != NULL)
            {
                strcpy(buffer2, string_strip(buffer).c_str());
                if (sscanf(buffer2, "[[%s]]", buffer) == 1)
                {
                    fseek(f, pos, SEEK_SET);
                    break;
                }
                value += buffer2;
                pos = ftell(f);
            }
            if (value.empty())
            {
                error_reason = string_format("Fail to read the value of [[[ %SECTION% ]]] [[ %KEY% ]]", { {"KEY", key}, {"SECTION", section} });
            }
            values[{section, key}] = value;
            value_unused.insert({ section, key });
        }
    }
}

bool Configuration_Reader::Section_Exist(std::string section)
{
    return keys.count(section);
}

bool Configuration_Reader::Key_Exist(std::string section, std::string key)
{
    return values.count({ section, key });
}

std::string Configuration_Reader::Get_Value(std::string section, std::string key)
{
    value_unused.erase({ section, key });
    return values[{section, key}];
}

void Configuration_Reader::Close()
{
    fclose(f);
}

void JIT_Function::Compile(std::string source)
{
    std::string common_cuh =
#include "jit.h"
        const char* headers[1] = { common_cuh.c_str() };
    const char* header_names[1] = { "common.h" };
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, headers, header_names);

    std::string arch = "-arch sm_";
    arch += CUDA_ARCH_BIN;
    const char* opts[] = { "--use_fast_math", arch.c_str() };
    if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS)
    {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log_ = new char[logSize];
        nvrtcGetProgramLog(prog, log_);
        error_reason = log_;
        delete log_;
        return;
    }
    size_t pos1 = source.find("extern");
    size_t pos2 = source.find(string_format("%q%C%q%", { {"q", {'"'}} }), pos1);
    if (pos2 == source.npos)
    {
        error_reason = R"(extern "C" should be placed in front of the function name)";
        return;
    }
    pos1 = source.find_first_of("(", pos2);
    pos2 = source.find_last_of(" ", pos1);
    std::string name = string_strip(source.substr(pos2, pos1 - pos2));
    if (name == "__launch_bounds__")
    {
        pos1 = source.find_first_of("(", pos1 + 1);
        pos2 = source.find_last_of(" ", pos1);
        name = string_strip(source.substr(pos2, pos1 - pos2));
    }
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);
    CUmodule module;
    if (cuModuleLoadDataEx(&module, ptx, 0, 0, 0) != CUDA_SUCCESS)
    {
        error_reason = string_format("Fail to load the module from PTX for %f%", { {"f", name} });
        return;
    }
    if (cuModuleGetFunction(&function, module, name.c_str()) != CUDA_SUCCESS)
    {
        error_reason = string_format("Fail to get the name from the module for %f%", { {"f", name} });
        return;
    }
    delete ptx;

    return;
}

CUresult JIT_Function::operator() (dim3 blocks, dim3 threads, cudaStream_t stream, unsigned int shared_memory_size, std::vector<void*> args)
{
    return cuLaunchKernel(function,
        blocks.x, blocks.y, blocks.z,
        threads.x, threads.y, threads.z,
        shared_memory_size, stream,
        &args[0], NULL);
}
