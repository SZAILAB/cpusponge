#include "cpu_phase2_gates.h"

#include <algorithm>
#include <cctype>
#ifdef __linux__
#include <dlfcn.h>
#endif
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

void LogReenabledFeature(CONTROLLER* controller, const char* feature_id,
                         const char* command_hint, const char* mode_hint) {
    controller->printf("PHASE2-REENABLED: %s requested by %s (%s)\n",
                       feature_id, command_hint, mode_hint);
}

void ThrowScopedPluginConfigError(CONTROLLER* controller, const std::string& reason) {
    controller->Throw_SPONGE_Error(
        spongeErrorValueErrorCommand,
        "cpu_phase2::Check_Config_Gates",
        reason.c_str());
}

void ThrowScopedCVCombineConfigError(CONTROLLER* controller, const std::string& reason) {
    controller->Throw_SPONGE_Error(
        spongeErrorValueErrorCommand,
        "cpu_phase2::Check_Config_Gates",
        reason.c_str());
}

void ThrowScopedPluginRuntimeError(CONTROLLER* controller, const std::string& reason) {
    controller->Throw_SPONGE_Error(
        spongeErrorNotImplemented,
        "cpu_phase2::Check_Config_Gates",
        reason.c_str());
}

bool IsRegularFile(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat st {};
    return (stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

bool EndsWith(const std::string& value, const char* suffix) {
    const std::string sfx(suffix);
    if (value.size() < sfx.size()) {
        return false;
    }
    return value.compare(value.size() - sfx.size(), sfx.size(), sfx) == 0;
}

std::vector<std::string> SplitTokens(const std::string& raw) {
    std::vector<std::string> out;
    std::istringstream iss(raw);
    std::string token;
    while (iss >> token) {
        out.push_back(token);
    }
    return out;
}

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::string Trim(const std::string& text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string StripInlineComment(const std::string& text) {
    const size_t hash_pos = text.find('#');
    if (hash_pos == std::string::npos) {
        return text;
    }
    return text.substr(0, hash_pos);
}

struct CVCombineSectionConfig {
    bool is_combination = false;
    std::vector<std::string> cv_terms;
    std::string function_expr;
};

void ParseCVConfigFile(const std::string& cv_in_file,
                       std::map<std::string, CVCombineSectionConfig>* out_configs) {
    std::ifstream in(cv_in_file);
    if (!in) {
        return;
    }
    std::string active_section;
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        std::string line = Trim(StripInlineComment(raw_line));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            active_section = Trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (line.back() == '{') {
            active_section = Trim(line.substr(0, line.size() - 1));
            continue;
        }
        if (line == "}") {
            active_section.clear();
            continue;
        }
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }
        const std::string key = Trim(line.substr(0, eq_pos));
        const std::string value = Trim(line.substr(eq_pos + 1));
        if (key.empty()) {
            continue;
        }

        std::string section = active_section;
        std::string field = key;
        if (section.empty()) {
            const size_t split_pos = key.rfind('_');
            if (split_pos == std::string::npos || split_pos == 0 || split_pos + 1 >= key.size()) {
                continue;
            }
            section = key.substr(0, split_pos);
            field = key.substr(split_pos + 1);
        }
        if (section.empty()) {
            continue;
        }

        const std::string lower_field = ToLower(field);
        CVCombineSectionConfig& cfg = out_configs->operator[](section);
        if (lower_field == "cv_type") {
            cfg.is_combination = (ToLower(value) == "combination");
        } else if (lower_field == "cv") {
            cfg.cv_terms = SplitTokens(value);
        } else if (lower_field == "function") {
            cfg.function_expr = value;
        }
    }
}

bool IsAllowedCVCombineCharacter(char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || std::isspace(uc)) {
        return true;
    }
    switch (c) {
        case '_':
        case '+':
        case '-':
        case '*':
        case '/':
        case '(':
        case ')':
        case '.':
        case ',':
            return true;
        default:
            return false;
    }
}

std::vector<std::string> ExtractIdentifierTokens(const std::string& expr) {
    std::vector<std::string> tokens;
    for (size_t i = 0; i < expr.size();) {
        const unsigned char c = static_cast<unsigned char>(expr[i]);
        if (std::isalpha(c) || expr[i] == '_') {
            size_t j = i + 1;
            while (j < expr.size()) {
                const unsigned char cj = static_cast<unsigned char>(expr[j]);
                if (!std::isalnum(cj) && expr[j] != '_') {
                    break;
                }
                ++j;
            }
            tokens.push_back(expr.substr(i, j - i));
            i = j;
        } else {
            ++i;
        }
    }
    return tokens;
}

bool IsAllowedCVCombineBuiltin(const std::string& lower_token) {
    static const std::set<std::string> kAllowed = {
        "abs", "acos", "asin", "atan", "ceil", "cos", "exp",
        "fabs", "floor", "log", "max", "min", "pow", "sin",
        "sqrt", "tan",
    };
    return kAllowed.count(lower_token) > 0;
}

bool ValidateCVCombineScopedInputs(CONTROLLER* controller, const std::string& cv_in_file) {
    std::map<std::string, CVCombineSectionConfig> configs;
    ParseCVConfigFile(cv_in_file, &configs);

    bool combination_requested = false;
    for (const auto& item : configs) {
        const std::string& section_name = item.first;
        const CVCombineSectionConfig& cfg = item.second;
        if (!cfg.is_combination) {
            continue;
        }
        combination_requested = true;

        if (cfg.cv_terms.empty()) {
            std::string reason = "Reason:\n\tcv_combine section `";
            reason += section_name;
            reason += "` is missing required `CV` terms.\n\tTrack: PHASE2-REENABLE-CV-COMBINE";
            ThrowScopedCVCombineConfigError(controller, reason);
        }
        if (Trim(cfg.function_expr).empty()) {
            std::string reason = "Reason:\n\tcv_combine section `";
            reason += section_name;
            reason += "` is missing required `function` expression.\n\tTrack: PHASE2-REENABLE-CV-COMBINE";
            ThrowScopedCVCombineConfigError(controller, reason);
        }
        for (char c : cfg.function_expr) {
            if (!IsAllowedCVCombineCharacter(c)) {
                std::string reason = "Reason:\n\tcv_combine function contains unsupported character `";
                reason.push_back(c);
                reason += "` in section `";
                reason += section_name;
                reason += "`.\n\tTrack: PHASE2-REENABLE-CV-COMBINE";
                ThrowScopedCVCombineConfigError(controller, reason);
            }
        }

        std::set<std::string> allowed_identifiers(cfg.cv_terms.begin(), cfg.cv_terms.end());
        const std::vector<std::string> identifier_tokens = ExtractIdentifierTokens(cfg.function_expr);
        for (const auto& token : identifier_tokens) {
            if (allowed_identifiers.count(token) > 0) {
                continue;
            }
            const std::string lower_token = ToLower(token);
            if (lower_token == "e" || lower_token == "pi" || IsAllowedCVCombineBuiltin(lower_token)) {
                continue;
            }
            std::string reason = "Reason:\n\tcv_combine function uses unknown identifier `";
            reason += token;
            reason += "` in section `";
            reason += section_name;
            reason += "`.\n\tTrack: PHASE2-REENABLE-CV-COMBINE";
            ThrowScopedCVCombineConfigError(controller, reason);
        }

        controller->printf("PHASE2-CV-COMBINE: scoped expression check passed for section %s\n",
                           section_name.c_str());
    }

    return combination_requested;
}

#ifdef __linux__
using PluginNameFunc = std::string (*)();
using PluginVersionCheckFunc = std::string (*)(int);

std::string LastDlError() {
    const char* err = dlerror();
    if (err == nullptr) {
        return "unknown dynamic loader error";
    }
    return err;
}

void RequirePluginSymbolOrThrow(CONTROLLER* controller, void* handle,
                                const std::string& plugin_path,
                                const char* symbol_name) {
    dlerror();
    const void* symbol = dlsym(handle, symbol_name);
    const char* err = dlerror();
    if (symbol == nullptr || err != nullptr) {
        std::string reason = "Reason:\n\tPlugin ";
        reason += plugin_path;
        reason += " is missing required symbol `";
        reason += symbol_name;
        reason += "`.\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME";
        ThrowScopedPluginRuntimeError(controller, reason);
    }
}

void ValidatePluginRuntimeScopedAbi(CONTROLLER* controller, const std::string& path) {
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        std::string reason = "Reason:\n\tFailed to load plugin shared object: ";
        reason += path;
        reason += "\n\t";
        reason += LastDlError();
        reason += "\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME";
        ThrowScopedPluginRuntimeError(controller, reason);
    }

    RequirePluginSymbolOrThrow(controller, handle, path, "Name");
    RequirePluginSymbolOrThrow(controller, handle, path, "Version");
    RequirePluginSymbolOrThrow(controller, handle, path, "Version_Check");
    RequirePluginSymbolOrThrow(controller, handle, path, "Initial");

    PluginNameFunc name_func = reinterpret_cast<PluginNameFunc>(dlsym(handle, "Name"));
    PluginNameFunc version_func = reinterpret_cast<PluginNameFunc>(dlsym(handle, "Version"));
    PluginVersionCheckFunc version_check_func =
        reinterpret_cast<PluginVersionCheckFunc>(dlsym(handle, "Version_Check"));

    const std::string plugin_name = name_func();
    const std::string plugin_version = version_func();
    const std::string version_check_error = version_check_func(controller->last_modify_date);
    if (!version_check_error.empty()) {
        std::string reason = "Reason:\n\tPlugin version check failed for ";
        reason += path;
        reason += " (";
        reason += plugin_name;
        reason += " ";
        reason += plugin_version;
        reason += "): ";
        reason += version_check_error;
        reason += "\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME";
        dlclose(handle);
        ThrowScopedPluginConfigError(controller, reason);
    }

    controller->printf(
        "PHASE2-PLUGIN-RUNTIME: scoped ABI check passed for %s (name=%s version=%s)\n",
        path.c_str(), plugin_name.c_str(), plugin_version.c_str());
    dlclose(handle);
}
#endif

void ValidatePluginRuntimeScopedInputs(CONTROLLER* controller) {
    const std::string plugin_raw = controller->Original_Command("plugin");
    const std::vector<std::string> plugin_paths = SplitTokens(plugin_raw);
    if (plugin_paths.empty()) {
        ThrowScopedPluginConfigError(
            controller,
            "Reason:\n\tCommand `plugin` is set but no plugin path is provided.\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME");
    }
    for (const auto& path : plugin_paths) {
        if (!EndsWith(path, ".so")) {
            ThrowScopedPluginConfigError(
                controller,
                "Reason:\n\tCPU scoped plugin runtime only accepts `.so` plugin paths in Phase 2 gatecheck.\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME");
        }
        if (!IsRegularFile(path)) {
            std::string reason = "Reason:\n\tplugin path does not exist or is not a regular file: ";
            reason += path;
            reason += "\n\tTrack: PHASE2-REENABLE-PLUGIN-RUNTIME";
            controller->Throw_SPONGE_Error(
                spongeErrorOpenFileFailed,
                "cpu_phase2::Check_Config_Gates",
                reason.c_str());
        }
#ifdef __linux__
        ValidatePluginRuntimeScopedAbi(controller, path);
#else
        controller->printf("PHASE2-PLUGIN-RUNTIME: scoped file check passed for %s\n", path.c_str());
#endif
    }
}

}  // namespace

namespace cpu_phase2 {

void Check_Config_Gates(CONTROLLER* controller) {
    if (controller->Command_Exist("pairwise_force", "in_file")) {
        LogReenabledFeature(controller, "GATE-JIT-PAIRWISE", "pairwise_force_in_file", "phase2_stub_static");
    }
    if (controller->Command_Exist("listed_forces", "in_file")) {
        LogReenabledFeature(controller, "GATE-JIT-LISTED", "listed_forces_in_file", "phase2_stub_static");
    }
    if (controller->Command_Exist("soft_walls", "in_file")) {
        LogReenabledFeature(controller, "GATE-JIT-SOFTWALL", "soft_walls_in_file", "phase2_stub_static");
    }

    if (controller->Command_Exist("cv_in_file")) {
        const std::string cv_in_file = controller->Command("cv_in_file");
        if (!IsRegularFile(cv_in_file)) {
            std::string reason = "Reason:\n\tcv_in_file does not exist or is not a regular file: ";
            reason += cv_in_file;
            controller->Throw_SPONGE_Error(
                spongeErrorOpenFileFailed,
                "cpu_phase2::Check_Config_Gates",
                reason.c_str());
        }
        if (ValidateCVCombineScopedInputs(controller, cv_in_file)) {
            LogReenabledFeature(controller, "GATE-JIT-CV-COMBINE", "cv_in_file", "phase2_runtime_scoped");
        }
    }

    if (controller->Command_Exist("plugin")) {
        ValidatePluginRuntimeScopedInputs(controller);
        LogReenabledFeature(controller, "GATE-PLUGIN-RUNTIME", "plugin", "phase2_runtime_scoped");
    }
}

}  // namespace cpu_phase2
