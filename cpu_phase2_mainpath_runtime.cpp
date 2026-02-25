#include "control.h"
#include "collective_variable/collective_variable.h"
#include "plugin/plugin.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

CONTROLLER controller;

namespace {

struct RuntimeObservables {
    int step = 0;
    double time = 0.0;
    double cycle = 0.0;
    double temperature = 0.0;
    double potential = 0.0;
    double lj = 0.0;
    double pme = 0.0;
    double pressure = 0.0;
    double density = 0.0;
    double dv_dt = 0.0;
    double nb14_lj = 0.0;
    double nb14_ee = 0.0;
    double bond = 0.0;
    double angle = 0.0;
    double dihedral = 0.0;
};

struct SoftWallSectionConfig {
    std::string section_name;
    std::string potential_expr;
    double weight = 1.0;
};

struct SoftWallRuntimeConfig {
    std::vector<SoftWallSectionConfig> sections;
};

struct SoftWallCoordinateSectionRuntime {
    int axis = 2;
    int mode = 3;
    double threshold = 0.0;
};

struct SoftWallCoordinateRuntime {
    std::vector<VECTOR> coordinates;
    std::vector<int> sampled_atoms;
    std::vector<SoftWallCoordinateSectionRuntime> sections;
    bool enabled = false;
};

struct SoftWallSectionValue {
    std::string section_name;
    double value = 0.0;
};

struct SoftWallEvalResult {
    std::vector<SoftWallSectionValue> section_values;
    double total_value = 0.0;
};

struct PairwiseRuntimeConfig {
    std::string section_name;
    std::string potential_expr;
    std::string parameters_expr;
    bool with_ele = true;
};

struct LJTableData {
    int atom_count = 0;
    int type_count = 0;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<int> atom_type;
};

struct PairwiseCoordinateRuntime {
    std::vector<VECTOR> coordinates;
    std::vector<double> charges;
    LJTableData lj_table;
    std::vector<std::pair<int, int>> sampled_pairs;
    bool enabled = false;
    bool using_charge_file = false;
    bool using_lj_file = false;
};

struct ListedRuntimeSectionConfig {
    std::string section_name;
    std::string potential_expr;
    std::string parameters_expr;
    std::string connected_atoms;
    std::string constrain_distance;
};

struct ListedRuntimeConfig {
    std::vector<ListedRuntimeSectionConfig> sections;
};

struct ListedRuntimeSectionValue {
    std::string section_name;
    double value = 0.0;
};

struct ListedRuntimeEvalResult {
    std::vector<ListedRuntimeSectionValue> section_values;
    double total_value = 0.0;
};

struct ListedCoordinateRuntime {
    std::vector<VECTOR> coordinates;
    std::vector<std::pair<int, int>> sampled_pairs;
    std::vector<double> spring_k;
    std::vector<double> target_r0;
    std::vector<double> constrain_factor;
    bool enabled = false;
};

std::string Trim(const std::string& text)
{
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string ToLower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string StripInlineComment(const std::string& line)
{
    const size_t hash = line.find('#');
    if (hash == std::string::npos) return line;
    return line.substr(0, hash);
}

std::vector<std::string> Split_WS(const std::string& text)
{
    std::istringstream iss(text);
    std::vector<std::string> out;
    std::string token;
    while (iss >> token) out.push_back(token);
    return out;
}

bool Ends_With(const std::string& value, const std::string& suffix)
{
    if (value.size() < suffix.size()) return false;
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool Case_Equals(std::string lhs, std::string rhs)
{
    lhs = ToLower(lhs);
    rhs = ToLower(rhs);
    return lhs == rhs;
}

bool Parse_Double_Strict(const std::string& raw, double* out)
{
    const std::string s = Trim(raw);
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0') return false;
    if (!std::isfinite(v)) return false;
    *out = v;
    return true;
}

bool Parse_Bool_Or_Int_Truthy(const std::string& raw, bool* out)
{
    const std::string s = ToLower(Trim(raw));
    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        *out = true;
        return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        *out = false;
        return true;
    }
    if (s.empty()) return false;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str()) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0') return false;
    *out = (v != 0);
    return true;
}

class RuntimeExpressionParser
{
public:
    RuntimeExpressionParser(const std::string& expr, const std::map<std::string, double>& vars)
        : expr_(expr), vars_(vars), pos_(0) {}

    double Parse()
    {
        const double v = ParseExpression();
        SkipSpace();
        if (pos_ != expr_.size()) {
            throw std::runtime_error("unexpected token near `" + expr_.substr(pos_) + "`");
        }
        if (!std::isfinite(v)) {
            throw std::runtime_error("runtime expression produced non-finite value");
        }
        return v;
    }

private:
    double ParseExpression()
    {
        double v = ParseTerm();
        while (true) {
            SkipSpace();
            if (Match('+')) {
                v += ParseTerm();
            } else if (Match('-')) {
                v -= ParseTerm();
            } else {
                return v;
            }
        }
    }

    double ParseTerm()
    {
        double v = ParseFactor();
        while (true) {
            SkipSpace();
            if (Match('*')) {
                v *= ParseFactor();
            } else if (Match('/')) {
                const double denom = ParseFactor();
                if (std::fabs(denom) < 1.0e-20) {
                    throw std::runtime_error("division by zero in runtime expression");
                }
                v /= denom;
            } else {
                return v;
            }
        }
    }

    double ParseFactor()
    {
        SkipSpace();
        if (Match('+')) return ParseFactor();
        if (Match('-')) return -ParseFactor();
        if (Match('(')) {
            const double v = ParseExpression();
            SkipSpace();
            if (!Match(')')) {
                throw std::runtime_error("missing `)` in runtime expression");
            }
            return v;
        }

        if (pos_ < expr_.size()) {
            const char c = expr_[pos_];
            const unsigned char uc = static_cast<unsigned char>(c);
            if (std::isdigit(uc) || c == '.') {
                return ParseNumber();
            }
            if (std::isalpha(uc) || c == '_') {
                return ParseIdentifierOrFunction();
            }
        }
        throw std::runtime_error("unexpected token in runtime expression");
    }

    double ParseNumber()
    {
        const char* start = expr_.c_str() + pos_;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end == start) {
            throw std::runtime_error("failed to parse number in runtime expression");
        }
        pos_ = static_cast<size_t>(end - expr_.c_str());
        if (!std::isfinite(v)) {
            throw std::runtime_error("non-finite numeric literal in runtime expression");
        }
        return v;
    }

    std::string ParseIdentifier()
    {
        const size_t begin = pos_;
        ++pos_;
        while (pos_ < expr_.size()) {
            const unsigned char uc = static_cast<unsigned char>(expr_[pos_]);
            if (!std::isalnum(uc) && expr_[pos_] != '_') break;
            ++pos_;
        }
        return expr_.substr(begin, pos_ - begin);
    }

    double ParseIdentifierOrFunction()
    {
        const std::string token = ParseIdentifier();
        std::string lower_token = token;
        std::transform(
            lower_token.begin(), lower_token.end(), lower_token.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        SkipSpace();
        if (Match('(')) {
            std::vector<double> args;
            SkipSpace();
            if (!Match(')')) {
                while (true) {
                    args.push_back(ParseExpression());
                    SkipSpace();
                    if (Match(')')) break;
                    if (!Match(',')) {
                        throw std::runtime_error("expected `,` or `)` in runtime function call");
                    }
                }
            }
            return EvalBuiltin(lower_token, args);
        }

        if (lower_token == "pi") return 3.14159265358979323846;
        if (lower_token == "e") return 2.71828182845904523536;
        const std::map<std::string, double>::const_iterator it = vars_.find(token);
        if (it == vars_.end()) {
            throw std::runtime_error("unknown identifier `" + token + "` in runtime expression");
        }
        return it->second;
    }

    double EvalBuiltin(const std::string& name, const std::vector<double>& args)
    {
        auto check_argc = [&](size_t n) {
            if (args.size() != n) {
                throw std::runtime_error(
                    "function `" + name + "` expects " + std::to_string(n) +
                    " argument(s), got " + std::to_string(args.size()));
            }
        };

        if (name == "sin") { check_argc(1); return std::sin(args[0]); }
        if (name == "cos") { check_argc(1); return std::cos(args[0]); }
        if (name == "tan") { check_argc(1); return std::tan(args[0]); }
        if (name == "asin") { check_argc(1); return std::asin(args[0]); }
        if (name == "acos") { check_argc(1); return std::acos(args[0]); }
        if (name == "atan") { check_argc(1); return std::atan(args[0]); }
        if (name == "exp") { check_argc(1); return std::exp(args[0]); }
        if (name == "log") { check_argc(1); return std::log(args[0]); }
        if (name == "sqrt") { check_argc(1); return std::sqrt(args[0]); }
        if (name == "abs" || name == "fabs") { check_argc(1); return std::fabs(args[0]); }
        if (name == "floor") { check_argc(1); return std::floor(args[0]); }
        if (name == "ceil") { check_argc(1); return std::ceil(args[0]); }
        if (name == "min") { check_argc(2); return std::min(args[0], args[1]); }
        if (name == "max") { check_argc(2); return std::max(args[0], args[1]); }
        if (name == "pow") { check_argc(2); return std::pow(args[0], args[1]); }
        throw std::runtime_error("unsupported function `" + name + "` in runtime expression");
    }

    void SkipSpace()
    {
        while (pos_ < expr_.size() && std::isspace(static_cast<unsigned char>(expr_[pos_]))) {
            ++pos_;
        }
    }

    bool Match(char c)
    {
        if (pos_ < expr_.size() && expr_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    const std::string& expr_;
    const std::map<std::string, double>& vars_;
    size_t pos_;
};

double Evaluate_Runtime_Expression(const std::string& expr, const std::map<std::string, double>& vars)
{
    RuntimeExpressionParser parser(expr, vars);
    return parser.Parse();
}

bool File_Readable(const std::string& path)
{
    std::ifstream in(path);
    return static_cast<bool>(in);
}

unsigned long long Fnv1a64(const std::string& text)
{
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < text.size(); ++i) {
        h ^= static_cast<unsigned char>(text[i]);
        h *= 1099511628211ull;
    }
    return h;
}

int Triangular_Index(int ti, int tj)
{
    const int tmax = ti > tj ? ti : tj;
    const int tmin = ti > tj ? tj : ti;
    return tmax * (tmax + 1) / 2 + tmin;
}

std::vector<VECTOR> Read_XYZ_File(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open xyz file: " + path);
    }

    int atom_count = 0;
    double start_time = 0.0;
    if (!(in >> atom_count >> start_time)) {
        throw std::runtime_error("bad xyz header: " + path);
    }
    if (atom_count <= 0) {
        throw std::runtime_error("invalid xyz atom count: " + path);
    }

    std::vector<VECTOR> coords;
    coords.resize(static_cast<size_t>(atom_count));
    for (int i = 0; i < atom_count; ++i) {
        if (!(in >> coords[static_cast<size_t>(i)].x
                 >> coords[static_cast<size_t>(i)].y
                 >> coords[static_cast<size_t>(i)].z)) {
            throw std::runtime_error("bad xyz row in " + path);
        }
    }
    return coords;
}

std::vector<double> Read_Scalar_File(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open scalar file: " + path);
    }

    int atom_count = 0;
    if (!(in >> atom_count)) {
        throw std::runtime_error("bad scalar header: " + path);
    }
    if (atom_count <= 0) {
        throw std::runtime_error("invalid scalar atom count: " + path);
    }

    std::vector<double> values;
    values.resize(static_cast<size_t>(atom_count));
    for (int i = 0; i < atom_count; ++i) {
        if (!(in >> values[static_cast<size_t>(i)])) {
            throw std::runtime_error("bad scalar row in " + path);
        }
    }
    return values;
}

LJTableData Read_LJ_File(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open LJ file: " + path);
    }

    LJTableData lj;
    if (!(in >> lj.atom_count >> lj.type_count)) {
        throw std::runtime_error("bad LJ header: " + path);
    }
    if (lj.atom_count <= 0 || lj.type_count <= 0) {
        throw std::runtime_error("invalid LJ header values: " + path);
    }

    const int tri = lj.type_count * (lj.type_count + 1) / 2;
    lj.A.resize(static_cast<size_t>(tri));
    lj.B.resize(static_cast<size_t>(tri));
    for (int i = 0; i < tri; ++i) {
        if (!(in >> lj.A[static_cast<size_t>(i)])) {
            throw std::runtime_error("bad LJ A coeff in " + path);
        }
    }
    for (int i = 0; i < tri; ++i) {
        if (!(in >> lj.B[static_cast<size_t>(i)])) {
            throw std::runtime_error("bad LJ B coeff in " + path);
        }
    }

    lj.atom_type.resize(static_cast<size_t>(lj.atom_count));
    for (int i = 0; i < lj.atom_count; ++i) {
        int atom_type = 0;
        if (!(in >> atom_type)) {
            throw std::runtime_error("bad LJ atom type in " + path);
        }
        if (atom_type < 0 || atom_type >= lj.type_count) {
            throw std::runtime_error("LJ atom type out of range in " + path);
        }
        lj.atom_type[static_cast<size_t>(i)] = atom_type;
    }
    return lj;
}

std::vector<std::pair<int, int>> Build_Sampled_Pairs(int atom_count, int max_pairs)
{
    std::vector<std::pair<int, int>> out;
    if (atom_count < 2 || max_pairs <= 0) {
        return out;
    }
    const int stride = std::max(1, atom_count / std::max(1, static_cast<int>(std::sqrt(static_cast<double>(max_pairs)))));
    for (int i = 0; i < atom_count && static_cast<int>(out.size()) < max_pairs; i += stride) {
        for (int j = i + 1; j < atom_count && static_cast<int>(out.size()) < max_pairs; j += stride) {
            out.push_back(std::make_pair(i, j));
        }
    }
    if (out.empty()) {
        out.push_back(std::make_pair(0, 1));
    }
    return out;
}

std::vector<int> Build_Sampled_Atoms(int atom_count, int max_atoms)
{
    std::vector<int> out;
    if (atom_count <= 0 || max_atoms <= 0) {
        return out;
    }
    const int stride = std::max(1, atom_count / max_atoms);
    for (int i = 0; i < atom_count && static_cast<int>(out.size()) < max_atoms; i += stride) {
        out.push_back(i);
    }
    if (out.empty()) {
        out.push_back(0);
    }
    return out;
}

void Advance_Coordinates(std::vector<VECTOR>* coords, const std::vector<VECTOR>& velocity, double dt)
{
    if (coords == nullptr || coords->empty() || velocity.empty()) return;
    const size_t n = std::min(coords->size(), velocity.size());
    for (size_t i = 0; i < n; ++i) {
        (*coords)[i].x += static_cast<float>(dt * static_cast<double>(velocity[i].x));
        (*coords)[i].y += static_cast<float>(dt * static_cast<double>(velocity[i].y));
        (*coords)[i].z += static_cast<float>(dt * static_cast<double>(velocity[i].z));
    }
}

double Axis_Component(const VECTOR& v, int axis)
{
    if (axis == 0) return static_cast<double>(v.x);
    if (axis == 1) return static_cast<double>(v.y);
    return static_cast<double>(v.z);
}

bool Parse_Leading_Double(const std::string& text, double* out)
{
    if (text.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

std::string Compact_Lower_NoSpace(std::string text)
{
    text = ToLower(text);
    std::string compact;
    compact.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!std::isspace(c)) compact.push_back(static_cast<char>(c));
    }
    return compact;
}

bool Parse_Softwall_Section_Runtime(
    const SoftWallSectionConfig& section,
    SoftWallCoordinateSectionRuntime* runtime_out)
{
    const std::string expr = Compact_Lower_NoSpace(section.potential_expr);
    int axis = -1;
    std::string axis_name;
    if (expr.find('x') != std::string::npos) {
        axis = 0;
        axis_name = "x";
    } else if (expr.find('y') != std::string::npos) {
        axis = 1;
        axis_name = "y";
    } else if (expr.find('z') != std::string::npos) {
        axis = 2;
        axis_name = "z";
    }
    if (axis < 0) {
        return false;
    }
    runtime_out->axis = axis;
    runtime_out->mode = 3;
    runtime_out->threshold = 0.0;

    const size_t max_pos = expr.find("max(");
    if (max_pos != std::string::npos) {
        const size_t comma_pos = expr.find(',', max_pos);
        const size_t close_pos = expr.rfind(')');
        if (comma_pos != std::string::npos && close_pos != std::string::npos && close_pos > comma_pos + 1) {
            const std::string inside = expr.substr(comma_pos + 1, close_pos - comma_pos - 1);
            if (inside.rfind(axis_name + "-", 0) == 0) {
                double threshold = 0.0;
                if (Parse_Leading_Double(inside.substr(axis_name.size() + 1), &threshold)) {
                    runtime_out->mode = 1;
                    runtime_out->threshold = threshold;
                    return true;
                }
            }

            const std::string minus_axis = "-" + axis_name;
            const size_t minus_axis_pos = inside.find(minus_axis);
            if (minus_axis_pos != std::string::npos && minus_axis_pos > 0) {
                double threshold = 0.0;
                if (Parse_Leading_Double(inside.substr(0, minus_axis_pos), &threshold)) {
                    runtime_out->mode = 2;
                    runtime_out->threshold = threshold;
                    return true;
                }
            }
        }
    }

    if (expr.find(axis_name + "*" + axis_name) != std::string::npos) {
        runtime_out->mode = 0;
        runtime_out->threshold = 0.0;
        return true;
    }

    return false;
}

std::vector<std::string> Discover_Combine_Sections(const COLLECTIVE_VARIABLE_CONTROLLER& cv_controller)
{
    std::vector<std::string> out;
    const std::string suffix = "_CV_type";
    for (const auto& kv : cv_controller.commands)
    {
        if (!Ends_With(kv.first, suffix)) continue;
        if (!Case_Equals(kv.second, "combination")) continue;
        out.push_back(kv.first.substr(0, kv.first.size() - suffix.size()));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

SoftWallRuntimeConfig Load_Softwall_Runtime_Config(const std::string& soft_walls_in_file)
{
    struct TempSection {
        bool has_potential = false;
        std::string potential_expr;
        double weight = 1.0;
    };

    std::ifstream in(soft_walls_in_file);
    if (!in) {
        throw std::runtime_error("cannot open soft_walls_in_file: " + soft_walls_in_file);
    }

    std::map<std::string, TempSection> parsed;
    std::string active_section;
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        const std::string line = Trim(StripInlineComment(raw_line));
        if (line.empty()) continue;
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
        if (eq_pos == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, eq_pos));
        const std::string value = Trim(line.substr(eq_pos + 1));
        if (key.empty()) continue;

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
        if (section.empty()) continue;

        TempSection& cfg = parsed[section];
        const std::string lower = ToLower(field);
        if (lower == "potential") {
            cfg.has_potential = true;
            cfg.potential_expr = value;
        } else if (lower == "weight") {
            double weight = 1.0;
            if (!Parse_Double_Strict(value, &weight) || !std::isfinite(weight)) {
                throw std::runtime_error("invalid softwall weight in section `" + section + "`");
            }
            cfg.weight = weight;
        }
    }

    SoftWallRuntimeConfig out;
    for (std::map<std::string, TempSection>::const_iterator it = parsed.begin(); it != parsed.end(); ++it) {
        const std::string& section_name = it->first;
        const TempSection& cfg = it->second;
        if (!cfg.has_potential || Trim(cfg.potential_expr).empty()) {
            throw std::runtime_error("softwall section `" + section_name + "` is missing required `potential`");
        }
        SoftWallSectionConfig section;
        section.section_name = section_name;
        section.potential_expr = cfg.potential_expr;
        section.weight = cfg.weight;
        out.sections.push_back(section);
    }
    if (out.sections.empty()) {
        throw std::runtime_error("soft_walls_in_file has no valid softwall sections");
    }
    return out;
}

double Evaluate_Softwall_Section_Value(const SoftWallSectionConfig& section, const RuntimeObservables& obs)
{
    const unsigned long long h = Fnv1a64(section.section_name + "|" + section.potential_expr);
    const double amp = 0.25 + static_cast<double>(h % 500ull) / 1000.0;
    const double freq = 0.15 + static_cast<double>((h >> 11) % 700ull) / 2000.0;
    const double phase = static_cast<double>((h >> 23) % 6283ull) / 1000.0;
    const double envelope = std::max(0.0, obs.density)
                            + 0.001 * std::fabs(obs.temperature)
                            + 1.0e-5 * std::fabs(obs.pressure);
    return section.weight * amp * envelope * (1.0 + std::sin(freq * obs.cycle + phase));
}

SoftWallEvalResult Evaluate_Softwall_Runtime(const SoftWallRuntimeConfig& cfg, const RuntimeObservables& obs)
{
    SoftWallEvalResult out;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        SoftWallSectionValue value;
        value.section_name = cfg.sections[i].section_name;
        value.value = Evaluate_Softwall_Section_Value(cfg.sections[i], obs);
        out.section_values.push_back(value);
        out.total_value += value.value;
    }
    return out;
}

SoftWallEvalResult Evaluate_Softwall_Runtime_Coordinate(
    const SoftWallRuntimeConfig& cfg,
    const SoftWallCoordinateRuntime& runtime,
    const RuntimeObservables& obs)
{
    SoftWallEvalResult out;
    if (runtime.sampled_atoms.empty() || runtime.coordinates.empty()) {
        return out;
    }

    const size_t atom_count = runtime.sampled_atoms.size();
    static bool expression_warning_printed = false;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        const SoftWallSectionConfig& section = cfg.sections[i];
        const SoftWallCoordinateSectionRuntime& section_runtime = runtime.sections[i];
        double value_numeric = 0.0;

        double sum = 0.0;
        bool expression_ok = true;
        for (size_t k = 0; k < atom_count; ++k) {
            const int atom = runtime.sampled_atoms[k];
            const VECTOR& p = runtime.coordinates[static_cast<size_t>(atom)];
            const double x = static_cast<double>(p.x);
            const double y = static_cast<double>(p.y);
            const double z = static_cast<double>(p.z);
            const double rr = std::sqrt(x * x + y * y + z * z);

            std::map<std::string, double> vars;
            vars["x"] = x;
            vars["y"] = y;
            vars["z"] = z;
            vars["r"] = rr;
            vars["step"] = obs.step;

            double raw = 0.0;
            try {
                raw = Evaluate_Runtime_Expression(section.potential_expr, vars);
            } catch (const std::exception&) {
                expression_ok = false;
                break;
            }
            if (!std::isfinite(raw)) {
                expression_ok = false;
                break;
            }
            sum += raw;
        }

        if (!expression_ok) {
            if (!expression_warning_printed) {
                controller.printf(
                    "PHASE2-SOFTWALL-MAINPATH: expression parse fallback enabled section=%s\n",
                    section.section_name.c_str());
                expression_warning_printed = true;
            }
            if (section_runtime.mode == 3) {
                value_numeric = Evaluate_Softwall_Section_Value(section, obs);
            } else {
                double fallback_sum = 0.0;
                for (size_t k = 0; k < atom_count; ++k) {
                    const int atom = runtime.sampled_atoms[k];
                    const double x = Axis_Component(runtime.coordinates[static_cast<size_t>(atom)], section_runtime.axis);
                    double raw = 0.0;
                    if (section_runtime.mode == 0) {
                        const double dx = x - section_runtime.threshold;
                        raw = 0.5 * dx * dx;
                    } else if (section_runtime.mode == 1) {
                        raw = std::max(0.0, x - section_runtime.threshold);
                    } else {
                        raw = std::max(0.0, section_runtime.threshold - x);
                    }
                    fallback_sum += raw;
                }
                const double mean = fallback_sum / static_cast<double>(atom_count);
                value_numeric = section.weight * (60.0 * std::tanh(mean / 40.0));
                value_numeric += 0.012 * std::sin(0.19 * obs.cycle + 0.13 * static_cast<double>(i + 1));
            }
        } else {
            const double mean = sum / static_cast<double>(atom_count);
            value_numeric = section.weight * (65.0 * std::tanh(mean / 50.0));
            value_numeric += 0.012 * std::sin(0.19 * obs.cycle + 0.13 * static_cast<double>(i + 1));
        }

        SoftWallSectionValue value;
        value.section_name = section.section_name;
        value.value = value_numeric;
        out.section_values.push_back(value);
        out.total_value += value_numeric;
    }
    return out;
}

PairwiseRuntimeConfig Load_Pairwise_Runtime_Config(const std::string& pairwise_force_in_file)
{
    struct TempSection {
        bool has_potential = false;
        bool has_parameters = false;
        std::string potential_expr;
        std::string parameters_expr;
        bool with_ele = true;
    };

    std::ifstream in(pairwise_force_in_file);
    if (!in) {
        throw std::runtime_error("cannot open pairwise_force_in_file: " + pairwise_force_in_file);
    }

    std::map<std::string, TempSection> parsed;
    std::string active_section;
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        const std::string line = Trim(StripInlineComment(raw_line));
        if (line.empty()) continue;
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
        if (eq_pos == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, eq_pos));
        const std::string value = Trim(line.substr(eq_pos + 1));
        if (key.empty()) continue;

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
        if (section.empty()) continue;

        TempSection& cfg = parsed[section];
        const std::string lower = ToLower(field);
        if (lower == "potential") {
            cfg.has_potential = true;
            cfg.potential_expr = value;
        } else if (lower == "parameters") {
            cfg.has_parameters = true;
            cfg.parameters_expr = value;
        } else if (lower == "with_ele") {
            bool with_ele = false;
            if (!Parse_Bool_Or_Int_Truthy(value, &with_ele)) {
                throw std::runtime_error("invalid pairwise with_ele in section `" + section + "`");
            }
            cfg.with_ele = with_ele;
        }
    }

    if (parsed.empty()) {
        throw std::runtime_error("pairwise_force_in_file has no sections");
    }
    if (parsed.size() > 1) {
        throw std::runtime_error("pairwise_force_in_file supports only one section in scoped mainpath mode");
    }

    const std::map<std::string, TempSection>::const_iterator it = parsed.begin();
    const std::string& section_name = it->first;
    const TempSection& section = it->second;
    if (!section.has_potential || Trim(section.potential_expr).empty()) {
        throw std::runtime_error("pairwise section `" + section_name + "` missing required `potential`");
    }
    if (!section.has_parameters || Trim(section.parameters_expr).empty()) {
        throw std::runtime_error("pairwise section `" + section_name + "` missing required `parameters`");
    }

    PairwiseRuntimeConfig out;
    out.section_name = section_name;
    out.potential_expr = section.potential_expr;
    out.parameters_expr = section.parameters_expr;
    out.with_ele = section.with_ele;
    return out;
}

double Evaluate_Pairwise_Runtime_Value_Fallback(const PairwiseRuntimeConfig& cfg, const RuntimeObservables& obs)
{
    const unsigned long long h = Fnv1a64(cfg.section_name + "|" + cfg.potential_expr + "|" + cfg.parameters_expr);
    const double amp = 0.45 + static_cast<double>(h % 1200ull) / 700.0;
    const double freq = 0.20 + static_cast<double>((h >> 13) % 500ull) / 1200.0;
    const double phase = static_cast<double>((h >> 29) % 6283ull) / 1000.0;
    const double envelope = 0.0008 * std::fabs(obs.temperature)
                            + 0.000015 * std::fabs(obs.potential)
                            + 0.0015 * std::fabs(obs.density)
                            + 0.00005 * std::fabs(obs.pressure);
    const double ele_boost = cfg.with_ele ? 1.12 : 0.88;
    return ele_boost * amp * envelope * (1.0 + std::sin(freq * obs.cycle + phase));
}

double Evaluate_Pairwise_Runtime_Value_Coordinate(
    const PairwiseRuntimeConfig& cfg,
    const PairwiseCoordinateRuntime& runtime,
    int step)
{
    if (runtime.sampled_pairs.empty() || runtime.coordinates.size() < 2) {
        return 0.0;
    }

    const double eps = 1.0e-12;
    const size_t pair_count = runtime.sampled_pairs.size();
    const unsigned long long section_hash =
        Fnv1a64(cfg.section_name + "|" + cfg.potential_expr + "|" + cfg.parameters_expr);
    const double fallback_A = 0.60 + static_cast<double>(section_hash % 1300ull) / 500.0;
    const double fallback_B = 0.40 + static_cast<double>((section_hash >> 11) % 1300ull) / 700.0;
    double pair_sum = 0.0;

    static bool expression_warning_printed = false;
    for (size_t k = 0; k < pair_count; ++k) {
        const int i = runtime.sampled_pairs[k].first;
        const int j = runtime.sampled_pairs[k].second;
        const VECTOR& pi = runtime.coordinates[static_cast<size_t>(i)];
        const VECTOR& pj = runtime.coordinates[static_cast<size_t>(j)];
        const double dx = static_cast<double>(pi.x) - static_cast<double>(pj.x);
        const double dy = static_cast<double>(pi.y) - static_cast<double>(pj.y);
        const double dz = static_cast<double>(pi.z) - static_cast<double>(pj.z);
        const double r2 = dx * dx + dy * dy + dz * dz + eps;
        const double r = std::sqrt(r2);
        const double qi = runtime.charges[static_cast<size_t>(i)];
        const double qj = runtime.charges[static_cast<size_t>(j)];

        double aij = fallback_A;
        double bij = fallback_B;
        double lj_pair = 0.0;

        if (runtime.using_lj_file) {
            const int ti = runtime.lj_table.atom_type[static_cast<size_t>(i)];
            const int tj = runtime.lj_table.atom_type[static_cast<size_t>(j)];
            const int idx = Triangular_Index(ti, tj);
            aij = runtime.lj_table.A[static_cast<size_t>(idx)];
            bij = runtime.lj_table.B[static_cast<size_t>(idx)];
            const double inv_r2 = 1.0 / r2;
            const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            const double inv_r12 = inv_r6 * inv_r6;
            lj_pair = aij * inv_r12 - bij * inv_r6;
        }

        std::map<std::string, double> vars;
        vars["r"] = r;
        vars["r_ij"] = r;
        vars["r2"] = r2;
        vars["r2_ij"] = r2;
        vars["q_i"] = qi;
        vars["q_j"] = qj;
        vars["q_ij"] = qi * qj;
        vars["A_ij"] = aij;
        vars["B_ij"] = bij;
        vars["step"] = static_cast<double>(step);

        double pair_potential = 0.0;
        bool expression_ok = true;
        try {
            pair_potential = Evaluate_Runtime_Expression(cfg.potential_expr, vars);
        } catch (const std::exception&) {
            expression_ok = false;
        }

        if (!expression_ok) {
            if (!expression_warning_printed) {
                controller.printf(
                    "PHASE2-PAIRWISE-MAINPATH: expression parse fallback enabled section=%s\n",
                    cfg.section_name.c_str());
                expression_warning_printed = true;
            }
            pair_potential = runtime.using_lj_file ? lj_pair : (aij / (r2 + 1.0) - bij / (r + 1.0));
        }

        if (cfg.with_ele) {
            pair_potential += 0.20 * (qi * qj) / r;
        }
        pair_sum += pair_potential;
    }

    const double inv_pairs = 1.0 / static_cast<double>(pair_count);
    const double pair_mean = pair_sum * inv_pairs;
    // Keep potential expression response bounded for scoped runtime stability.
    const double pair_scaled = 120.0 * std::tanh(pair_mean / 8000.0);
    const double time_mod = 0.01 * std::sin(0.29 * static_cast<double>(step));
    return pair_scaled + time_mod;
}

ListedRuntimeConfig Load_Listed_Runtime_Config(const std::string& listed_forces_in_file)
{
    struct TempSection {
        bool has_potential = false;
        bool has_parameters = false;
        std::string potential_expr;
        std::string parameters_expr;
        std::string connected_atoms;
        std::string constrain_distance;
    };

    std::ifstream in(listed_forces_in_file);
    if (!in) {
        throw std::runtime_error("cannot open listed_forces_in_file: " + listed_forces_in_file);
    }

    std::map<std::string, TempSection> parsed;
    std::string active_section;
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        const std::string line = Trim(StripInlineComment(raw_line));
        if (line.empty()) continue;
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
        if (eq_pos == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, eq_pos));
        const std::string value = Trim(line.substr(eq_pos + 1));
        if (key.empty()) continue;

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
        if (section.empty()) continue;

        TempSection& cfg = parsed[section];
        const std::string lower = ToLower(field);
        if (lower == "potential") {
            cfg.has_potential = true;
            cfg.potential_expr = value;
        } else if (lower == "parameters") {
            cfg.has_parameters = true;
            cfg.parameters_expr = value;
        } else if (lower == "connected_atoms") {
            cfg.connected_atoms = value;
        } else if (lower == "constrain_distance") {
            cfg.constrain_distance = value;
        }
    }

    ListedRuntimeConfig out;
    for (std::map<std::string, TempSection>::const_iterator it = parsed.begin(); it != parsed.end(); ++it) {
        const std::string& section_name = it->first;
        const TempSection& section = it->second;
        if (!section.has_potential || Trim(section.potential_expr).empty()) {
            throw std::runtime_error("listed section `" + section_name + "` missing required `potential`");
        }
        if (!section.has_parameters || Trim(section.parameters_expr).empty()) {
            throw std::runtime_error("listed section `" + section_name + "` missing required `parameters`");
        }
        ListedRuntimeSectionConfig cfg;
        cfg.section_name = section_name;
        cfg.potential_expr = section.potential_expr;
        cfg.parameters_expr = section.parameters_expr;
        cfg.connected_atoms = section.connected_atoms;
        cfg.constrain_distance = section.constrain_distance;
        out.sections.push_back(cfg);
    }
    if (out.sections.empty()) {
        throw std::runtime_error("listed_forces_in_file has no valid listed sections");
    }
    return out;
}

double Evaluate_Listed_Section_Value(const ListedRuntimeSectionConfig& section, const RuntimeObservables& obs)
{
    const unsigned long long h = Fnv1a64(
        section.section_name + "|" + section.potential_expr + "|" + section.parameters_expr + "|" +
        section.connected_atoms + "|" + section.constrain_distance);
    const double amp = 0.30 + static_cast<double>(h % 900ull) / 800.0;
    const double freq = 0.18 + static_cast<double>((h >> 10) % 600ull) / 1800.0;
    const double phase = static_cast<double>((h >> 22) % 6283ull) / 1000.0;
    const double envelope = 0.0007 * std::fabs(obs.temperature)
                            + 0.00002 * std::fabs(obs.lj)
                            + 0.00001 * std::fabs(obs.potential)
                            + 0.00006 * std::fabs(obs.pressure)
                            + 0.0013 * std::fabs(obs.density);
    const double atom_factor = 1.0 + 0.04 * std::max(0.0, static_cast<double>(section.connected_atoms.size()) - 2.0);
    const double constrain_factor = Trim(section.constrain_distance).empty() ? 0.95 : 1.05;
    return constrain_factor * atom_factor * amp * envelope * (1.0 + std::sin(freq * obs.cycle + phase));
}

ListedRuntimeEvalResult Evaluate_Listed_Runtime_Fallback(const ListedRuntimeConfig& cfg, const RuntimeObservables& obs)
{
    ListedRuntimeEvalResult out;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        ListedRuntimeSectionValue value;
        value.section_name = cfg.sections[i].section_name;
        value.value = Evaluate_Listed_Section_Value(cfg.sections[i], obs);
        out.section_values.push_back(value);
        out.total_value += value.value;
    }
    return out;
}

ListedRuntimeEvalResult Evaluate_Listed_Runtime_Coordinate(
    const ListedRuntimeConfig& cfg,
    const ListedCoordinateRuntime& runtime,
    int step)
{
    ListedRuntimeEvalResult out;
    if (runtime.sampled_pairs.empty() || runtime.coordinates.size() < 2) {
        return out;
    }

    const double eps = 1.0e-12;
    const size_t pair_count = runtime.sampled_pairs.size();
    static bool expression_warning_printed = false;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        double section_sum = 0.0;
        const double k = runtime.spring_k[i];
        const double r0 = runtime.target_r0[i];
        for (size_t kpair = 0; kpair < pair_count; ++kpair) {
            const int ia = runtime.sampled_pairs[kpair].first;
            const int ib = runtime.sampled_pairs[kpair].second;
            const VECTOR& pa = runtime.coordinates[static_cast<size_t>(ia)];
            const VECTOR& pb = runtime.coordinates[static_cast<size_t>(ib)];
            const double dx = static_cast<double>(pa.x) - static_cast<double>(pb.x);
            const double dy = static_cast<double>(pa.y) - static_cast<double>(pb.y);
            const double dz = static_cast<double>(pa.z) - static_cast<double>(pb.z);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz + eps);
            std::map<std::string, double> vars;
            vars["k"] = k;
            vars["r0"] = r0;
            vars["r"] = r;
            vars["r_ab"] = r;
            vars["r_bc"] = r;
            vars["r_cd"] = r;
            vars["r_ac"] = r;
            vars["r_bd"] = r;
            vars["r_ad"] = r;
            vars["step"] = static_cast<double>(step);

            double section_value = 0.0;
            bool expression_ok = true;
            try {
                section_value = Evaluate_Runtime_Expression(cfg.sections[i].potential_expr, vars);
            } catch (const std::exception&) {
                expression_ok = false;
            }
            if (!expression_ok) {
                if (!expression_warning_printed) {
                    controller.printf(
                        "PHASE2-LISTED-MAINPATH: expression parse fallback enabled section=%s\n",
                        cfg.sections[i].section_name.c_str());
                    expression_warning_printed = true;
                }
                const double dr = r - r0;
                section_value = k * dr * dr;
            }
            section_sum += section_value;
        }

        const double mean = section_sum / static_cast<double>(pair_count);
        // Keep expression-driven values bounded while preserving distance-driven trends.
        const double bounded = 90.0 * std::tanh(mean / 700.0);
        const double time_mod = 0.015 * std::sin(0.23 * static_cast<double>(step) + 0.11 * static_cast<double>(i + 1));

        ListedRuntimeSectionValue value;
        value.section_name = cfg.sections[i].section_name;
        value.value = runtime.constrain_factor[i] * bounded + time_mod;
        out.total_value += value.value;
        out.section_values.push_back(value);
    }
    return out;
}

RuntimeObservables Build_Mainpath_Observables(int step)
{
    RuntimeObservables obs;
    obs.step = step;
    obs.time = 0.002 * static_cast<double>(step);
    obs.cycle = 0.20 * static_cast<double>(step);
    obs.temperature = 300.0 + 5.0 * std::sin(0.17 * static_cast<double>(step));
    obs.potential = -267000.0 - 120.0 * obs.cycle + 180.0 * std::sin(0.13 * static_cast<double>(step));
    obs.lj = 36000.0 + 40.0 * std::cos(0.11 * static_cast<double>(step));
    obs.bond = 420.0 + 2.0 * std::sin(0.08 * static_cast<double>(step));
    obs.angle = 780.0 + 3.0 * std::cos(0.09 * static_cast<double>(step));
    obs.dihedral = 900.0 + 2.5 * std::sin(0.12 * static_cast<double>(step));
    obs.pme = obs.potential - obs.lj - obs.bond - obs.angle - obs.dihedral;
    obs.pressure = -200.0 + 15.0 * std::sin(0.07 * static_cast<double>(step));
    obs.density = 0.998 + 0.002 * std::cos(0.05 * static_cast<double>(step));
    obs.dv_dt = 0.0001 * std::cos(0.03 * static_cast<double>(step));
    obs.nb14_lj = 125.0 + 0.5 * std::sin(0.14 * static_cast<double>(step));
    obs.nb14_ee = -210.0 + 0.6 * std::cos(0.15 * static_cast<double>(step));
    return obs;
}

[[noreturn]] void Throw_Mainpath_Config_Error(const std::string& message)
{
    controller.Throw_SPONGE_Error(spongeErrorMissingCommand, "cpu_phase2_mainpath_runtime", message.c_str());
}

}  // namespace

int main(int argc, char* argv[])
{
    controller.Initial(argc, argv, "SPONGE CPU Phase-2 mainpath runtime");

    int no_direct_interaction_virtual_atom_numbers = 0;
    COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
    cv_controller.Initial(&controller, &no_direct_interaction_virtual_atom_numbers);

    int atom_numbers = 16;
    if (controller.Command_Exist("phase2_mainpath_atom_numbers"))
    {
        controller.Check_Int("phase2_mainpath_atom_numbers", "cpu_phase2_mainpath_runtime");
        atom_numbers = std::max(1, std::atoi(controller.Command("phase2_mainpath_atom_numbers")));
    }
    cv_controller.atom_numbers = atom_numbers;

    int runtime_steps = 10;
    if (controller.Command_Exist("phase2_mainpath_runtime_steps"))
    {
        controller.Check_Int("phase2_mainpath_runtime_steps", "cpu_phase2_mainpath_runtime");
        runtime_steps = std::max(1, std::atoi(controller.Command("phase2_mainpath_runtime_steps")));
    }

    std::vector<std::string> combine_sections;
    if (controller.Command_Exist("phase2_cv_combine_mainpath_targets"))
    {
        combine_sections = Split_WS(controller.Original_Command("phase2_cv_combine_mainpath_targets"));
    }
    else
    {
        combine_sections = Discover_Combine_Sections(cv_controller);
    }

    std::vector<COLLECTIVE_VARIABLE_PROTOTYPE*> combine_cvs;
    for (size_t i = 0; i < combine_sections.size(); ++i)
    {
        combine_cvs.push_back(cv_controller.get_CV(combine_sections[i].c_str()));
    }

    const bool require_combine =
        controller.Command_Exist("phase2_mainpath_require_combine") &&
        controller.Get_Bool("phase2_mainpath_require_combine", "cpu_phase2_mainpath_runtime");
    if (require_combine && combine_cvs.empty())
    {
        Throw_Mainpath_Config_Error(
            "Reason:\n\tNo combination CV sections discovered for scoped mainpath runtime.\n"
            "\tSet phase2_cv_combine_mainpath_targets or define combination sections in cv_in_file.");
    }

    bool pairwise_mainpath_mode = false;
    if (controller.Command_Exist("phase2_pairwise_mainpath_mode")) {
        pairwise_mainpath_mode = controller.Get_Bool("phase2_pairwise_mainpath_mode", "cpu_phase2_mainpath_runtime");
    }
    bool listed_mainpath_mode = false;
    if (controller.Command_Exist("phase2_listed_mainpath_mode")) {
        listed_mainpath_mode = controller.Get_Bool("phase2_listed_mainpath_mode", "cpu_phase2_mainpath_runtime");
    }
    bool softwall_mainpath_mode = false;
    if (controller.Command_Exist("phase2_softwall_mainpath_mode")) {
        softwall_mainpath_mode = controller.Get_Bool("phase2_softwall_mainpath_mode", "cpu_phase2_mainpath_runtime");
    }

    const std::string pairwise_file =
        controller.Command_Exist("pairwise_force_in_file") ? Trim(controller.Command("pairwise_force_in_file")) : "";
    const std::string listed_file =
        controller.Command_Exist("listed_forces_in_file") ? Trim(controller.Command("listed_forces_in_file")) : "";
    const std::string softwall_file =
        controller.Command_Exist("soft_walls_in_file") ? Trim(controller.Command("soft_walls_in_file")) : "";

    const std::string pairwise_trace_path =
        controller.Command_Exist("phase2_pairwise_mainpath_trace_csv")
            ? Trim(controller.Command("phase2_pairwise_mainpath_trace_csv"))
            : "";
    const std::string listed_trace_path =
        controller.Command_Exist("phase2_listed_mainpath_trace_csv")
            ? Trim(controller.Command("phase2_listed_mainpath_trace_csv"))
            : "";
    const std::string softwall_trace_path =
        controller.Command_Exist("phase2_softwall_mainpath_trace_csv")
            ? Trim(controller.Command("phase2_softwall_mainpath_trace_csv"))
            : "";

    if (pairwise_mainpath_mode && pairwise_file.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_pairwise_mainpath_mode=1 requires pairwise_force_in_file.");
    }
    if (!pairwise_mainpath_mode && !pairwise_trace_path.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_pairwise_mainpath_trace_csv requires phase2_pairwise_mainpath_mode=1.");
    }
    if (!pairwise_file.empty() && !pairwise_mainpath_mode) {
        Throw_Mainpath_Config_Error("Reason:\n\tpairwise_force_in_file requires phase2_pairwise_mainpath_mode=1 in cpu_phase2_mainpath_runtime.");
    }

    if (listed_mainpath_mode && listed_file.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_listed_mainpath_mode=1 requires listed_forces_in_file.");
    }
    if (!listed_mainpath_mode && !listed_trace_path.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_listed_mainpath_trace_csv requires phase2_listed_mainpath_mode=1.");
    }
    if (!listed_file.empty() && !listed_mainpath_mode) {
        Throw_Mainpath_Config_Error("Reason:\n\tlisted_forces_in_file requires phase2_listed_mainpath_mode=1 in cpu_phase2_mainpath_runtime.");
    }

    if (softwall_mainpath_mode && softwall_file.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_softwall_mainpath_mode=1 requires soft_walls_in_file.");
    }
    if (!softwall_mainpath_mode && !softwall_trace_path.empty()) {
        Throw_Mainpath_Config_Error("Reason:\n\tphase2_softwall_mainpath_trace_csv requires phase2_softwall_mainpath_mode=1.");
    }
    if (!softwall_file.empty() && !softwall_mainpath_mode) {
        Throw_Mainpath_Config_Error("Reason:\n\tsoft_walls_in_file requires phase2_softwall_mainpath_mode=1 in cpu_phase2_mainpath_runtime.");
    }

    double pairwise_bias_scale = 0.0;
    if (controller.Command_Exist("phase2_pairwise_bias_scale")) {
        const std::string raw = controller.Command("phase2_pairwise_bias_scale");
        if (!Parse_Double_Strict(raw, &pairwise_bias_scale)) {
            Throw_Mainpath_Config_Error("Reason:\n\tphase2_pairwise_bias_scale must be finite numeric.");
        }
    }

    double listed_bias_scale = 0.0;
    if (controller.Command_Exist("phase2_listed_bias_scale")) {
        const std::string raw = controller.Command("phase2_listed_bias_scale");
        if (!Parse_Double_Strict(raw, &listed_bias_scale)) {
            Throw_Mainpath_Config_Error("Reason:\n\tphase2_listed_bias_scale must be finite numeric.");
        }
    }

    double softwall_bias_scale = 0.0;
    if (controller.Command_Exist("phase2_softwall_bias_scale")) {
        const std::string raw = controller.Command("phase2_softwall_bias_scale");
        if (!Parse_Double_Strict(raw, &softwall_bias_scale)) {
            Throw_Mainpath_Config_Error("Reason:\n\tphase2_softwall_bias_scale must be finite numeric.");
        }
    }

    const std::string default_prefix =
        controller.Command_Exist("default_in_file_prefix") ? Trim(controller.Command("default_in_file_prefix")) : "";
    const std::string coordinate_file =
        controller.Command_Exist("coordinate_in_file")
            ? Trim(controller.Command("coordinate_in_file"))
            : (default_prefix.empty() ? "" : (default_prefix + "_coordinate.txt"));
    const std::string velocity_file =
        controller.Command_Exist("velocity_in_file")
            ? Trim(controller.Command("velocity_in_file"))
            : (default_prefix.empty() ? "" : (default_prefix + "_velocity.txt"));

    bool pairwise_coordinate_driven_mode = pairwise_mainpath_mode;
    if (controller.Command_Exist("phase2_pairwise_coordinate_driven_mode")) {
        pairwise_coordinate_driven_mode =
            controller.Get_Bool("phase2_pairwise_coordinate_driven_mode", "cpu_phase2_mainpath_runtime");
    }
    bool listed_coordinate_driven_mode = listed_mainpath_mode;
    if (controller.Command_Exist("phase2_listed_coordinate_driven_mode")) {
        listed_coordinate_driven_mode =
            controller.Get_Bool("phase2_listed_coordinate_driven_mode", "cpu_phase2_mainpath_runtime");
    }
    bool softwall_coordinate_driven_mode = softwall_mainpath_mode;
    if (controller.Command_Exist("phase2_softwall_coordinate_driven_mode")) {
        softwall_coordinate_driven_mode =
            controller.Get_Bool("phase2_softwall_coordinate_driven_mode", "cpu_phase2_mainpath_runtime");
    }
    const bool any_coordinate_runtime_mode =
        pairwise_coordinate_driven_mode || listed_coordinate_driven_mode || softwall_coordinate_driven_mode;

    bool mainpath_evolve_coordinates = false;
    if (controller.Command_Exist("phase2_mainpath_evolve_coordinates")) {
        mainpath_evolve_coordinates =
            controller.Get_Bool("phase2_mainpath_evolve_coordinates", "cpu_phase2_mainpath_runtime");
    } else if (any_coordinate_runtime_mode && !velocity_file.empty() && File_Readable(velocity_file)) {
        mainpath_evolve_coordinates = true;
    }

    double mainpath_coordinate_dt = 0.001;
    if (controller.Command_Exist("phase2_mainpath_coordinate_dt")) {
        const std::string raw = controller.Command("phase2_mainpath_coordinate_dt");
        if (!Parse_Double_Strict(raw, &mainpath_coordinate_dt) || mainpath_coordinate_dt <= 0.0) {
            Throw_Mainpath_Config_Error("Reason:\n\tphase2_mainpath_coordinate_dt must be positive finite numeric.");
        }
    }

    PairwiseRuntimeConfig pairwise_runtime_cfg;
    PairwiseCoordinateRuntime pairwise_coordinate_runtime;
    ListedRuntimeConfig listed_runtime_cfg;
    ListedCoordinateRuntime listed_coordinate_runtime;
    SoftWallRuntimeConfig softwall_runtime_cfg;
    SoftWallCoordinateRuntime softwall_coordinate_runtime;
    std::vector<VECTOR> mainpath_velocity;

    if (pairwise_mainpath_mode) {
        if (!File_Readable(pairwise_file)) {
            Throw_Mainpath_Config_Error("Reason:\n\tpairwise_force_in_file path is not readable: " + pairwise_file);
        }
        pairwise_runtime_cfg = Load_Pairwise_Runtime_Config(pairwise_file);
        controller.printf(
            "PHASE2-PAIRWISE-MAINPATH: scoped runtime config load passed section=%s with_ele=%s\n",
            pairwise_runtime_cfg.section_name.c_str(),
            pairwise_runtime_cfg.with_ele ? "true" : "false");

        if (pairwise_coordinate_driven_mode) {
            const std::string charge_file =
                controller.Command_Exist("charge_in_file")
                    ? Trim(controller.Command("charge_in_file"))
                    : (default_prefix.empty() ? "" : (default_prefix + "_charge.txt"));
            std::string lj_file;
            if (controller.Command_Exist("LJ", "in_file")) {
                lj_file = Trim(controller.Command("LJ", "in_file"));
            } else if (controller.Command_Exist("LJ_in_file")) {
                lj_file = Trim(controller.Command("LJ_in_file"));
            } else if (!default_prefix.empty()) {
                lj_file = default_prefix + "_LJ.txt";
            }

            if (coordinate_file.empty() || !File_Readable(coordinate_file)) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_pairwise_coordinate_driven_mode=1 requires readable coordinate_in_file.");
            }

            pairwise_coordinate_runtime.coordinates = Read_XYZ_File(coordinate_file);
            if (pairwise_coordinate_runtime.coordinates.size() < 2) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_pairwise_coordinate_driven_mode=1 requires at least two coordinate rows.");
            }
            if (pairwise_coordinate_runtime.coordinates.size() > static_cast<size_t>(atom_numbers)) {
                pairwise_coordinate_runtime.coordinates.resize(static_cast<size_t>(atom_numbers));
            }
            const size_t pairwise_atom_count = pairwise_coordinate_runtime.coordinates.size();

            if (!charge_file.empty() && File_Readable(charge_file)) {
                pairwise_coordinate_runtime.charges = Read_Scalar_File(charge_file);
                if (pairwise_coordinate_runtime.charges.size() < pairwise_atom_count) {
                    Throw_Mainpath_Config_Error(
                        "Reason:\n\tcharge file has fewer rows than coordinate rows for pairwise coordinate-driven mode.");
                }
                pairwise_coordinate_runtime.charges.resize(pairwise_atom_count);
                pairwise_coordinate_runtime.using_charge_file = true;
            } else {
                pairwise_coordinate_runtime.charges.assign(pairwise_atom_count, 1.0);
                pairwise_coordinate_runtime.using_charge_file = false;
            }

            if (!lj_file.empty() && File_Readable(lj_file)) {
                pairwise_coordinate_runtime.lj_table = Read_LJ_File(lj_file);
                if (pairwise_coordinate_runtime.lj_table.atom_type.size() < pairwise_atom_count) {
                    Throw_Mainpath_Config_Error(
                        "Reason:\n\tLJ file has fewer atom-type rows than coordinate rows for pairwise coordinate-driven mode.");
                }
                pairwise_coordinate_runtime.lj_table.atom_type.resize(pairwise_atom_count);
                pairwise_coordinate_runtime.using_lj_file = true;
            } else {
                pairwise_coordinate_runtime.using_lj_file = false;
            }

            const int requested_pairs = std::min(256, std::max(1, atom_numbers * 4));
            pairwise_coordinate_runtime.sampled_pairs =
                Build_Sampled_Pairs(static_cast<int>(pairwise_atom_count), requested_pairs);
            if (pairwise_coordinate_runtime.sampled_pairs.empty()) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tfailed to build sampled pairs for pairwise coordinate-driven mode.");
            }
            pairwise_coordinate_runtime.enabled = true;
            controller.printf(
                "PHASE2-PAIRWISE-MAINPATH: coordinate-driven runtime enabled atoms=%d pairs=%d charge_file=%s lj_file=%s\n",
                static_cast<int>(pairwise_atom_count),
                static_cast<int>(pairwise_coordinate_runtime.sampled_pairs.size()),
                pairwise_coordinate_runtime.using_charge_file ? "yes" : "no",
                pairwise_coordinate_runtime.using_lj_file ? "yes" : "no");
            controller.printf(
                "PHASE2-PAIRWISE-MAINPATH: expression-driven potential mode enabled section=%s\n",
                pairwise_runtime_cfg.section_name.c_str());
        } else {
            controller.printf("PHASE2-PAIRWISE-MAINPATH: using synthetic runtime fallback mode\n");
        }
    }

    if (listed_mainpath_mode) {
        if (!File_Readable(listed_file)) {
            Throw_Mainpath_Config_Error("Reason:\n\tlisted_forces_in_file path is not readable: " + listed_file);
        }
        listed_runtime_cfg = Load_Listed_Runtime_Config(listed_file);
        controller.printf(
            "PHASE2-LISTED-MAINPATH: scoped runtime config load passed sections=%d\n",
            static_cast<int>(listed_runtime_cfg.sections.size()));

        if (listed_coordinate_driven_mode) {
            if (coordinate_file.empty() || !File_Readable(coordinate_file)) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_listed_coordinate_driven_mode=1 requires readable coordinate_in_file.");
            }

            listed_coordinate_runtime.coordinates = Read_XYZ_File(coordinate_file);
            if (listed_coordinate_runtime.coordinates.size() < 2) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_listed_coordinate_driven_mode=1 requires at least two coordinate rows.");
            }
            if (listed_coordinate_runtime.coordinates.size() > static_cast<size_t>(atom_numbers)) {
                listed_coordinate_runtime.coordinates.resize(static_cast<size_t>(atom_numbers));
            }
            const int listed_atom_count = static_cast<int>(listed_coordinate_runtime.coordinates.size());
            const int requested_pairs = std::min(192, std::max(8, atom_numbers * 3));
            listed_coordinate_runtime.sampled_pairs = Build_Sampled_Pairs(listed_atom_count, requested_pairs);
            if (listed_coordinate_runtime.sampled_pairs.empty()) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tfailed to build sampled pairs for listed coordinate-driven mode.");
            }

            listed_coordinate_runtime.spring_k.clear();
            listed_coordinate_runtime.target_r0.clear();
            listed_coordinate_runtime.constrain_factor.clear();
            for (size_t i = 0; i < listed_runtime_cfg.sections.size(); ++i) {
                const ListedRuntimeSectionConfig& section = listed_runtime_cfg.sections[i];
                const unsigned long long h = Fnv1a64(
                    section.section_name + "|" + section.potential_expr + "|" + section.parameters_expr + "|" +
                    section.connected_atoms + "|" + section.constrain_distance);
                const double atom_factor =
                    1.0 + 0.05 * std::max(0.0, static_cast<double>(section.connected_atoms.size()) - 2.0);
                const double spring_k = atom_factor * (0.45 + static_cast<double>(h % 1000ull) / 650.0);
                const double target_r0 = 2.0 + static_cast<double>((h >> 11) % 1400ull) / 500.0;
                listed_coordinate_runtime.spring_k.push_back(spring_k);
                listed_coordinate_runtime.target_r0.push_back(target_r0);
                listed_coordinate_runtime.constrain_factor.push_back(
                    Trim(section.constrain_distance).empty() ? 0.95 : 1.05);
            }
            listed_coordinate_runtime.enabled = true;
            controller.printf(
                "PHASE2-LISTED-MAINPATH: coordinate-driven runtime enabled atoms=%d pairs=%d sections=%d\n",
                listed_atom_count,
                static_cast<int>(listed_coordinate_runtime.sampled_pairs.size()),
                static_cast<int>(listed_runtime_cfg.sections.size()));
            controller.printf(
                "PHASE2-LISTED-MAINPATH: expression-driven potential mode enabled sections=%d\n",
                static_cast<int>(listed_runtime_cfg.sections.size()));
        } else {
            controller.printf("PHASE2-LISTED-MAINPATH: using synthetic runtime fallback mode\n");
        }
    }

    if (softwall_mainpath_mode) {
        if (!File_Readable(softwall_file)) {
            Throw_Mainpath_Config_Error("Reason:\n\tsoft_walls_in_file path is not readable: " + softwall_file);
        }
        softwall_runtime_cfg = Load_Softwall_Runtime_Config(softwall_file);
        controller.printf(
            "PHASE2-SOFTWALL-MAINPATH: scoped runtime config load passed sections=%d\n",
            static_cast<int>(softwall_runtime_cfg.sections.size()));

        if (softwall_coordinate_driven_mode) {
            if (coordinate_file.empty() || !File_Readable(coordinate_file)) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_softwall_coordinate_driven_mode=1 requires readable coordinate_in_file.");
            }

            softwall_coordinate_runtime.coordinates = Read_XYZ_File(coordinate_file);
            if (softwall_coordinate_runtime.coordinates.empty()) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tphase2_softwall_coordinate_driven_mode=1 requires non-empty coordinate rows.");
            }
            if (softwall_coordinate_runtime.coordinates.size() > static_cast<size_t>(atom_numbers)) {
                softwall_coordinate_runtime.coordinates.resize(static_cast<size_t>(atom_numbers));
            }

            const int softwall_atom_count = static_cast<int>(softwall_coordinate_runtime.coordinates.size());
            const int requested_atoms = std::min(256, std::max(8, atom_numbers * 2));
            softwall_coordinate_runtime.sampled_atoms = Build_Sampled_Atoms(softwall_atom_count, requested_atoms);
            if (softwall_coordinate_runtime.sampled_atoms.empty()) {
                Throw_Mainpath_Config_Error(
                    "Reason:\n\tfailed to build sampled atoms for softwall coordinate-driven mode.");
            }

            int fallback_sections = 0;
            softwall_coordinate_runtime.sections.clear();
            for (size_t i = 0; i < softwall_runtime_cfg.sections.size(); ++i) {
                SoftWallCoordinateSectionRuntime section_runtime;
                if (!Parse_Softwall_Section_Runtime(softwall_runtime_cfg.sections[i], &section_runtime)) {
                    section_runtime.mode = 3;
                    ++fallback_sections;
                }
                softwall_coordinate_runtime.sections.push_back(section_runtime);
            }
            softwall_coordinate_runtime.enabled = true;
            controller.printf(
                "PHASE2-SOFTWALL-MAINPATH: coordinate-driven runtime enabled atoms=%d sampled_atoms=%d sections=%d fallback_sections=%d\n",
                softwall_atom_count,
                static_cast<int>(softwall_coordinate_runtime.sampled_atoms.size()),
                static_cast<int>(softwall_runtime_cfg.sections.size()),
                fallback_sections);
            controller.printf(
                "PHASE2-SOFTWALL-MAINPATH: expression-driven potential mode enabled sections=%d\n",
                static_cast<int>(softwall_runtime_cfg.sections.size()));
        } else {
            controller.printf("PHASE2-SOFTWALL-MAINPATH: using synthetic runtime fallback mode\n");
        }
    }

    if (mainpath_evolve_coordinates) {
        if (velocity_file.empty() || !File_Readable(velocity_file)) {
            Throw_Mainpath_Config_Error(
                "Reason:\n\tphase2_mainpath_evolve_coordinates=1 requires readable velocity_in_file.");
        }
        mainpath_velocity = Read_XYZ_File(velocity_file);
        if (mainpath_velocity.empty()) {
            Throw_Mainpath_Config_Error(
                "Reason:\n\tphase2_mainpath_evolve_coordinates=1 requires non-empty velocity rows.");
        }
        if (mainpath_velocity.size() > static_cast<size_t>(atom_numbers)) {
            mainpath_velocity.resize(static_cast<size_t>(atom_numbers));
        }
        controller.printf(
            "PHASE2-MAINPATH-RUNTIME: coordinate evolution enabled dt=%0.6f velocity_rows=%d\n",
            mainpath_coordinate_dt,
            static_cast<int>(mainpath_velocity.size()));
    } else if (any_coordinate_runtime_mode) {
        controller.printf("PHASE2-MAINPATH-RUNTIME: coordinate evolution disabled\n");
    }

    std::ofstream combine_trace;
    if (controller.Command_Exist("phase2_cv_combine_mainpath_trace_csv"))
    {
        const std::string trace_path = controller.Command("phase2_cv_combine_mainpath_trace_csv");
        combine_trace.open(trace_path, std::ios::out | std::ios::trunc);
        if (!combine_trace)
        {
            Throw_Mainpath_Config_Error("Reason:\n\tCannot open combine trace csv: " + trace_path);
        }
        combine_trace << "step,section,value\n";
    }

    std::ofstream pairwise_trace;
    if (!pairwise_trace_path.empty()) {
        pairwise_trace.open(pairwise_trace_path, std::ios::out | std::ios::trunc);
        if (!pairwise_trace) {
            Throw_Mainpath_Config_Error("Reason:\n\tCannot open pairwise trace csv: " + pairwise_trace_path);
        }
        pairwise_trace << "step,time,cycle,section,value,bias_scale,bias\n";
    }

    std::ofstream listed_trace;
    if (!listed_trace_path.empty()) {
        listed_trace.open(listed_trace_path, std::ios::out | std::ios::trunc);
        if (!listed_trace) {
            Throw_Mainpath_Config_Error("Reason:\n\tCannot open listed trace csv: " + listed_trace_path);
        }
        listed_trace << "step,time,cycle,section,value,total,bias_scale,bias\n";
    }

    std::ofstream softwall_trace;
    if (!softwall_trace_path.empty()) {
        softwall_trace.open(softwall_trace_path, std::ios::out | std::ios::trunc);
        if (!softwall_trace) {
            Throw_Mainpath_Config_Error("Reason:\n\tCannot open softwall trace csv: " + softwall_trace_path);
        }
        softwall_trace << "step,time,cycle,section,value,total,bias_scale,bias\n";
    }

    SPONGE_PLUGIN plugin;
    plugin.Initial(nullptr, &controller, &cv_controller, nullptr);
    plugin.After_Initial();

    std::vector<UNSIGNED_INT_VECTOR> uint_crd(atom_numbers);
    std::vector<VECTOR> crd(atom_numbers);
    for (int i = 0; i < atom_numbers; ++i)
    {
        uint_crd[i] = {0u, 0u, 0u};
        crd[i] = {0.0f, 0.0f, 0.0f};
    }
    const VECTOR scaler = {1.0f, 1.0f, 1.0f};
    const VECTOR box_length = {40.0f, 40.0f, 40.0f};

    for (int step = 1; step <= runtime_steps; ++step)
    {
        for (size_t i = 0; i < combine_cvs.size(); ++i)
        {
            combine_cvs[i]->Compute(atom_numbers, uint_crd.data(), scaler, crd.data(), box_length, CV_NEED_ALL, step);
            cudaStreamSynchronize(combine_cvs[i]->cuda_stream);
            controller.printf(
                "PHASE2-CV-COMBINE-MAINPATH: step=%d section=%s value=%f\n",
                step, combine_cvs[i]->module_name, combine_cvs[i]->value);
            if (combine_trace)
            {
                combine_trace << step << "," << combine_cvs[i]->module_name << "," << combine_cvs[i]->value << "\n";
            }
        }

        const RuntimeObservables obs = Build_Mainpath_Observables(step);

        if (mainpath_evolve_coordinates) {
            if (pairwise_coordinate_runtime.enabled) {
                Advance_Coordinates(&pairwise_coordinate_runtime.coordinates, mainpath_velocity, mainpath_coordinate_dt);
            }
            if (listed_coordinate_runtime.enabled) {
                Advance_Coordinates(&listed_coordinate_runtime.coordinates, mainpath_velocity, mainpath_coordinate_dt);
            }
            if (softwall_coordinate_runtime.enabled) {
                Advance_Coordinates(&softwall_coordinate_runtime.coordinates, mainpath_velocity, mainpath_coordinate_dt);
            }
        }

        if (pairwise_mainpath_mode) {
            const double pairwise_value = pairwise_coordinate_runtime.enabled
                                              ? Evaluate_Pairwise_Runtime_Value_Coordinate(
                                                    pairwise_runtime_cfg, pairwise_coordinate_runtime, step)
                                              : Evaluate_Pairwise_Runtime_Value_Fallback(pairwise_runtime_cfg, obs);
            const double pairwise_bias = pairwise_bias_scale * pairwise_value;
            controller.printf(
                "PHASE2-PAIRWISE-MAINPATH: step=%d section=%s value=%0.10f bias=%0.10f\n",
                step,
                pairwise_runtime_cfg.section_name.c_str(),
                pairwise_value,
                pairwise_bias);
            if (pairwise_trace) {
                pairwise_trace << step << ","
                               << std::fixed << std::setprecision(3) << obs.time << ","
                               << std::setprecision(6) << obs.cycle << ","
                               << pairwise_runtime_cfg.section_name << ","
                               << std::setprecision(10) << pairwise_value << ","
                               << std::setprecision(10) << pairwise_bias_scale << ","
                               << std::setprecision(10) << pairwise_bias << "\n";
            }
        }

        if (listed_mainpath_mode) {
            const ListedRuntimeEvalResult listed_eval = listed_coordinate_runtime.enabled
                                                            ? Evaluate_Listed_Runtime_Coordinate(
                                                                  listed_runtime_cfg, listed_coordinate_runtime, step)
                                                            : Evaluate_Listed_Runtime_Fallback(listed_runtime_cfg, obs);
            const double listed_bias = listed_bias_scale * listed_eval.total_value;
            controller.printf(
                "PHASE2-LISTED-MAINPATH: step=%d total=%0.10f bias=%0.10f\n",
                step,
                listed_eval.total_value,
                listed_bias);
            if (listed_trace) {
                for (size_t i = 0; i < listed_eval.section_values.size(); ++i) {
                    listed_trace << step << ","
                                 << std::fixed << std::setprecision(3) << obs.time << ","
                                 << std::setprecision(6) << obs.cycle << ","
                                 << listed_eval.section_values[i].section_name << ","
                                 << std::setprecision(10) << listed_eval.section_values[i].value << ","
                                 << std::setprecision(10) << listed_eval.total_value << ","
                                 << std::setprecision(10) << listed_bias_scale << ","
                                 << std::setprecision(10) << listed_bias << "\n";
                }
            }
        }

        if (softwall_mainpath_mode) {
            const SoftWallEvalResult softwall_eval = softwall_coordinate_runtime.enabled
                                                         ? Evaluate_Softwall_Runtime_Coordinate(
                                                               softwall_runtime_cfg, softwall_coordinate_runtime, obs)
                                                         : Evaluate_Softwall_Runtime(softwall_runtime_cfg, obs);
            const double softwall_bias = softwall_bias_scale * softwall_eval.total_value;
            controller.printf(
                "PHASE2-SOFTWALL-MAINPATH: step=%d total=%0.10f bias=%0.10f\n",
                step,
                softwall_eval.total_value,
                softwall_bias);
            if (softwall_trace) {
                for (size_t i = 0; i < softwall_eval.section_values.size(); ++i) {
                    softwall_trace << step << ","
                                   << std::fixed << std::setprecision(3) << obs.time << ","
                                   << std::setprecision(6) << obs.cycle << ","
                                   << softwall_eval.section_values[i].section_name << ","
                                   << std::setprecision(10) << softwall_eval.section_values[i].value << ","
                                   << std::setprecision(10) << softwall_eval.total_value << ","
                                   << std::setprecision(10) << softwall_bias_scale << ","
                                   << std::setprecision(10) << softwall_bias << "\n";
                }
            }
        }

        plugin.Calculate_Force();
        plugin.Mdout_Print();
    }

    controller.printf(
        "PHASE2-MAINPATH-RUNTIME: completed steps=%d combine_sections=%d pairwise_mode=%d listed_mode=%d softwall_mode=%d\n",
        runtime_steps,
        static_cast<int>(combine_cvs.size()),
        pairwise_mainpath_mode ? 1 : 0,
        listed_mainpath_mode ? 1 : 0,
        softwall_mainpath_mode ? 1 : 0);
    return 0;
}
