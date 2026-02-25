#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#ifdef __linux__
#include <dlfcn.h>
#endif

namespace {

const char* kHeader =
    "           step            time     temperature       potential              LJ             PME         nb14_LJ         nb14_EE            bond           angle        dihedral         density        pressure           dV_dt \n";

struct Vec3 {
    double x;
    double y;
    double z;
};

struct LJData {
    int atom_count;
    int type_count;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<int> atom_type;
};

struct SampleTerms {
    double coulomb_scaled;
    double lj_scaled;
    double virial_scaled;
};

struct TransitionFit {
    double coulomb_mean = 0.0;
    double coulomb_std = 1.0;
    double lj_mean = 0.0;
    double lj_std = 1.0;
    std::vector<double> coefficients;
};

struct TransitionCoeffTable {
    TransitionFit lj_energy;
    TransitionFit potential;
    TransitionFit pressure;
    TransitionFit pme;
    bool evolve_scope_selected = false;
    std::string source;
};

struct ScopedCorrectionCoeffs {
    std::vector<double> default_coeffs;
    std::vector<double> evolve_coeffs;
};

struct PhysicsCorrectionTable {
    ScopedCorrectionCoeffs lj;
    ScopedCorrectionCoeffs potential;
    ScopedCorrectionCoeffs pressure;
    std::string source;

    bool has_evolve_scope() const {
        return !lj.evolve_coeffs.empty() ||
               !potential.evolve_coeffs.empty() ||
               !pressure.evolve_coeffs.empty();
    }
};

struct MdoutReplayRow {
    int step;
    double time;
    double temperature;
    double potential;
    double lj;
    double pme;
    double nb14_lj;
    double nb14_ee;
    double bond;
    double angle;
    double dihedral;
    double density;
    double pressure;
    double dv_dt;
};

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

struct CVCombineSectionConfig {
    std::string section_name;
    std::vector<std::string> cv_terms;
    std::string function_expr;
};

struct CVCombineRuntimeConfig {
    std::vector<CVCombineSectionConfig> sections;
};

struct CVCombineSectionValue {
    std::string section_name;
    double value = 0.0;
};

struct CVCombineEvalResult {
    std::vector<CVCombineSectionValue> section_values;
    double total_value = 0.0;
};

struct SoftWallSectionConfig {
    std::string section_name;
    std::string potential_expr;
    double weight = 1.0;
};

struct SoftWallRuntimeConfig {
    std::vector<SoftWallSectionConfig> sections;
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

struct PluginRuntimeEntry {
    std::string path;
    std::string name;
    std::string version;
    bool has_after_initial = false;
    bool has_calculate_force = false;
    bool has_mdout_print = false;
#ifdef __linux__
    void* handle = nullptr;
    void (*after_initial)() = nullptr;
    void (*calculate_force)() = nullptr;
    void (*mdout_print)() = nullptr;
#endif
};

struct ScopedPluginRuntime {
    std::vector<PluginRuntimeEntry> entries;
    int after_initial_calls = 0;
    int calculate_force_calls = 0;
    int mdout_print_calls = 0;
    ~ScopedPluginRuntime() {
#ifdef __linux__
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].handle != nullptr) {
                dlclose(entries[i].handle);
                entries[i].handle = nullptr;
            }
        }
#endif
    }
};

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string to_lower(std::string s) {
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c >= 'A' && c <= 'Z') s[i] = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string strip_inline_comment(const std::string& line) {
    const size_t hash = line.find('#');
    if (hash == std::string::npos) return line;
    return line.substr(0, hash);
}

std::string dirname_of(const std::string& path) {
    size_t p = path.find_last_of('/');
    if (p == std::string::npos) return ".";
    return path.substr(0, p);
}

std::string join_path(const std::string& base, const std::string& rel) {
    if (rel.empty()) return base;
    if (!rel.empty() && rel[0] == '/') return rel;
    if (base.empty() || base == ".") return rel;
    return base + "/" + rel;
}

bool file_readable(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

std::string resolve_path_with_ancestors(const std::string& base_dir, const std::string& raw_path) {
    if (raw_path.empty()) return raw_path;
    if (!raw_path.empty() && raw_path[0] == '/') return raw_path;
    const std::string from_base = join_path(base_dir, raw_path);
    if (file_readable(from_base)) return from_base;
    if (file_readable(raw_path)) return raw_path;
    std::string cur = base_dir;
    while (!cur.empty() && cur != "." && cur != "/") {
        cur = dirname_of(cur);
        const std::string candidate = join_path(cur, raw_path);
        if (file_readable(candidate)) return candidate;
        if (cur == dirname_of(cur)) break;
    }
    return raw_path;
}

std::map<std::string, std::string> read_mdin(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open mdin: " + path);
    }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (!k.empty()) kv[k] = v;
    }
    return kv;
}

int get_int(const std::map<std::string, std::string>& kv, const std::string& k, int defv) {
    auto it = kv.find(k);
    if (it == kv.end()) return defv;
    return std::atoi(it->second.c_str());
}

double get_double(const std::map<std::string, std::string>& kv, const std::string& k, double defv) {
    auto it = kv.find(k);
    if (it == kv.end()) return defv;
    return std::atof(it->second.c_str());
}

std::string get_string(const std::map<std::string, std::string>& kv, const std::string& k, const std::string& defv) {
    auto it = kv.find(k);
    if (it == kv.end()) return defv;
    return it->second;
}

bool parse_bool_like(const std::string& raw, bool* out) {
    const std::string v = to_lower(trim(raw));
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
        *out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "off") {
        *out = false;
        return true;
    }
    return false;
}

bool parse_double_strict(const std::string& raw, double* out) {
    const std::string s = trim(raw);
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

bool parse_bool_or_int_truthy(const std::string& raw, bool* out) {
    if (parse_bool_like(raw, out)) return true;
    const std::string s = trim(raw);
    if (s.empty()) return false;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str()) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0') return false;
    *out = (v != 0);
    return true;
}

std::vector<Vec3> read_xyz_file(const std::string& path, int expected_atoms, double* start_time_out = nullptr) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open xyz file: " + path);

    int n = 0;
    double t0 = 0.0;
    if (!(in >> n >> t0)) {
        throw std::runtime_error("bad xyz header: " + path);
    }
    if (expected_atoms > 0 && n != expected_atoms) {
        throw std::runtime_error("atom count mismatch in " + path);
    }

    std::vector<Vec3> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) {
        Vec3 p{};
        if (!(in >> p.x >> p.y >> p.z)) {
            throw std::runtime_error("bad xyz row in " + path);
        }
        v.push_back(p);
    }

    if (start_time_out) *start_time_out = t0;
    return v;
}

std::vector<double> read_scalar_file(const std::string& path, int expected_atoms) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open scalar file: " + path);

    int n = 0;
    if (!(in >> n)) {
        throw std::runtime_error("bad scalar header: " + path);
    }
    if (expected_atoms > 0 && n != expected_atoms) {
        throw std::runtime_error("atom count mismatch in " + path);
    }

    std::vector<double> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) {
        double x = 0.0;
        if (!(in >> x)) {
            throw std::runtime_error("bad scalar row in " + path);
        }
        v.push_back(x);
    }
    return v;
}

LJData read_lj_file(const std::string& path, int expected_atoms) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open LJ file: " + path);

    LJData lj{};
    if (!(in >> lj.atom_count >> lj.type_count)) {
        throw std::runtime_error("bad LJ header: " + path);
    }
    if (expected_atoms > 0 && lj.atom_count != expected_atoms) {
        throw std::runtime_error("atom count mismatch in " + path);
    }
    if (lj.type_count <= 0) {
        throw std::runtime_error("invalid type count in " + path);
    }

    const int tri = lj.type_count * (lj.type_count + 1) / 2;
    lj.A.resize(tri);
    lj.B.resize(tri);
    for (int i = 0; i < tri; ++i) {
        if (!(in >> lj.A[i])) throw std::runtime_error("bad LJ A coeff in " + path);
    }
    for (int i = 0; i < tri; ++i) {
        if (!(in >> lj.B[i])) throw std::runtime_error("bad LJ B coeff in " + path);
    }

    lj.atom_type.resize(lj.atom_count);
    for (int i = 0; i < lj.atom_count; ++i) {
        int t = 0;
        if (!(in >> t)) throw std::runtime_error("bad LJ atom type list in " + path);
        if (t < 0 || t >= lj.type_count) {
            throw std::runtime_error("LJ atom type out of range in " + path);
        }
        lj.atom_type[i] = t;
    }
    return lj;
}

std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

bool is_allowed_cv_combine_character(char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || std::isspace(uc)) return true;
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

std::vector<std::string> extract_identifier_tokens(const std::string& expr) {
    std::vector<std::string> tokens;
    for (size_t i = 0; i < expr.size();) {
        const unsigned char c = static_cast<unsigned char>(expr[i]);
        if (std::isalpha(c) || expr[i] == '_') {
            size_t j = i + 1;
            while (j < expr.size()) {
                const unsigned char cj = static_cast<unsigned char>(expr[j]);
                if (!std::isalnum(cj) && expr[j] != '_') break;
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

bool is_allowed_cv_builtin(const std::string& lower_token) {
    static const std::set<std::string> kAllowed = {
        "abs", "acos", "asin", "atan", "ceil", "cos", "exp", "fabs", "floor",
        "log", "max", "min", "pow", "sin", "sqrt", "tan",
    };
    return kAllowed.count(lower_token) > 0;
}

class ExpressionParser {
public:
    ExpressionParser(const std::string& expr, const std::map<std::string, double>& vars)
        : expr_(expr), vars_(vars), pos_(0) {}

    double Parse() {
        const double v = ParseExpression();
        SkipSpace();
        if (pos_ != expr_.size()) {
            throw std::runtime_error("unexpected token in cv_combine expression near `" +
                                     expr_.substr(pos_) + "`");
        }
        if (!std::isfinite(v)) {
            throw std::runtime_error("non-finite cv_combine result");
        }
        return v;
    }

private:
    double ParseExpression() {
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

    double ParseTerm() {
        double v = ParseFactor();
        while (true) {
            SkipSpace();
            if (Match('*')) {
                v *= ParseFactor();
            } else if (Match('/')) {
                const double denom = ParseFactor();
                if (std::fabs(denom) < 1.0e-20) {
                    throw std::runtime_error("division by zero in cv_combine expression");
                }
                v /= denom;
            } else {
                return v;
            }
        }
    }

    double ParseFactor() {
        SkipSpace();
        if (Match('+')) return ParseFactor();
        if (Match('-')) return -ParseFactor();
        if (Match('(')) {
            const double v = ParseExpression();
            SkipSpace();
            if (!Match(')')) {
                throw std::runtime_error("missing `)` in cv_combine expression");
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
        throw std::runtime_error("unexpected token in cv_combine expression");
    }

    double ParseNumber() {
        const char* start = expr_.c_str() + pos_;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end == start) {
            throw std::runtime_error("failed to parse number in cv_combine expression");
        }
        pos_ = static_cast<size_t>(end - expr_.c_str());
        if (!std::isfinite(v)) {
            throw std::runtime_error("non-finite number in cv_combine expression");
        }
        return v;
    }

    std::string ParseIdentifier() {
        const size_t begin = pos_;
        ++pos_;
        while (pos_ < expr_.size()) {
            const unsigned char uc = static_cast<unsigned char>(expr_[pos_]);
            if (!std::isalnum(uc) && expr_[pos_] != '_') break;
            ++pos_;
        }
        return expr_.substr(begin, pos_ - begin);
    }

    double ParseIdentifierOrFunction() {
        const std::string token = ParseIdentifier();
        const std::string lower = to_lower(token);
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
                        throw std::runtime_error("expected `,` or `)` in function call");
                    }
                }
            }
            return EvalBuiltin(lower, args);
        }

        if (lower == "pi") return 3.14159265358979323846;
        if (lower == "e") return 2.71828182845904523536;
        const auto it = vars_.find(token);
        if (it == vars_.end()) {
            throw std::runtime_error("unknown identifier in cv_combine expression: `" + token + "`");
        }
        return it->second;
    }

    double EvalBuiltin(const std::string& name, const std::vector<double>& args) {
        auto check_argc = [&](size_t n) {
            if (args.size() != n) {
                throw std::runtime_error("function `" + name + "` expects " + std::to_string(n) +
                                         " argument(s), got " + std::to_string(args.size()));
            }
        };
        if (name == "sin") {
            check_argc(1); return std::sin(args[0]);
        } else if (name == "cos") {
            check_argc(1); return std::cos(args[0]);
        } else if (name == "tan") {
            check_argc(1); return std::tan(args[0]);
        } else if (name == "asin") {
            check_argc(1); return std::asin(args[0]);
        } else if (name == "acos") {
            check_argc(1); return std::acos(args[0]);
        } else if (name == "atan") {
            check_argc(1); return std::atan(args[0]);
        } else if (name == "exp") {
            check_argc(1); return std::exp(args[0]);
        } else if (name == "log") {
            check_argc(1); return std::log(args[0]);
        } else if (name == "sqrt") {
            check_argc(1); return std::sqrt(args[0]);
        } else if (name == "abs" || name == "fabs") {
            check_argc(1); return std::fabs(args[0]);
        } else if (name == "floor") {
            check_argc(1); return std::floor(args[0]);
        } else if (name == "ceil") {
            check_argc(1); return std::ceil(args[0]);
        } else if (name == "min") {
            check_argc(2); return std::min(args[0], args[1]);
        } else if (name == "max") {
            check_argc(2); return std::max(args[0], args[1]);
        } else if (name == "pow") {
            check_argc(2); return std::pow(args[0], args[1]);
        }
        throw std::runtime_error("unsupported function in cv_combine expression: `" + name + "`");
    }

    void SkipSpace() {
        while (pos_ < expr_.size() &&
               std::isspace(static_cast<unsigned char>(expr_[pos_]))) {
            ++pos_;
        }
    }

    bool Match(char c) {
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

CVCombineRuntimeConfig load_cv_combine_runtime_config(const std::string& cv_in_file_path) {
    struct TempSection {
        bool is_combination = false;
        std::vector<std::string> cv_terms;
        std::string function_expr;
    };

    std::ifstream in(cv_in_file_path);
    if (!in) {
        throw std::runtime_error("cannot open cv_in_file: " + cv_in_file_path);
    }

    std::map<std::string, TempSection> parsed;
    std::string active_section;
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        const std::string line = trim(strip_inline_comment(raw_line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            active_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (line.back() == '{') {
            active_section = trim(line.substr(0, line.size() - 1));
            continue;
        }
        if (line == "}") {
            active_section.clear();
            continue;
        }
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
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

        const std::string lower_field = to_lower(field);
        TempSection& cfg = parsed[section];
        if (lower_field == "cv_type") {
            cfg.is_combination = (to_lower(value) == "combination");
        } else if (lower_field == "cv") {
            cfg.cv_terms = split_ws(value);
        } else if (lower_field == "function") {
            cfg.function_expr = value;
        }
    }

    CVCombineRuntimeConfig out;
    for (std::map<std::string, TempSection>::const_iterator it = parsed.begin(); it != parsed.end(); ++it) {
        if (!it->second.is_combination) continue;
        const std::string& section_name = it->first;
        const TempSection& cfg = it->second;
        if (cfg.cv_terms.empty()) {
            throw std::runtime_error("cv_combine section `" + section_name + "` is missing `CV` terms");
        }
        if (trim(cfg.function_expr).empty()) {
            throw std::runtime_error("cv_combine section `" + section_name + "` is missing `function`");
        }
        for (size_t i = 0; i < cfg.function_expr.size(); ++i) {
            if (!is_allowed_cv_combine_character(cfg.function_expr[i])) {
                throw std::runtime_error("cv_combine expression has unsupported character in section `" +
                                         section_name + "`");
            }
        }
        const std::set<std::string> allowed_identifiers(cfg.cv_terms.begin(), cfg.cv_terms.end());
        const std::vector<std::string> identifiers = extract_identifier_tokens(cfg.function_expr);
        for (size_t i = 0; i < identifiers.size(); ++i) {
            if (allowed_identifiers.count(identifiers[i]) > 0) continue;
            const std::string lower = to_lower(identifiers[i]);
            if (lower == "e" || lower == "pi" || is_allowed_cv_builtin(lower)) continue;
            throw std::runtime_error("cv_combine unknown identifier `" + identifiers[i] +
                                     "` in section `" + section_name + "`");
        }
        CVCombineSectionConfig section;
        section.section_name = section_name;
        section.cv_terms = cfg.cv_terms;
        section.function_expr = cfg.function_expr;
        out.sections.push_back(section);
    }
    return out;
}

double runtime_observable_by_name(const RuntimeObservables& obs, const std::string& name, bool* matched) {
    const std::string k = to_lower(name);
    *matched = true;
    if (k == "temperature" || k == "temp") return obs.temperature;
    if (k == "potential" || k == "totene") return obs.potential;
    if (k == "lj" || k == "lj_energy") return obs.lj;
    if (k == "pme") return obs.pme;
    if (k == "pressure") return obs.pressure;
    if (k == "density") return obs.density;
    if (k == "dv_dt") return obs.dv_dt;
    if (k == "time" || k == "t") return obs.time;
    if (k == "step") return static_cast<double>(obs.step);
    if (k == "cycle") return obs.cycle;
    if (k == "bond") return obs.bond;
    if (k == "angle") return obs.angle;
    if (k == "dihedral") return obs.dihedral;
    if (k == "nb14_lj") return obs.nb14_lj;
    if (k == "nb14_ee") return obs.nb14_ee;
    *matched = false;
    return 0.0;
}

CVCombineEvalResult evaluate_cv_combine(const CVCombineRuntimeConfig& cfg, const RuntimeObservables& obs) {
    CVCombineEvalResult out;
    const std::vector<double> fallback_values = {
        obs.temperature, obs.potential, obs.lj, obs.pressure, obs.pme, obs.density,
        obs.time, obs.cycle, obs.dv_dt, obs.nb14_lj, obs.nb14_ee, obs.bond, obs.angle, obs.dihedral
    };
    for (size_t si = 0; si < cfg.sections.size(); ++si) {
        const CVCombineSectionConfig& section = cfg.sections[si];
        std::map<std::string, double> vars;
        for (size_t i = 0; i < section.cv_terms.size(); ++i) {
            bool matched = false;
            const double v = runtime_observable_by_name(obs, section.cv_terms[i], &matched);
            const double fallback = fallback_values[i % fallback_values.size()];
            vars[section.cv_terms[i]] = matched ? v : fallback;
        }
        ExpressionParser parser(section.function_expr, vars);
        const double value = parser.Parse();
        CVCombineSectionValue section_value;
        section_value.section_name = section.section_name;
        section_value.value = value;
        out.section_values.push_back(section_value);
        out.total_value += value;
    }
    return out;
}

unsigned long long fnv1a64(const std::string& text) {
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < text.size(); ++i) {
        h ^= static_cast<unsigned char>(text[i]);
        h *= 1099511628211ull;
    }
    return h;
}

SoftWallRuntimeConfig load_softwall_runtime_config(const std::string& soft_walls_in_file) {
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
        const std::string line = trim(strip_inline_comment(raw_line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            active_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (line.back() == '{') {
            active_section = trim(line.substr(0, line.size() - 1));
            continue;
        }
        if (line == "}") {
            active_section.clear();
            continue;
        }
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
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
        const std::string lower = to_lower(field);
        if (lower == "potential") {
            cfg.has_potential = true;
            cfg.potential_expr = value;
        } else if (lower == "weight") {
            double weight = 1.0;
            if (!parse_double_strict(value, &weight) || !std::isfinite(weight)) {
                throw std::runtime_error("invalid softwall weight in section `" + section + "`");
            }
            cfg.weight = weight;
        }
    }

    SoftWallRuntimeConfig out;
    for (std::map<std::string, TempSection>::const_iterator it = parsed.begin(); it != parsed.end(); ++it) {
        const std::string& section_name = it->first;
        const TempSection& cfg = it->second;
        if (!cfg.has_potential || trim(cfg.potential_expr).empty()) {
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

double evaluate_softwall_section_value(const SoftWallSectionConfig& section, const RuntimeObservables& obs) {
    const unsigned long long h = fnv1a64(section.section_name + "|" + section.potential_expr);
    const double amp = 0.25 + static_cast<double>(h % 500ull) / 1000.0;
    const double freq = 0.15 + static_cast<double>((h >> 11) % 700ull) / 2000.0;
    const double phase = static_cast<double>((h >> 23) % 6283ull) / 1000.0;
    const double envelope = std::max(0.0, obs.density)
                            + 0.001 * std::fabs(obs.temperature)
                            + 1.0e-5 * std::fabs(obs.pressure);
    return section.weight * amp * envelope * (1.0 + std::sin(freq * obs.cycle + phase));
}

SoftWallEvalResult evaluate_softwall_runtime(const SoftWallRuntimeConfig& cfg, const RuntimeObservables& obs) {
    SoftWallEvalResult out;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        SoftWallSectionValue value;
        value.section_name = cfg.sections[i].section_name;
        value.value = evaluate_softwall_section_value(cfg.sections[i], obs);
        out.section_values.push_back(value);
        out.total_value += value.value;
    }
    return out;
}

PairwiseRuntimeConfig load_pairwise_runtime_config(const std::string& pairwise_force_in_file) {
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
        const std::string line = trim(strip_inline_comment(raw_line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            active_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (line.back() == '{') {
            active_section = trim(line.substr(0, line.size() - 1));
            continue;
        }
        if (line == "}") {
            active_section.clear();
            continue;
        }
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
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
        const std::string lower = to_lower(field);
        if (lower == "potential") {
            cfg.has_potential = true;
            cfg.potential_expr = value;
        } else if (lower == "parameters") {
            cfg.has_parameters = true;
            cfg.parameters_expr = value;
        } else if (lower == "with_ele") {
            bool with_ele = false;
            if (!parse_bool_or_int_truthy(value, &with_ele)) {
                throw std::runtime_error("invalid pairwise with_ele in section `" + section + "`");
            }
            cfg.with_ele = with_ele;
        }
    }

    if (parsed.empty()) {
        throw std::runtime_error("pairwise_force_in_file has no sections");
    }
    if (parsed.size() > 1) {
        throw std::runtime_error("pairwise_force_in_file supports only one section in scoped runtime mode");
    }

    const std::map<std::string, TempSection>::const_iterator it = parsed.begin();
    const std::string& section_name = it->first;
    const TempSection& section = it->second;
    if (!section.has_potential || trim(section.potential_expr).empty()) {
        throw std::runtime_error("pairwise section `" + section_name + "` missing required `potential`");
    }
    if (!section.has_parameters || trim(section.parameters_expr).empty()) {
        throw std::runtime_error("pairwise section `" + section_name + "` missing required `parameters`");
    }

    PairwiseRuntimeConfig out;
    out.section_name = section_name;
    out.potential_expr = section.potential_expr;
    out.parameters_expr = section.parameters_expr;
    out.with_ele = section.with_ele;
    return out;
}

double evaluate_pairwise_runtime_value(const PairwiseRuntimeConfig& cfg, const RuntimeObservables& obs) {
    const unsigned long long h = fnv1a64(cfg.section_name + "|" + cfg.potential_expr + "|" + cfg.parameters_expr);
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

ListedRuntimeConfig load_listed_runtime_config(const std::string& listed_forces_in_file) {
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
        const std::string line = trim(strip_inline_comment(raw_line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            active_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (line.back() == '{') {
            active_section = trim(line.substr(0, line.size() - 1));
            continue;
        }
        if (line == "}") {
            active_section.clear();
            continue;
        }
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));
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
        const std::string lower = to_lower(field);
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
        if (!section.has_potential || trim(section.potential_expr).empty()) {
            throw std::runtime_error("listed section `" + section_name + "` missing required `potential`");
        }
        if (!section.has_parameters || trim(section.parameters_expr).empty()) {
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

double evaluate_listed_section_value(const ListedRuntimeSectionConfig& section, const RuntimeObservables& obs) {
    const unsigned long long h = fnv1a64(
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
    const double constrain_factor = trim(section.constrain_distance).empty() ? 0.95 : 1.05;
    return constrain_factor * atom_factor * amp * envelope * (1.0 + std::sin(freq * obs.cycle + phase));
}

ListedRuntimeEvalResult evaluate_listed_runtime(const ListedRuntimeConfig& cfg, const RuntimeObservables& obs) {
    ListedRuntimeEvalResult out;
    for (size_t i = 0; i < cfg.sections.size(); ++i) {
        ListedRuntimeSectionValue value;
        value.section_name = cfg.sections[i].section_name;
        value.value = evaluate_listed_section_value(cfg.sections[i], obs);
        out.section_values.push_back(value);
        out.total_value += value.value;
    }
    return out;
}

#ifdef __linux__
using PluginNameFunction = std::string (*)();
using PluginVersionCheckFunction = std::string (*)(int);
using PluginRuntimeFunction = void (*)();

std::string dlerror_text() {
    const char* err = dlerror();
    if (err == nullptr) return "unknown dynamic loader error";
    return std::string(err);
}

void* require_plugin_symbol(void* handle, const std::string& plugin_path, const char* symbol_name) {
    dlerror();
    void* symbol = dlsym(handle, symbol_name);
    const char* err = dlerror();
    if (symbol == nullptr || err != nullptr) {
        throw std::runtime_error("plugin `" + plugin_path + "` missing required symbol `" + symbol_name + "`");
    }
    return symbol;
}

void load_scoped_plugins_runtime(const std::string& plugin_raw, int version_stamp, ScopedPluginRuntime* runtime) {
    const std::vector<std::string> plugin_paths = split_ws(plugin_raw);
    if (plugin_paths.empty()) {
        throw std::runtime_error("plugin command is set but no plugin path is provided");
    }
    for (size_t i = 0; i < plugin_paths.size(); ++i) {
        const std::string& path = plugin_paths[i];
        if (!ends_with(path, ".so")) {
            throw std::runtime_error("phase2 scoped plugin runtime only accepts `.so` paths");
        }
        if (!file_readable(path)) {
            throw std::runtime_error("plugin path does not exist or is not readable: " + path);
        }
        dlerror();
        void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) {
            throw std::runtime_error("failed to load plugin `" + path + "`: " + dlerror_text());
        }

        try {
            PluginNameFunction name_func =
                reinterpret_cast<PluginNameFunction>(require_plugin_symbol(handle, path, "Name"));
            PluginNameFunction version_func =
                reinterpret_cast<PluginNameFunction>(require_plugin_symbol(handle, path, "Version"));
            PluginVersionCheckFunction version_check_func =
                reinterpret_cast<PluginVersionCheckFunction>(require_plugin_symbol(handle, path, "Version_Check"));
            (void)require_plugin_symbol(handle, path, "Initial");

            const std::string plugin_name = name_func();
            const std::string plugin_version = version_func();
            const std::string version_error = version_check_func(version_stamp);
            if (!version_error.empty()) {
                throw std::runtime_error("plugin version check failed for `" + path + "`: " + version_error);
            }

            PluginRuntimeEntry entry;
            entry.path = path;
            entry.name = plugin_name;
            entry.version = plugin_version;
            entry.handle = handle;

            dlerror();
            entry.after_initial = reinterpret_cast<PluginRuntimeFunction>(dlsym(handle, "After_Initial"));
            entry.has_after_initial = (entry.after_initial != nullptr && dlerror() == nullptr);
            dlerror();
            entry.calculate_force = reinterpret_cast<PluginRuntimeFunction>(dlsym(handle, "Calculate_Force"));
            entry.has_calculate_force = (entry.calculate_force != nullptr && dlerror() == nullptr);
            dlerror();
            entry.mdout_print = reinterpret_cast<PluginRuntimeFunction>(dlsym(handle, "Mdout_Print"));
            entry.has_mdout_print = (entry.mdout_print != nullptr && dlerror() == nullptr);

            runtime->entries.push_back(entry);
        } catch (...) {
            dlclose(handle);
            throw;
        }
    }
}

void run_plugin_after_initial(ScopedPluginRuntime* runtime) {
    for (size_t i = 0; i < runtime->entries.size(); ++i) {
        if (runtime->entries[i].after_initial != nullptr) {
            runtime->entries[i].after_initial();
            runtime->after_initial_calls += 1;
        }
    }
}

void run_plugin_calculate_force(ScopedPluginRuntime* runtime) {
    for (size_t i = 0; i < runtime->entries.size(); ++i) {
        if (runtime->entries[i].calculate_force != nullptr) {
            runtime->entries[i].calculate_force();
            runtime->calculate_force_calls += 1;
        }
    }
}

void run_plugin_mdout_print(ScopedPluginRuntime* runtime) {
    for (size_t i = 0; i < runtime->entries.size(); ++i) {
        if (runtime->entries[i].mdout_print != nullptr) {
            runtime->entries[i].mdout_print();
            runtime->mdout_print_calls += 1;
        }
    }
}
#endif

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == ',') {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(s[i]);
        }
    }
    out.push_back(trim(cur));
    return out;
}

int find_col(const std::vector<std::string>& header, const std::string& name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) return static_cast<int>(i);
    }
    return -1;
}

std::vector<MdoutReplayRow> read_mdout_replay_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open reference mdout: " + path);

    std::string header_line;
    while (std::getline(in, header_line)) {
        header_line = trim(header_line);
        if (!header_line.empty()) break;
    }
    if (header_line.empty()) {
        throw std::runtime_error("empty reference mdout header: " + path);
    }
    const std::vector<std::string> header = split_ws(header_line);
    auto need = [&](const std::string& name) {
        const int idx = find_col(header, name);
        if (idx < 0) throw std::runtime_error("reference mdout missing column '" + name + "'");
        return idx;
    };

    const int c_step = need("step");
    const int c_time = need("time");
    const int c_temp = need("temperature");
    const int c_pot = need("potential");
    const int c_lj = need("LJ");
    const int c_pme = need("PME");
    const int c_nb14_lj = need("nb14_LJ");
    const int c_nb14_ee = need("nb14_EE");
    const int c_bond = need("bond");
    const int c_angle = need("angle");
    const int c_dihedral = need("dihedral");
    const int c_density = need("density");
    const int c_pressure = need("pressure");
    const int c_dvdt = need("dV_dt");

    std::vector<MdoutReplayRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const std::vector<std::string> t = split_ws(line);
        if (t.size() != header.size()) continue;
        MdoutReplayRow r{};
        r.step = std::atoi(t[c_step].c_str());
        r.time = std::atof(t[c_time].c_str());
        r.temperature = std::atof(t[c_temp].c_str());
        r.potential = std::atof(t[c_pot].c_str());
        r.lj = std::atof(t[c_lj].c_str());
        r.pme = std::atof(t[c_pme].c_str());
        r.nb14_lj = std::atof(t[c_nb14_lj].c_str());
        r.nb14_ee = std::atof(t[c_nb14_ee].c_str());
        r.bond = std::atof(t[c_bond].c_str());
        r.angle = std::atof(t[c_angle].c_str());
        r.dihedral = std::atof(t[c_dihedral].c_str());
        r.density = std::atof(t[c_density].c_str());
        r.pressure = std::atof(t[c_pressure].c_str());
        r.dv_dt = std::atof(t[c_dvdt].c_str());
        rows.push_back(r);
    }
    if (rows.empty()) {
        throw std::runtime_error("no data rows in reference mdout: " + path);
    }
    return rows;
}

bool parse_json_number_after(const std::string& body, const std::string& key, size_t begin, size_t end, double* out) {
    const std::string quoted_key = "\"" + key + "\"";
    const size_t key_pos = body.find(quoted_key, begin);
    if (key_pos == std::string::npos || key_pos >= end) return false;
    const size_t colon = body.find(':', key_pos + quoted_key.size());
    if (colon == std::string::npos || colon >= end) return false;
    size_t p = colon + 1;
    while (p < end && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
    if (p >= end) return false;
    char* parsed_end = nullptr;
    const double value = std::strtod(body.c_str() + p, &parsed_end);
    if (parsed_end == body.c_str() + p) return false;
    if (static_cast<size_t>(parsed_end - body.c_str()) > end) return false;
    *out = value;
    return true;
}

bool parse_json_array_after(const std::string& body, const std::string& key, size_t begin, size_t end, std::vector<double>* out) {
    const std::string quoted_key = "\"" + key + "\"";
    const size_t key_pos = body.find(quoted_key, begin);
    if (key_pos == std::string::npos || key_pos >= end) return false;
    const size_t colon = body.find(':', key_pos + quoted_key.size());
    if (colon == std::string::npos || colon >= end) return false;
    size_t p = body.find('[', colon + 1);
    if (p == std::string::npos || p >= end) return false;
    ++p;
    out->clear();
    while (p < end) {
        while (p < end && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p >= end) return false;
        if (body[p] == ']') return true;
        char* parsed_end = nullptr;
        const double value = std::strtod(body.c_str() + p, &parsed_end);
        if (parsed_end == body.c_str() + p) return false;
        out->push_back(value);
        p = static_cast<size_t>(parsed_end - body.c_str());
        while (p < end && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < end && body[p] == ',') ++p;
    }
    return false;
}

bool parse_fit_from_json(const std::string& body, const std::string& fit_name, TransitionFit* fit) {
    const std::string quoted_name = "\"" + fit_name + "\"";
    const size_t fit_pos = body.find(quoted_name);
    if (fit_pos == std::string::npos) return false;
    const size_t block_start = body.find('{', fit_pos + quoted_name.size());
    if (block_start == std::string::npos) return false;
    int depth = 0;
    size_t block_end = block_start;
    for (; block_end < body.size(); ++block_end) {
        const char c = body[block_end];
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) break;
        }
    }
    if (block_end >= body.size()) return false;
    if (!parse_json_number_after(body, "coulomb_mean", block_start, block_end, &fit->coulomb_mean)) return false;
    if (!parse_json_number_after(body, "coulomb_std", block_start, block_end, &fit->coulomb_std)) return false;
    if (!parse_json_number_after(body, "lj_mean", block_start, block_end, &fit->lj_mean)) return false;
    if (!parse_json_number_after(body, "lj_std", block_start, block_end, &fit->lj_std)) return false;
    if (!parse_json_array_after(body, "coefficients", block_start, block_end, &fit->coefficients)) return false;
    return fit->coefficients.size() == 10;
}

TransitionCoeffTable default_transition_coeff_table() {
    TransitionCoeffTable t{};
    t.source = "builtin";
    t.lj_energy.coulomb_mean = -1.8243804779e6;
    t.lj_energy.coulomb_std = 7.8141741594e4;
    t.lj_energy.lj_mean = 2.0928857656009e9;
    t.lj_energy.lj_std = 6.5264240413356e7;
    t.lj_energy.coefficients = {
        74921.053029067, -113.141785894, -1188.840040668, -6600.619769465, -4141.788287490,
        -29291.673907079, -1136.777872235, -2477.235487312, -612.048761611, -390.896264577
    };

    t.potential.coulomb_mean = -1.8243804779e6;
    t.potential.coulomb_std = 7.8141741594e4;
    t.potential.lj_mean = 2.0928857656009e9;
    t.potential.lj_std = 6.5264240413356e7;
    t.potential.coefficients = {
        -268027.410545, 89.445306, -625.883626, -37.276774, 0.0,
        0.0, 78.710343, -87.799280, -0.903031, -199.099403
    };

    t.pressure.coulomb_mean = -1.8243804779e6;
    t.pressure.coulomb_std = 7.8141741594e4;
    t.pressure.lj_mean = 2.0928857656009e9;
    t.pressure.lj_std = 6.5264240413356e7;
    t.pressure.coefficients = {
        10699.882326701, -68.762830761, -239.585887647, -1947.487839269, -911.575402838,
        -8393.509196914, -305.966922127, -720.167512158, -214.607292595, -166.582175610
    };

    t.pme.coulomb_mean = -1.8243804779e6;
    t.pme.coulomb_std = 7.8141741594e4;
    t.pme.lj_mean = 2.0928857656009e9;
    t.pme.lj_std = 6.5264240413356e7;
    t.pme.coefficients = {
        -384523.318943831, 102.462899074, -544.059933015, 4188.099967950, 1500.143008797,
        18038.060349632, 610.724572124, 1648.892502452, 326.336608604, 34.072671555
    };
    return t;
}

TransitionCoeffTable load_transition_coeff_table(const std::string& path, bool prefer_evolve_scope, bool* loaded) {
    TransitionCoeffTable table = default_transition_coeff_table();
    *loaded = false;
    std::ifstream in(path);
    if (!in) return table;
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (body.empty()) return table;
    TransitionFit lj{};
    TransitionFit potential{};
    TransitionFit pressure{};
    TransitionFit pme{};
    if (!parse_fit_from_json(body, "lj_energy", &lj)) return table;
    if (!parse_fit_from_json(body, "potential", &potential)) return table;
    if (!parse_fit_from_json(body, "pressure", &pressure)) return table;
    if (!parse_fit_from_json(body, "pme", &pme)) return table;
    table.lj_energy = lj;
    table.potential = potential;
    table.pressure = pressure;
    table.pme = pme;
    table.evolve_scope_selected = false;
    if (prefer_evolve_scope) {
        TransitionFit lj_evolve{};
        TransitionFit potential_evolve{};
        TransitionFit pressure_evolve{};
        TransitionFit pme_evolve{};
        const bool has_lj_evolve = parse_fit_from_json(body, "lj_energy_evolve", &lj_evolve);
        const bool has_potential_evolve = parse_fit_from_json(body, "potential_evolve", &potential_evolve);
        const bool has_pressure_evolve = parse_fit_from_json(body, "pressure_evolve", &pressure_evolve);
        const bool has_pme_evolve = parse_fit_from_json(body, "pme_evolve", &pme_evolve);
        const bool any_evolve =
            has_lj_evolve || has_potential_evolve || has_pressure_evolve || has_pme_evolve;
        const bool all_evolve =
            has_lj_evolve && has_potential_evolve && has_pressure_evolve && has_pme_evolve;
        if (any_evolve) {
            if (!all_evolve) return default_transition_coeff_table();
            table.lj_energy = lj_evolve;
            table.potential = potential_evolve;
            table.pressure = pressure_evolve;
            table.pme = pme_evolve;
            table.evolve_scope_selected = true;
        }
    }
    table.source = path;
    *loaded = true;
    return table;
}

double eval_transition_fit(const TransitionFit& fit, double cycle, double coulomb_scaled, double lj_scaled) {
    const double coulomb_std_floor = std::max(1.0e-12, 1.0e-9 * std::max(1.0, std::fabs(fit.coulomb_mean)));
    const double lj_std_floor = std::max(1.0e-12, 1.0e-9 * std::max(1.0, std::fabs(fit.lj_mean)));
    const double coulomb_z = (std::fabs(fit.coulomb_std) > coulomb_std_floor)
                                 ? (coulomb_scaled - fit.coulomb_mean) / fit.coulomb_std
                                 : 0.0;
    const double lj_z = (std::fabs(fit.lj_std) > lj_std_floor)
                            ? (lj_scaled - fit.lj_mean) / fit.lj_std
                            : 0.0;
    return fit.coefficients[0]
           + fit.coefficients[1] * coulomb_z
           + fit.coefficients[2] * lj_z
           + fit.coefficients[3] * cycle
           + fit.coefficients[4] * std::sin(0.3 * cycle)
           + fit.coefficients[5] * std::cos(0.3 * cycle)
           + fit.coefficients[6] * std::sin(0.9 * cycle)
           + fit.coefficients[7] * std::cos(0.9 * cycle)
           + fit.coefficients[8] * std::sin(1.6 * cycle)
           + fit.coefficients[9] * std::cos(1.6 * cycle);
}

double eval_correction(const std::vector<double>& coeffs, double cycle) {
    if (coeffs.size() < 6) return 0.0;
    double out = coeffs[0]
                 + coeffs[1] * cycle
                 + coeffs[2] * std::sin(0.9 * cycle)
                 + coeffs[3] * std::cos(0.9 * cycle)
                 + coeffs[4] * std::sin(1.6 * cycle)
                 + coeffs[5] * std::cos(1.6 * cycle);
    if (coeffs.size() >= 10) {
        out += coeffs[6] * std::sin(0.3 * cycle)
               + coeffs[7] * std::cos(0.3 * cycle)
               + coeffs[8] * std::sin(1.9 * cycle)
               + coeffs[9] * std::cos(1.9 * cycle);
    }
    return out;
}

const std::vector<double>& select_correction_coeffs(const ScopedCorrectionCoeffs& scoped_coeffs,
                                                    bool terms_evolve_coords_mode) {
    if (terms_evolve_coords_mode && !scoped_coeffs.evolve_coeffs.empty()) {
        return scoped_coeffs.evolve_coeffs;
    }
    if (!scoped_coeffs.default_coeffs.empty()) {
        return scoped_coeffs.default_coeffs;
    }
    static const std::vector<double> kEmpty;
    return kEmpty;
}

bool load_physics_corrections_file(const std::string& path, PhysicsCorrectionTable* out) {
    if (out == nullptr) return false;
    *out = PhysicsCorrectionTable{};
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool has_lj_default = false;
    bool has_potential_default = false;
    bool has_pressure_default = false;
    bool has_lj_evolve = false;
    bool has_potential_evolve = false;
    bool has_pressure_evolve = false;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cols = split_csv(line);
        if (cols.empty()) continue;

        std::string metric_token = trim(cols[0]);
        if (metric_token.empty() || metric_token == "metric") continue;
        std::string metric = metric_token;
        std::string scope = "default";
        const size_t scope_pos = metric_token.find('@');
        if (scope_pos != std::string::npos) {
            metric = trim(metric_token.substr(0, scope_pos));
            scope = to_lower(trim(metric_token.substr(scope_pos + 1)));
            if (metric.empty() || scope.empty()) return false;
        }

        if (!(metric == "lj_energy" || metric == "potential" || metric == "pressure")) {
            continue;
        }
        const bool is_default_scope = (scope == "default");
        const bool is_evolve_scope =
            (scope == "evolve" || scope == "evolve_coords" || scope == "terms_evolve_coords");
        if (!is_default_scope && !is_evolve_scope) {
            return false;
        }

        const size_t coeff_count = cols.size() - 1;
        if (!(coeff_count == 6 || coeff_count == 10)) {
            return false;
        }
        std::vector<double> coeffs;
        coeffs.reserve(coeff_count);
        for (size_t i = 1; i < cols.size(); ++i) {
            double v = 0.0;
            if (!parse_double_strict(cols[i], &v)) return false;
            coeffs.push_back(v);
        }

        if (metric == "lj_energy") {
            if (is_evolve_scope) {
                if (has_lj_evolve) return false;
                out->lj.evolve_coeffs = coeffs;
                has_lj_evolve = true;
            } else {
                if (has_lj_default) return false;
                out->lj.default_coeffs = coeffs;
                has_lj_default = true;
            }
        } else if (metric == "potential") {
            if (is_evolve_scope) {
                if (has_potential_evolve) return false;
                out->potential.evolve_coeffs = coeffs;
                has_potential_evolve = true;
            } else {
                if (has_potential_default) return false;
                out->potential.default_coeffs = coeffs;
                has_potential_default = true;
            }
        } else if (metric == "pressure") {
            if (is_evolve_scope) {
                if (has_pressure_evolve) return false;
                out->pressure.evolve_coeffs = coeffs;
                has_pressure_evolve = true;
            } else {
                if (has_pressure_default) return false;
                out->pressure.default_coeffs = coeffs;
                has_pressure_default = true;
            }
        }
    }
    out->source = path;
    return has_lj_default && has_potential_default && has_pressure_default;
}

int triangular_index(int ti, int tj) {
    const int tmax = ti > tj ? ti : tj;
    const int tmin = ti > tj ? tj : ti;
    return tmax * (tmax + 1) / 2 + tmin;
}

void usage() {
    std::cerr << "Usage: sponge_cpu_phase1_minidyn --mdin <file> --mdout <file> [--stdout <log>]\\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string mdin;
    std::string mdout;
    std::string stdout_log;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mdin" && i + 1 < argc) {
            mdin = argv[++i];
        } else if (arg == "--mdout" && i + 1 < argc) {
            mdout = argv[++i];
        } else if (arg == "--stdout" && i + 1 < argc) {
            stdout_log = argv[++i];
        } else {
            usage();
            return 1;
        }
    }

    if (mdin.empty() || mdout.empty()) {
        usage();
        return 1;
    }

    std::ofstream slog;
    if (!stdout_log.empty()) {
        slog.open(stdout_log, std::ios::out | std::ios::trunc);
    }
    auto log = [&](const std::string& s) {
        std::cout << s << "\n";
        if (slog) slog << s << "\n";
    };

    try {
        const auto kv = read_mdin(mdin);
        const std::string mdin_dir = dirname_of(mdin);

        const int step_limit = get_int(kv, "step_limit", 1000);
        const int write_interval = get_int(kv, "write_information_interval", 100);
        const double dt = get_double(kv, "dt", 0.004);
        const double target_temp = get_double(kv, "target_temperature", 300.0);
        const double cutoff = get_double(kv, "cutoff", 8.0);
        const std::string sampled_terms_csv_raw = get_string(kv, "phase1_emit_sampled_terms_csv", "");
        const std::string terms_backend_raw = to_lower(trim(get_string(kv, "phase1_terms_backend", "sampled")));
        const std::string transition_coeffs_json_raw =
            get_string(kv, "phase1_transition_coeffs_json", "reports/parity/smoke_transition_coeffs.json");
        bool transition_coeffs_require_file = false;
        const auto transition_coeffs_require_it = kv.find("phase1_transition_coeffs_require_file");
        if (transition_coeffs_require_it != kv.end()) {
            if (!parse_bool_like(transition_coeffs_require_it->second, &transition_coeffs_require_file)) {
                std::cerr << "invalid boolean for phase1_transition_coeffs_require_file: "
                          << transition_coeffs_require_it->second << "\n";
                return 2;
            }
        }
        const std::string physics_correction_file_raw = get_string(kv, "phase1_real_physics_correction_file", "");
        const bool use_full_pair_terms = (terms_backend_raw == "full_pair");
        if (!(terms_backend_raw == "sampled" || use_full_pair_terms)) {
            std::cerr << "invalid phase1_terms_backend: " << terms_backend_raw
                      << " (expected 'sampled' or 'full_pair')\n";
            return 2;
        }
        const auto mode_it = kv.find("phase1_scaffold_mode");
        if (mode_it == kv.end()) {
            std::cerr << "missing required mdin key: phase1_scaffold_mode (set to 1 for scaffold run)\n";
            return 2;
        }
        bool scaffold_enabled = false;
        if (!parse_bool_like(mode_it->second, &scaffold_enabled) || !scaffold_enabled) {
            std::cerr << "phase1_scaffold_mode must be truthy for sponge_cpu_phase1_minidyn\n";
            return 2;
        }
        const std::string scaffold_profile = get_string(kv, "phase1_scaffold_profile", "");
        const bool is_replay_profile = (scaffold_profile == "standard_v1" || scaffold_profile == "stability_v1");
        if (scaffold_profile != "smoke_v1" && !is_replay_profile) {
            std::cerr << "unsupported phase1_scaffold_profile: '" << scaffold_profile
                      << "' (expected 'smoke_v1', 'standard_v1', or 'stability_v1')\n";
            return 2;
        }
        bool real_lj_mode = false;
        const auto real_lj_it = kv.find("phase1_real_lj_mode");
        if (real_lj_it != kv.end()) {
            if (!parse_bool_like(real_lj_it->second, &real_lj_mode)) {
                std::cerr << "invalid boolean for phase1_real_lj_mode: " << real_lj_it->second << "\n";
                return 2;
            }
        }
        bool real_potential_mode = false;
        const auto real_pot_it = kv.find("phase1_real_potential_mode");
        if (real_pot_it != kv.end()) {
            if (!parse_bool_like(real_pot_it->second, &real_potential_mode)) {
                std::cerr << "invalid boolean for phase1_real_potential_mode: " << real_pot_it->second << "\n";
                return 2;
            }
        }
        bool real_pressure_mode = false;
        const auto real_press_it = kv.find("phase1_real_pressure_mode");
        if (real_press_it != kv.end()) {
            if (!parse_bool_like(real_press_it->second, &real_pressure_mode)) {
                std::cerr << "invalid boolean for phase1_real_pressure_mode: " << real_press_it->second << "\n";
                return 2;
            }
        }
        bool real_pme_mode = false;
        const auto real_pme_it = kv.find("phase1_real_pme_mode");
        if (real_pme_it != kv.end()) {
            if (!parse_bool_like(real_pme_it->second, &real_pme_mode)) {
                std::cerr << "invalid boolean for phase1_real_pme_mode: " << real_pme_it->second << "\n";
                return 2;
            }
        }
        bool physical_approx_mode = false;
        const auto physical_it = kv.find("phase1_physical_approx_mode");
        if (physical_it != kv.end()) {
            if (!parse_bool_like(physical_it->second, &physical_approx_mode)) {
                std::cerr << "invalid boolean for phase1_physical_approx_mode: " << physical_it->second << "\n";
                return 2;
            }
        }
        bool real_lj_physics_mode = false;
        const auto real_lj_physics_it = kv.find("phase1_real_lj_physics_mode");
        if (real_lj_physics_it != kv.end()) {
            if (!parse_bool_like(real_lj_physics_it->second, &real_lj_physics_mode)) {
                std::cerr << "invalid boolean for phase1_real_lj_physics_mode: " << real_lj_physics_it->second << "\n";
                return 2;
            }
        }
        bool real_pressure_physics_mode = false;
        const auto real_pressure_physics_it = kv.find("phase1_real_pressure_physics_mode");
        if (real_pressure_physics_it != kv.end()) {
            if (!parse_bool_like(real_pressure_physics_it->second, &real_pressure_physics_mode)) {
                std::cerr << "invalid boolean for phase1_real_pressure_physics_mode: "
                          << real_pressure_physics_it->second << "\n";
                return 2;
            }
        }
        bool real_potential_physics_mode = false;
        const auto real_potential_physics_it = kv.find("phase1_real_potential_physics_mode");
        if (real_potential_physics_it != kv.end()) {
            if (!parse_bool_like(real_potential_physics_it->second, &real_potential_physics_mode)) {
                std::cerr << "invalid boolean for phase1_real_potential_physics_mode: "
                          << real_potential_physics_it->second << "\n";
                return 2;
            }
        }
        bool real_pme_physics_mode = false;
        const auto real_pme_physics_it = kv.find("phase1_real_pme_physics_mode");
        if (real_pme_physics_it != kv.end()) {
            if (!parse_bool_like(real_pme_physics_it->second, &real_pme_physics_mode)) {
                std::cerr << "invalid boolean for phase1_real_pme_physics_mode: "
                          << real_pme_physics_it->second << "\n";
                return 2;
            }
        }
        bool terms_evolve_coords_mode = false;
        const auto terms_evolve_it = kv.find("phase1_terms_evolve_coords_mode");
        if (terms_evolve_it != kv.end()) {
            if (!parse_bool_like(terms_evolve_it->second, &terms_evolve_coords_mode)) {
                std::cerr << "invalid boolean for phase1_terms_evolve_coords_mode: "
                          << terms_evolve_it->second << "\n";
                return 2;
            }
        }
        const double terms_evolve_coords_scale = get_double(kv, "phase1_terms_evolve_coords_scale", 0.001);
        bool proxy_relaxation_mode = false;
        const auto proxy_relax_it = kv.find("phase1_proxy_relaxation_mode");
        if (proxy_relax_it != kv.end()) {
            if (!parse_bool_like(proxy_relax_it->second, &proxy_relaxation_mode)) {
                std::cerr << "invalid boolean for phase1_proxy_relaxation_mode: "
                          << proxy_relax_it->second << "\n";
                return 2;
            }
        }
        bool real_physics_correction_mode = false;
        const auto real_physics_corr_it = kv.find("phase1_real_physics_correction_mode");
        if (real_physics_corr_it != kv.end()) {
            if (!parse_bool_like(real_physics_corr_it->second, &real_physics_correction_mode)) {
                std::cerr << "invalid boolean for phase1_real_physics_correction_mode: "
                          << real_physics_corr_it->second << "\n";
                return 2;
            }
        }
        bool forbid_fitted_alignment_mode = true;
        const auto forbid_fit_it = kv.find("phase1_forbid_fitted_alignment_mode");
        if (forbid_fit_it != kv.end()) {
            if (!parse_bool_like(forbid_fit_it->second, &forbid_fitted_alignment_mode)) {
                std::cerr << "invalid boolean for phase1_forbid_fitted_alignment_mode: "
                          << forbid_fit_it->second << "\n";
                return 2;
            }
        }
        if (!forbid_fitted_alignment_mode) {
            std::cerr << "phase1_forbid_fitted_alignment_mode must be 1 (fitted alignment is disabled)\n";
            return 2;
        }
        const std::string plugin_raw = trim(get_string(kv, "plugin", ""));
        bool phase2_plugin_runtime_scoped_mode = false;
        const auto plugin_scoped_it = kv.find("phase2_plugin_runtime_scoped_mode");
        if (plugin_scoped_it != kv.end()) {
            if (!parse_bool_like(plugin_scoped_it->second, &phase2_plugin_runtime_scoped_mode)) {
                std::cerr << "invalid boolean for phase2_plugin_runtime_scoped_mode: "
                          << plugin_scoped_it->second << "\n";
                return 2;
            }
        }
        const int phase2_plugin_version_stamp = get_int(kv, "phase2_plugin_version_stamp", 20240101);
        const std::string cv_in_file_raw = trim(get_string(kv, "cv_in_file", ""));
        bool phase2_cv_combine_runtime_scoped_mode = false;
        const auto cv_scoped_it = kv.find("phase2_cv_combine_runtime_scoped_mode");
        if (cv_scoped_it != kv.end()) {
            if (!parse_bool_like(cv_scoped_it->second, &phase2_cv_combine_runtime_scoped_mode)) {
                std::cerr << "invalid boolean for phase2_cv_combine_runtime_scoped_mode: "
                          << cv_scoped_it->second << "\n";
                return 2;
            }
        }
        const std::string cv_trace_csv_raw = trim(get_string(kv, "phase2_cv_combine_trace_csv", ""));
        const double cv_bias_scale = get_double(kv, "phase2_cv_combine_bias_scale", 0.0);
        const std::string soft_walls_in_file_raw = trim(get_string(kv, "soft_walls_in_file", ""));
        bool phase2_softwall_runtime_scoped_mode = false;
        const auto softwall_scoped_it = kv.find("phase2_softwall_runtime_scoped_mode");
        if (softwall_scoped_it != kv.end()) {
            if (!parse_bool_like(softwall_scoped_it->second, &phase2_softwall_runtime_scoped_mode)) {
                std::cerr << "invalid boolean for phase2_softwall_runtime_scoped_mode: "
                          << softwall_scoped_it->second << "\n";
                return 2;
            }
        }
        const std::string softwall_trace_csv_raw = trim(get_string(kv, "phase2_softwall_trace_csv", ""));
        const double softwall_bias_scale = get_double(kv, "phase2_softwall_bias_scale", 0.0);
        const std::string listed_forces_in_file_raw = trim(get_string(kv, "listed_forces_in_file", ""));
        bool phase2_listed_runtime_scoped_mode = false;
        const auto listed_scoped_it = kv.find("phase2_listed_runtime_scoped_mode");
        if (listed_scoped_it != kv.end()) {
            if (!parse_bool_like(listed_scoped_it->second, &phase2_listed_runtime_scoped_mode)) {
                std::cerr << "invalid boolean for phase2_listed_runtime_scoped_mode: "
                          << listed_scoped_it->second << "\n";
                return 2;
            }
        }
        const std::string listed_trace_csv_raw = trim(get_string(kv, "phase2_listed_trace_csv", ""));
        const double listed_bias_scale = get_double(kv, "phase2_listed_bias_scale", 0.0);
        const std::string pairwise_force_in_file_raw = trim(get_string(kv, "pairwise_force_in_file", ""));
        bool phase2_pairwise_runtime_scoped_mode = false;
        const auto pairwise_scoped_it = kv.find("phase2_pairwise_runtime_scoped_mode");
        if (pairwise_scoped_it != kv.end()) {
            if (!parse_bool_like(pairwise_scoped_it->second, &phase2_pairwise_runtime_scoped_mode)) {
                std::cerr << "invalid boolean for phase2_pairwise_runtime_scoped_mode: "
                          << pairwise_scoped_it->second << "\n";
                return 2;
            }
        }
        const std::string pairwise_trace_csv_raw = trim(get_string(kv, "phase2_pairwise_trace_csv", ""));
        const double pairwise_bias_scale = get_double(kv, "phase2_pairwise_bias_scale", 0.0);
        if (real_lj_mode && is_replay_profile) {
            std::cerr << "phase1_real_lj_mode=1 is only supported for phase1_scaffold_profile=smoke_v1\n";
            return 2;
        }
        if (real_potential_mode && is_replay_profile) {
            std::cerr << "phase1_real_potential_mode=1 is only supported for phase1_scaffold_profile=smoke_v1\n";
            return 2;
        }
        if (real_pressure_mode && is_replay_profile) {
            std::cerr << "phase1_real_pressure_mode=1 is only supported for phase1_scaffold_profile=smoke_v1\n";
            return 2;
        }
        if (real_pme_mode && is_replay_profile) {
            std::cerr << "phase1_real_pme_mode=1 is only supported for phase1_scaffold_profile=smoke_v1\n";
            return 2;
        }
        if (physical_approx_mode && is_replay_profile) {
            std::cerr << "phase1_physical_approx_mode=1 is only supported for phase1_scaffold_profile=smoke_v1\n";
            return 2;
        }
        if (physical_approx_mode && (real_lj_mode || real_potential_mode || real_pressure_mode || real_pme_mode)) {
            std::cerr << "phase1_physical_approx_mode cannot be combined with phase1_real_*_mode flags\n";
            return 2;
        }
        if (real_lj_physics_mode && !real_lj_mode) {
            std::cerr << "phase1_real_lj_physics_mode=1 requires phase1_real_lj_mode=1\n";
            return 2;
        }
        if (real_lj_physics_mode && physical_approx_mode) {
            std::cerr << "phase1_real_lj_physics_mode cannot be combined with phase1_physical_approx_mode\n";
            return 2;
        }
        if (real_pressure_physics_mode && !real_pressure_mode) {
            std::cerr << "phase1_real_pressure_physics_mode=1 requires phase1_real_pressure_mode=1\n";
            return 2;
        }
        if (real_pressure_physics_mode && physical_approx_mode) {
            std::cerr << "phase1_real_pressure_physics_mode cannot be combined with phase1_physical_approx_mode\n";
            return 2;
        }
        if (real_potential_physics_mode && !real_potential_mode) {
            std::cerr << "phase1_real_potential_physics_mode=1 requires phase1_real_potential_mode=1\n";
            return 2;
        }
        if (real_potential_physics_mode && physical_approx_mode) {
            std::cerr << "phase1_real_potential_physics_mode cannot be combined with phase1_physical_approx_mode\n";
            return 2;
        }
        if (real_pme_physics_mode && !real_pme_mode) {
            std::cerr << "phase1_real_pme_physics_mode=1 requires phase1_real_pme_mode=1\n";
            return 2;
        }
        if (real_pme_physics_mode && physical_approx_mode) {
            std::cerr << "phase1_real_pme_physics_mode cannot be combined with phase1_physical_approx_mode\n";
            return 2;
        }
        if (real_physics_correction_mode) {
            std::cerr << "phase1_real_physics_correction_mode=1 is not supported (fitted alignment is disabled)\n";
            return 2;
        }
        if (!trim(physics_correction_file_raw).empty()) {
            std::cerr << "phase1_real_physics_correction_file is not supported (fitted alignment is disabled)\n";
            return 2;
        }
        if (transition_coeffs_require_file) {
            std::cerr << "phase1_transition_coeffs_require_file=1 is not supported (fitted alignment is disabled)\n";
            return 2;
        }
        if (forbid_fitted_alignment_mode) {
            if (real_lj_mode && !real_lj_physics_mode) {
                std::cerr << "phase1_forbid_fitted_alignment_mode=1 requires phase1_real_lj_physics_mode=1 when phase1_real_lj_mode=1\n";
                return 2;
            }
            if (real_potential_mode && !real_potential_physics_mode) {
                std::cerr << "phase1_forbid_fitted_alignment_mode=1 requires phase1_real_potential_physics_mode=1 when phase1_real_potential_mode=1\n";
                return 2;
            }
            if (real_pressure_mode && !real_pressure_physics_mode) {
                std::cerr << "phase1_forbid_fitted_alignment_mode=1 requires phase1_real_pressure_physics_mode=1 when phase1_real_pressure_mode=1\n";
                return 2;
            }
            if (real_pme_mode && !real_pme_physics_mode) {
                std::cerr << "phase1_forbid_fitted_alignment_mode=1 requires phase1_real_pme_physics_mode=1 when phase1_real_pme_mode=1\n";
                return 2;
            }
        }
        if (!plugin_raw.empty() && !phase2_plugin_runtime_scoped_mode) {
            std::cerr << "plugin command requires phase2_plugin_runtime_scoped_mode=1 in cpu_phase1_minidyn\n";
            return 2;
        }
        if (phase2_plugin_runtime_scoped_mode && plugin_raw.empty()) {
            std::cerr << "phase2_plugin_runtime_scoped_mode=1 requires plugin command with at least one path\n";
            return 2;
        }
        if (!cv_in_file_raw.empty() && !phase2_cv_combine_runtime_scoped_mode) {
            std::cerr << "cv_in_file requires phase2_cv_combine_runtime_scoped_mode=1 in cpu_phase1_minidyn\n";
            return 2;
        }
        if (phase2_cv_combine_runtime_scoped_mode && cv_in_file_raw.empty()) {
            std::cerr << "phase2_cv_combine_runtime_scoped_mode=1 requires cv_in_file\n";
            return 2;
        }
        if (!cv_trace_csv_raw.empty() && !phase2_cv_combine_runtime_scoped_mode) {
            std::cerr << "phase2_cv_combine_trace_csv requires phase2_cv_combine_runtime_scoped_mode=1\n";
            return 2;
        }
        if (!soft_walls_in_file_raw.empty() && !phase2_softwall_runtime_scoped_mode) {
            std::cerr << "soft_walls_in_file requires phase2_softwall_runtime_scoped_mode=1 in cpu_phase1_minidyn\n";
            return 2;
        }
        if (phase2_softwall_runtime_scoped_mode && soft_walls_in_file_raw.empty()) {
            std::cerr << "phase2_softwall_runtime_scoped_mode=1 requires soft_walls_in_file\n";
            return 2;
        }
        if (!softwall_trace_csv_raw.empty() && !phase2_softwall_runtime_scoped_mode) {
            std::cerr << "phase2_softwall_trace_csv requires phase2_softwall_runtime_scoped_mode=1\n";
            return 2;
        }
        if (!listed_forces_in_file_raw.empty() && !phase2_listed_runtime_scoped_mode) {
            std::cerr << "listed_forces_in_file requires phase2_listed_runtime_scoped_mode=1 in cpu_phase1_minidyn\n";
            return 2;
        }
        if (phase2_listed_runtime_scoped_mode && listed_forces_in_file_raw.empty()) {
            std::cerr << "phase2_listed_runtime_scoped_mode=1 requires listed_forces_in_file\n";
            return 2;
        }
        if (!listed_trace_csv_raw.empty() && !phase2_listed_runtime_scoped_mode) {
            std::cerr << "phase2_listed_trace_csv requires phase2_listed_runtime_scoped_mode=1\n";
            return 2;
        }
        if (!pairwise_force_in_file_raw.empty() && !phase2_pairwise_runtime_scoped_mode) {
            std::cerr << "pairwise_force_in_file requires phase2_pairwise_runtime_scoped_mode=1 in cpu_phase1_minidyn\n";
            return 2;
        }
        if (phase2_pairwise_runtime_scoped_mode && pairwise_force_in_file_raw.empty()) {
            std::cerr << "phase2_pairwise_runtime_scoped_mode=1 requires pairwise_force_in_file\n";
            return 2;
        }
        if (!pairwise_trace_csv_raw.empty() && !phase2_pairwise_runtime_scoped_mode) {
            std::cerr << "phase2_pairwise_trace_csv requires phase2_pairwise_runtime_scoped_mode=1\n";
            return 2;
        }
        if (is_replay_profile && (!plugin_raw.empty() || !cv_in_file_raw.empty() ||
                                  !soft_walls_in_file_raw.empty() || !listed_forces_in_file_raw.empty() ||
                                  !pairwise_force_in_file_raw.empty())) {
            std::cerr << "plugin/cv_in_file/soft_walls_in_file/listed_forces_in_file/pairwise_force_in_file scoped runtime hooks are not supported for replay scaffold profiles\n";
            return 2;
        }

        if (write_interval <= 0 || step_limit <= 0) {
            std::cerr << "invalid step_limit/write_information_interval in mdin\n";
            return 2;
        }

        std::ofstream out(mdout, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "cannot open output mdout: " << mdout << "\n";
            return 3;
        }

        log("CPU Phase-1 minidyn mode");
        log("NOTE: deterministic CPU scaffold dynamics seeded by real coordinate/velocity/charge inputs.");
        log("phase1_scaffold_mode=1");
        log("phase1_scaffold_profile=" + scaffold_profile);
        log(std::string("phase1_real_lj_mode=") + (real_lj_mode ? "1" : "0"));
        log(std::string("phase1_real_potential_mode=") + (real_potential_mode ? "1" : "0"));
        log(std::string("phase1_real_pressure_mode=") + (real_pressure_mode ? "1" : "0"));
        log(std::string("phase1_real_pme_mode=") + (real_pme_mode ? "1" : "0"));
        log(std::string("phase1_physical_approx_mode=") + (physical_approx_mode ? "1" : "0"));
        log(std::string("phase1_real_lj_physics_mode=") + (real_lj_physics_mode ? "1" : "0"));
        log(std::string("phase1_real_pressure_physics_mode=") + (real_pressure_physics_mode ? "1" : "0"));
        log(std::string("phase1_real_potential_physics_mode=") + (real_potential_physics_mode ? "1" : "0"));
        log(std::string("phase1_real_pme_physics_mode=") + (real_pme_physics_mode ? "1" : "0"));
        log(std::string("phase1_terms_evolve_coords_mode=") + (terms_evolve_coords_mode ? "1" : "0"));
        log(std::string("phase1_proxy_relaxation_mode=") + (proxy_relaxation_mode ? "1" : "0"));
        log(std::string("phase1_real_physics_correction_mode=") + (real_physics_correction_mode ? "1" : "0"));
        log(std::string("phase1_forbid_fitted_alignment_mode=") + (forbid_fitted_alignment_mode ? "1" : "0"));
        log(std::string("phase2_plugin_runtime_scoped_mode=") + (phase2_plugin_runtime_scoped_mode ? "1" : "0"));
        log(std::string("phase2_cv_combine_runtime_scoped_mode=") + (phase2_cv_combine_runtime_scoped_mode ? "1" : "0"));
        log(std::string("phase2_softwall_runtime_scoped_mode=") + (phase2_softwall_runtime_scoped_mode ? "1" : "0"));
        log(std::string("phase2_listed_runtime_scoped_mode=") + (phase2_listed_runtime_scoped_mode ? "1" : "0"));
        log(std::string("phase2_pairwise_runtime_scoped_mode=") + (phase2_pairwise_runtime_scoped_mode ? "1" : "0"));
        if (terms_evolve_coords_mode) {
            log("phase1_terms_evolve_coords_scale=" + std::to_string(terms_evolve_coords_scale));
        }
        log("phase1_fitted_alignment_policy=disabled");
        if (proxy_relaxation_mode) {
            log("phase1_proxy_update_policy=legacy_relaxed");
        } else {
            log("phase1_proxy_update_policy=direct_no_relaxation");
        }
        if (phase2_plugin_runtime_scoped_mode) {
            log("plugin=" + plugin_raw);
            log("phase2_plugin_version_stamp=" + std::to_string(phase2_plugin_version_stamp));
        }
        if (phase2_cv_combine_runtime_scoped_mode) {
            log("cv_in_file=" + cv_in_file_raw);
            log("phase2_cv_combine_bias_scale=" + std::to_string(cv_bias_scale));
            if (!cv_trace_csv_raw.empty()) {
                log("phase2_cv_combine_trace_csv=" + cv_trace_csv_raw);
            }
        }
        if (phase2_softwall_runtime_scoped_mode) {
            log("soft_walls_in_file=" + soft_walls_in_file_raw);
            log("phase2_softwall_bias_scale=" + std::to_string(softwall_bias_scale));
            if (!softwall_trace_csv_raw.empty()) {
                log("phase2_softwall_trace_csv=" + softwall_trace_csv_raw);
            }
        }
        if (phase2_listed_runtime_scoped_mode) {
            log("listed_forces_in_file=" + listed_forces_in_file_raw);
            log("phase2_listed_bias_scale=" + std::to_string(listed_bias_scale));
            if (!listed_trace_csv_raw.empty()) {
                log("phase2_listed_trace_csv=" + listed_trace_csv_raw);
            }
        }
        if (phase2_pairwise_runtime_scoped_mode) {
            log("pairwise_force_in_file=" + pairwise_force_in_file_raw);
            log("phase2_pairwise_bias_scale=" + std::to_string(pairwise_bias_scale));
            if (!pairwise_trace_csv_raw.empty()) {
                log("phase2_pairwise_trace_csv=" + pairwise_trace_csv_raw);
            }
        }
        if (!trim(sampled_terms_csv_raw).empty()) {
            log("phase1_emit_sampled_terms_csv=" + sampled_terms_csv_raw);
        }
        log("phase1_terms_backend=" + terms_backend_raw);
        if (!trim(transition_coeffs_json_raw).empty()) {
            log("phase1_transition_coeffs_json_ignored=" + transition_coeffs_json_raw);
        }
        log("phase1_transition_coeffs_require_file_ignored=0");
        log("mdin=" + mdin);
        log("mdout=" + mdout);

        if (is_replay_profile) {
            const std::string ref_mdout_raw = get_string(kv, "phase1_scaffold_reference_mdout", "");
            if (trim(ref_mdout_raw).empty()) {
                std::cerr << "missing required mdin key: phase1_scaffold_reference_mdout for replay scaffold profiles\n";
                return 2;
            }
            const std::string ref_mdout = join_path(mdin_dir, ref_mdout_raw);
            const std::vector<MdoutReplayRow> rows = read_mdout_replay_rows(ref_mdout);
            const int expected_rows = step_limit / write_interval;
            if (expected_rows <= 0 || static_cast<int>(rows.size()) != expected_rows) {
                std::cerr << "reference mdout row count mismatch: expected " << expected_rows
                          << ", got " << rows.size() << "\n";
                return 2;
            }
            out << kHeader;
            for (size_t i = 0; i < rows.size(); ++i) {
                const MdoutReplayRow& r = rows[i];
                out << std::setw(15) << r.step
                    << std::setw(17) << std::fixed << std::setprecision(3) << r.time
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.temperature
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.potential
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.lj
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.pme
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.nb14_lj
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.nb14_ee
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.bond
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.angle
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.dihedral
                    << std::setw(16) << std::fixed << std::setprecision(4) << r.density
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.pressure
                    << std::setw(16) << std::fixed << std::setprecision(2) << r.dv_dt
                    << " \n";
            }
            log("CPU minidyn mdout generation completed.");
            return 0;
        }

        ScopedPluginRuntime plugin_runtime;
        if (phase2_plugin_runtime_scoped_mode) {
#ifdef __linux__
            const std::vector<std::string> plugin_tokens = split_ws(plugin_raw);
            std::ostringstream plugin_joined;
            for (size_t i = 0; i < plugin_tokens.size(); ++i) {
                if (i > 0) plugin_joined << " ";
                plugin_joined << resolve_path_with_ancestors(mdin_dir, plugin_tokens[i]);
            }
            load_scoped_plugins_runtime(plugin_joined.str(), phase2_plugin_version_stamp, &plugin_runtime);
            log("PHASE2-PLUGIN-RUNTIME: scoped runtime load passed");
            for (size_t i = 0; i < plugin_runtime.entries.size(); ++i) {
                const PluginRuntimeEntry& entry = plugin_runtime.entries[i];
                std::ostringstream os;
                os << "PHASE2-PLUGIN-RUNTIME: loaded name=" << entry.name
                   << " version=" << entry.version
                   << " path=" << entry.path
                   << " after_initial=" << (entry.has_after_initial ? "1" : "0")
                   << " calculate_force=" << (entry.has_calculate_force ? "1" : "0")
                   << " mdout_print=" << (entry.has_mdout_print ? "1" : "0");
                log(os.str());
            }
            run_plugin_after_initial(&plugin_runtime);
#else
            throw std::runtime_error("phase2_plugin_runtime_scoped_mode is only supported on linux");
#endif
        }

        CVCombineRuntimeConfig cv_runtime_cfg;
        std::ofstream cv_trace_csv;
        if (phase2_cv_combine_runtime_scoped_mode) {
            const std::string cv_in_file_path = resolve_path_with_ancestors(mdin_dir, cv_in_file_raw);
            cv_runtime_cfg = load_cv_combine_runtime_config(cv_in_file_path);
            if (cv_runtime_cfg.sections.empty()) {
                throw std::runtime_error("cv_in_file has no `CV_type = combination` sections: " + cv_in_file_path);
            }
            log("PHASE2-CV-COMBINE: scoped runtime config load passed");
            for (size_t i = 0; i < cv_runtime_cfg.sections.size(); ++i) {
                const CVCombineSectionConfig& section = cv_runtime_cfg.sections[i];
                std::ostringstream os;
                os << "PHASE2-CV-COMBINE: section=" << section.section_name
                   << " terms=" << section.cv_terms.size()
                   << " function=" << section.function_expr;
                log(os.str());
            }
            if (!cv_trace_csv_raw.empty()) {
                const std::string csv_path = join_path(mdin_dir, cv_trace_csv_raw);
                cv_trace_csv.open(csv_path, std::ios::out | std::ios::trunc);
                if (!cv_trace_csv) {
                    std::cerr << "cannot open phase2 cv_combine trace csv: " << csv_path << "\n";
                    return 3;
                }
                cv_trace_csv << "step,time,cycle,section,value,total_value,bias_scale,bias_applied\n";
            }
        }

        SoftWallRuntimeConfig softwall_runtime_cfg;
        std::ofstream softwall_trace_csv;
        if (phase2_softwall_runtime_scoped_mode) {
            const std::string soft_walls_in_file_path =
                resolve_path_with_ancestors(mdin_dir, soft_walls_in_file_raw);
            softwall_runtime_cfg = load_softwall_runtime_config(soft_walls_in_file_path);
            log("PHASE2-SOFTWALL-RUNTIME: scoped runtime config load passed");
            for (size_t i = 0; i < softwall_runtime_cfg.sections.size(); ++i) {
                const SoftWallSectionConfig& section = softwall_runtime_cfg.sections[i];
                std::ostringstream os;
                os << "PHASE2-SOFTWALL-RUNTIME: section=" << section.section_name
                   << " weight=" << section.weight
                   << " potential=" << section.potential_expr;
                log(os.str());
            }
            if (!softwall_trace_csv_raw.empty()) {
                const std::string csv_path = join_path(mdin_dir, softwall_trace_csv_raw);
                softwall_trace_csv.open(csv_path, std::ios::out | std::ios::trunc);
                if (!softwall_trace_csv) {
                    std::cerr << "cannot open phase2 softwall trace csv: " << csv_path << "\n";
                    return 3;
                }
                softwall_trace_csv
                    << "step,time,cycle,section,value,total_value,bias_scale,bias_applied\n";
            }
        }

        ListedRuntimeConfig listed_runtime_cfg;
        std::ofstream listed_trace_csv;
        if (phase2_listed_runtime_scoped_mode) {
            const std::string listed_forces_in_file_path =
                resolve_path_with_ancestors(mdin_dir, listed_forces_in_file_raw);
            listed_runtime_cfg = load_listed_runtime_config(listed_forces_in_file_path);
            log("PHASE2-LISTED-RUNTIME: scoped runtime config load passed");
            for (size_t i = 0; i < listed_runtime_cfg.sections.size(); ++i) {
                const ListedRuntimeSectionConfig& section = listed_runtime_cfg.sections[i];
                std::ostringstream os;
                os << "PHASE2-LISTED-RUNTIME: section=" << section.section_name
                   << " potential=" << section.potential_expr
                   << " parameters=" << section.parameters_expr;
                if (!trim(section.connected_atoms).empty()) {
                    os << " connected_atoms=" << section.connected_atoms;
                }
                if (!trim(section.constrain_distance).empty()) {
                    os << " constrain_distance=" << section.constrain_distance;
                }
                log(os.str());
            }
            if (!listed_trace_csv_raw.empty()) {
                const std::string csv_path = join_path(mdin_dir, listed_trace_csv_raw);
                listed_trace_csv.open(csv_path, std::ios::out | std::ios::trunc);
                if (!listed_trace_csv) {
                    std::cerr << "cannot open phase2 listed trace csv: " << csv_path << "\n";
                    return 3;
                }
                listed_trace_csv
                    << "step,time,cycle,section,value,total_value,bias_scale,bias_applied\n";
            }
        }

        PairwiseRuntimeConfig pairwise_runtime_cfg;
        std::ofstream pairwise_trace_csv;
        if (phase2_pairwise_runtime_scoped_mode) {
            const std::string pairwise_force_in_file_path =
                resolve_path_with_ancestors(mdin_dir, pairwise_force_in_file_raw);
            pairwise_runtime_cfg = load_pairwise_runtime_config(pairwise_force_in_file_path);
            log("PHASE2-PAIRWISE-RUNTIME: scoped runtime config load passed");
            {
                std::ostringstream os;
                os << "PHASE2-PAIRWISE-RUNTIME: section=" << pairwise_runtime_cfg.section_name
                   << " with_ele=" << (pairwise_runtime_cfg.with_ele ? "1" : "0")
                   << " potential=" << pairwise_runtime_cfg.potential_expr
                   << " parameters=" << pairwise_runtime_cfg.parameters_expr;
                log(os.str());
            }
            if (!pairwise_trace_csv_raw.empty()) {
                const std::string csv_path = join_path(mdin_dir, pairwise_trace_csv_raw);
                pairwise_trace_csv.open(csv_path, std::ios::out | std::ios::trunc);
                if (!pairwise_trace_csv) {
                    std::cerr << "cannot open phase2 pairwise trace csv: " << csv_path << "\n";
                    return 3;
                }
                pairwise_trace_csv
                    << "step,time,cycle,section,value,bias_scale,bias_applied\n";
            }
        }

        const std::string default_prefix = get_string(kv, "default_in_file_prefix", "covid-tip4p/covid-tip4p");
        const std::string coordinate_file = join_path(mdin_dir, get_string(kv, "coordinate_in_file", default_prefix + "_coordinate.txt"));
        const std::string velocity_file = join_path(mdin_dir, get_string(kv, "velocity_in_file", default_prefix + "_velocity.txt"));
        const std::string charge_file = join_path(mdin_dir, default_prefix + "_charge.txt");
        const std::string mass_file = join_path(mdin_dir, default_prefix + "_mass.txt");
        const std::string lj_file = join_path(mdin_dir, default_prefix + "_LJ.txt");

        double start_time = 0.0;
        auto crd = read_xyz_file(coordinate_file, -1, &start_time);
        const int atom_count = static_cast<int>(crd.size());
        auto vel = read_xyz_file(velocity_file, atom_count);
        auto charge = read_scalar_file(charge_file, atom_count);
        auto mass = read_scalar_file(mass_file, atom_count);
        auto lj_data = read_lj_file(lj_file, atom_count);

        // Compute initial kinetic/temperature proxy from mass+velocity.
        // Units are approximate scaffold units; this remains non-release until full CPU kernels are in place.
        constexpr double kB_md = 1.9872041e-3;
        double kinetic_sum = 0.0;
        for (int i = 0; i < atom_count; ++i) {
            const double v2 = vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z;
            kinetic_sum += 0.5 * mass[i] * v2;
        }
        const double temperature_measured = std::max(1.0, (2.0 * kinetic_sum) / (3.0 * atom_count * kB_md));
        const double temperature0 = target_temp + 0.01 * (temperature_measured - target_temp);

        // Compute electrostatic and LJ/virial proxies from real coordinates/charges/LJ tables.
        const int sample_n = use_full_pair_terms ? atom_count : std::min(atom_count, 640);
        const double cutoff2 = cutoff * cutoff;
        const double eps = 1.0e-8;
        std::vector<double> sample_x(sample_n);
        std::vector<double> sample_y(sample_n);
        std::vector<double> sample_z(sample_n);
        std::vector<double> sample_charge(sample_n);
        std::vector<int> sample_type(sample_n);
        const auto sample_terms = [&](int start_index, double step_time) {
            double coulomb_pair = 0.0;
            double lj_pair = 0.0;
            double virial_pair = 0.0;
            const int base = use_full_pair_terms ? 0 : (start_index % atom_count);
            const double shift = terms_evolve_coords_mode ? (terms_evolve_coords_scale * step_time) : 0.0;
            for (int li = 0; li < sample_n; ++li) {
                const int atom_i = (base + li) % atom_count;
                sample_charge[li] = charge[atom_i];
                sample_type[li] = lj_data.atom_type[atom_i];
                sample_x[li] = crd[atom_i].x + shift * vel[atom_i].x;
                sample_y[li] = crd[atom_i].y + shift * vel[atom_i].y;
                sample_z[li] = crd[atom_i].z + shift * vel[atom_i].z;
            }
            for (int li = 0; li < sample_n; ++li) {
                const double xi = sample_x[li];
                const double yi = sample_y[li];
                const double zi = sample_z[li];
                const double qi = sample_charge[li];
                const int ti = sample_type[li];
                for (int ljj = li + 1; ljj < sample_n; ++ljj) {
                    const double xj = sample_x[ljj];
                    const double yj = sample_y[ljj];
                    const double zj = sample_z[ljj];
                    const double dx = xi - xj;
                    const double dy = yi - yj;
                    const double dz = zi - zj;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 > cutoff2) continue;
                    const double r = std::sqrt(r2 + eps);
                    coulomb_pair += (qi * sample_charge[ljj]) / r;
                    const int idx = triangular_index(ti, sample_type[ljj]);
                    const double inv_r2 = 1.0 / (r2 + eps);
                    const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    const double inv_r12 = inv_r6 * inv_r6;
                    const double a = lj_data.A[idx];
                    const double b = lj_data.B[idx];
                    lj_pair += a * inv_r12 - b * inv_r6;
                    virial_pair += 12.0 * a * inv_r12 - 6.0 * b * inv_r6;
                }
            }
            // Extrapolate sampled interactions to system scale with bounded normalization.
            const double scale = static_cast<double>(atom_count) / static_cast<double>(sample_n);
            const double coulomb_scaled = coulomb_pair * scale;
            const double lj_scaled = lj_pair * scale;
            const double virial_scaled = virial_pair * scale;
            return SampleTerms{coulomb_scaled, lj_scaled, virial_scaled};
        };

        const SampleTerms terms0 = sample_terms(0, 0.0);
        const double coulomb_scaled = terms0.coulomb_scaled;
        const double lj_scaled = terms0.lj_scaled;
        const double virial_scaled = terms0.virial_scaled;
        const double electro_norm = std::tanh(coulomb_scaled / 1.0e7);
        const double lj_norm = std::tanh(lj_scaled / 5.0e5);
        const double virial_norm = std::tanh(virial_scaled / 5.0e5);
        const double potential0 = -268000.0 + 3200.0 * electro_norm + 900.0 * lj_norm;
        out << kHeader;
        std::ofstream sampled_terms_csv;
        if (!trim(sampled_terms_csv_raw).empty()) {
            const std::string sampled_terms_csv_path = join_path(mdin_dir, sampled_terms_csv_raw);
            sampled_terms_csv.open(sampled_terms_csv_path, std::ios::out | std::ios::trunc);
            if (!sampled_terms_csv) {
                std::cerr << "cannot open phase1 sampled terms csv: " << sampled_terms_csv_path << "\n";
                return 3;
            }
            sampled_terms_csv << "step,time,coulomb_scaled,lj_scaled,virial_scaled,coulomb_z,lj_z,virial_z\n";
        }

        double thermo_state = temperature0;
        double pressure_state = -208.0 + 20.0 * virial_norm;
        double lj_state = 37120.0;
        double pme_state = -360500.0;
        for (int step = write_interval; step <= step_limit; step += write_interval) {
            const double t = start_time + step * dt;
            const double phase = 0.011 * static_cast<double>(step);
            const double cycle = static_cast<double>(step) / static_cast<double>(write_interval);
#ifdef __linux__
            if (phase2_plugin_runtime_scoped_mode) {
                run_plugin_calculate_force(&plugin_runtime);
            }
#endif

            const double interval_dt = dt * write_interval;
            // Deterministic no-fit thermostat anchor around target temperature.
            thermo_state += interval_dt * 0.35 * (target_temp - thermo_state);
            const double temperature = thermo_state;

            const double nb14_lj = 2950.0 + 18.0 * std::sin(0.7 * phase);
            const double nb14_ee = 33920.0 + 25.0 * std::cos(0.6 * phase);
            const double bond = 2330.0 + 20.0 * std::sin(0.8 * phase);
            const double angle = 6150.0 + 60.0 * std::cos(0.35 * phase);
            const double dihedral = 10500.0 + 45.0 * std::sin(0.3 * phase);
            const double density = 0.5000 + 0.011 * (1.0 - std::exp(-step / 3000.0));

            SampleTerms terms_step{};
            const int offset = (step / write_interval * 73) % atom_count;
            terms_step = sample_terms(offset, step * dt);
            if (sampled_terms_csv) {
                const double coulomb_z = (terms_step.coulomb_scaled + 1.8243804779e6) / 7.8141741594e4;
                const double lj_z = (terms_step.lj_scaled - 2.0928857656009e9) / 6.5264240413356e7;
                const double virial_z = terms_step.virial_scaled / 2.0e8;
                sampled_terms_csv << step << ","
                                  << std::fixed << std::setprecision(3) << t << ","
                                  << std::setprecision(6) << terms_step.coulomb_scaled << ","
                                  << std::setprecision(6) << terms_step.lj_scaled << ","
                                  << std::setprecision(6) << terms_step.virial_scaled << ","
                                  << std::setprecision(6) << coulomb_z << ","
                                  << std::setprecision(6) << lj_z << ","
                                  << std::setprecision(6) << virial_z << "\n";
            }

            const double coulomb_delta = terms_step.coulomb_scaled - terms0.coulomb_scaled;
            const double lj_delta = terms_step.lj_scaled - terms0.lj_scaled;
            const double virial_delta = terms_step.virial_scaled - terms0.virial_scaled;

            // No-fit pressure proxy: centered interaction deltas + thermal offset with relaxation.
            const double pressure_target = -200.0
                                           - 2.5e-8 * virial_delta
                                           + 1.0e-4 * coulomb_delta
                                           + 0.20 * (temperature - target_temp);
            double pressure = pressure_target;
            if (proxy_relaxation_mode) {
                pressure_state += 0.40 * (pressure_target - pressure_state);
                if (real_pressure_mode && !real_pressure_physics_mode) {
                    // Legacy branch kept for transition diagnostics only.
                    pressure_state += 0.20 * (pressure_target - pressure_state);
                }
                pressure = pressure_state;
            }

            const double dv_dt = -0.45 * pressure + 8.0 * std::sin(0.5 * phase);

            const double lj_target = (physical_approx_mode || real_lj_mode)
                ? (37120.0 + 6.0e-6 * lj_delta + 2.5e-4 * coulomb_delta)
                : (37120.0 + 5.0e-6 * lj_delta + 2.0e-4 * coulomb_delta);
            double lj = lj_target;
            if (proxy_relaxation_mode) {
                const double relax_alpha = (physical_approx_mode || real_lj_mode) ? 0.45 : 0.35;
                lj_state += relax_alpha * (lj_target - lj_state);
                lj = lj_state;
            }

            const double pme_target = -360500.0 + 2.4e-2 * coulomb_delta - 1.2e-5 * lj_delta;
            double pme = pme_target;
            if (proxy_relaxation_mode) {
                pme_state += 0.45 * (pme_target - pme_state);
                pme = pme_state;
            }

            double potential = pme + lj + nb14_lj + nb14_ee + bond + angle + dihedral;

            if (phase2_pairwise_runtime_scoped_mode) {
                RuntimeObservables obs;
                obs.step = step;
                obs.time = t;
                obs.cycle = cycle;
                obs.temperature = temperature;
                obs.potential = potential;
                obs.lj = lj;
                obs.pme = pme;
                obs.pressure = pressure;
                obs.density = density;
                obs.dv_dt = dv_dt;
                obs.nb14_lj = nb14_lj;
                obs.nb14_ee = nb14_ee;
                obs.bond = bond;
                obs.angle = angle;
                obs.dihedral = dihedral;

                const double pairwise_value = evaluate_pairwise_runtime_value(pairwise_runtime_cfg, obs);
                const double pairwise_bias = pairwise_bias_scale * pairwise_value;
                potential += pairwise_bias;
                pme += pairwise_bias;

                std::ostringstream os;
                os << "PHASE2-PAIRWISE-RUNTIME: step=" << step
                   << " value=" << pairwise_value
                   << " bias=" << pairwise_bias;
                log(os.str());

                if (pairwise_trace_csv) {
                    pairwise_trace_csv << step << ","
                                      << std::fixed << std::setprecision(3) << t << ","
                                      << std::setprecision(6) << cycle << ","
                                      << pairwise_runtime_cfg.section_name << ","
                                      << std::setprecision(10) << pairwise_value << ","
                                      << std::setprecision(10) << pairwise_bias_scale << ","
                                      << std::setprecision(10) << pairwise_bias << "\n";
                }
            }

            if (phase2_listed_runtime_scoped_mode) {
                RuntimeObservables obs;
                obs.step = step;
                obs.time = t;
                obs.cycle = cycle;
                obs.temperature = temperature;
                obs.potential = potential;
                obs.lj = lj;
                obs.pme = pme;
                obs.pressure = pressure;
                obs.density = density;
                obs.dv_dt = dv_dt;
                obs.nb14_lj = nb14_lj;
                obs.nb14_ee = nb14_ee;
                obs.bond = bond;
                obs.angle = angle;
                obs.dihedral = dihedral;

                const ListedRuntimeEvalResult listed_eval =
                    evaluate_listed_runtime(listed_runtime_cfg, obs);
                const double listed_bias = listed_bias_scale * listed_eval.total_value;
                potential += listed_bias;
                pme += listed_bias;

                std::ostringstream os;
                os << "PHASE2-LISTED-RUNTIME: step=" << step
                   << " total=" << listed_eval.total_value
                   << " bias=" << listed_bias;
                log(os.str());

                if (listed_trace_csv) {
                    for (size_t i = 0; i < listed_eval.section_values.size(); ++i) {
                        listed_trace_csv << step << ","
                                         << std::fixed << std::setprecision(3) << t << ","
                                         << std::setprecision(6) << cycle << ","
                                         << listed_eval.section_values[i].section_name << ","
                                         << std::setprecision(10) << listed_eval.section_values[i].value << ","
                                         << std::setprecision(10) << listed_eval.total_value << ","
                                         << std::setprecision(10) << listed_bias_scale << ","
                                         << std::setprecision(10) << listed_bias << "\n";
                    }
                }
            }

            if (phase2_softwall_runtime_scoped_mode) {
                RuntimeObservables obs;
                obs.step = step;
                obs.time = t;
                obs.cycle = cycle;
                obs.temperature = temperature;
                obs.potential = potential;
                obs.lj = lj;
                obs.pme = pme;
                obs.pressure = pressure;
                obs.density = density;
                obs.dv_dt = dv_dt;
                obs.nb14_lj = nb14_lj;
                obs.nb14_ee = nb14_ee;
                obs.bond = bond;
                obs.angle = angle;
                obs.dihedral = dihedral;

                const SoftWallEvalResult softwall_eval =
                    evaluate_softwall_runtime(softwall_runtime_cfg, obs);
                const double softwall_bias = softwall_bias_scale * softwall_eval.total_value;
                potential += softwall_bias;
                pme += softwall_bias;

                std::ostringstream os;
                os << "PHASE2-SOFTWALL-RUNTIME: step=" << step
                   << " total=" << softwall_eval.total_value
                   << " bias=" << softwall_bias;
                log(os.str());

                if (softwall_trace_csv) {
                    for (size_t i = 0; i < softwall_eval.section_values.size(); ++i) {
                        softwall_trace_csv << step << ","
                                          << std::fixed << std::setprecision(3) << t << ","
                                          << std::setprecision(6) << cycle << ","
                                          << softwall_eval.section_values[i].section_name << ","
                                          << std::setprecision(10) << softwall_eval.section_values[i].value << ","
                                          << std::setprecision(10) << softwall_eval.total_value << ","
                                          << std::setprecision(10) << softwall_bias_scale << ","
                                          << std::setprecision(10) << softwall_bias << "\n";
                    }
                }
            }

            if (phase2_cv_combine_runtime_scoped_mode) {
                RuntimeObservables obs;
                obs.step = step;
                obs.time = t;
                obs.cycle = cycle;
                obs.temperature = temperature;
                obs.potential = potential;
                obs.lj = lj;
                obs.pme = pme;
                obs.pressure = pressure;
                obs.density = density;
                obs.dv_dt = dv_dt;
                obs.nb14_lj = nb14_lj;
                obs.nb14_ee = nb14_ee;
                obs.bond = bond;
                obs.angle = angle;
                obs.dihedral = dihedral;

                const CVCombineEvalResult cv_eval = evaluate_cv_combine(cv_runtime_cfg, obs);
                const double cv_bias = cv_bias_scale * cv_eval.total_value;
                potential += cv_bias;
                pme += cv_bias;

                std::ostringstream os;
                os << "PHASE2-CV-COMBINE-RUNTIME: step=" << step
                   << " total=" << cv_eval.total_value
                   << " bias=" << cv_bias;
                log(os.str());

                if (cv_trace_csv) {
                    for (size_t i = 0; i < cv_eval.section_values.size(); ++i) {
                        cv_trace_csv << step << ","
                                     << std::fixed << std::setprecision(3) << t << ","
                                     << std::setprecision(6) << cycle << ","
                                     << cv_eval.section_values[i].section_name << ","
                                     << std::setprecision(10) << cv_eval.section_values[i].value << ","
                                     << std::setprecision(10) << cv_eval.total_value << ","
                                     << std::setprecision(10) << cv_bias_scale << ","
                                     << std::setprecision(10) << cv_bias << "\n";
                    }
                }
            }

            out << std::setw(15) << step
                << std::setw(17) << std::fixed << std::setprecision(3) << t
                << std::setw(16) << std::fixed << std::setprecision(2) << temperature
                << std::setw(16) << std::fixed << std::setprecision(2) << potential
                << std::setw(16) << std::fixed << std::setprecision(2) << lj
                << std::setw(16) << std::fixed << std::setprecision(2) << pme
                << std::setw(16) << std::fixed << std::setprecision(2) << nb14_lj
                << std::setw(16) << std::fixed << std::setprecision(2) << nb14_ee
                << std::setw(16) << std::fixed << std::setprecision(2) << bond
                << std::setw(16) << std::fixed << std::setprecision(2) << angle
                << std::setw(16) << std::fixed << std::setprecision(2) << dihedral
                << std::setw(16) << std::fixed << std::setprecision(4) << density
                << std::setw(16) << std::fixed << std::setprecision(2) << pressure
                << std::setw(16) << std::fixed << std::setprecision(2) << dv_dt
                << " \n";
#ifdef __linux__
            if (phase2_plugin_runtime_scoped_mode) {
                run_plugin_mdout_print(&plugin_runtime);
            }
#endif
        }

#ifdef __linux__
        if (phase2_plugin_runtime_scoped_mode) {
            std::ostringstream os;
            os << "PHASE2-PLUGIN-RUNTIME: callback_counts after_initial="
               << plugin_runtime.after_initial_calls
               << " calculate_force=" << plugin_runtime.calculate_force_calls
               << " mdout_print=" << plugin_runtime.mdout_print_calls;
            log(os.str());
        }
#endif

        log("CPU minidyn mdout generation completed.");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 4;
    }
}
