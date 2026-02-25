#include "combine.h"

#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace {

bool Is_Allowed_Combine_Expr_Char(char c)
{
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || std::isspace(uc)) return true;
    switch (c)
    {
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

std::vector<std::string> Extract_Identifier_Tokens(const std::string& expr)
{
    std::vector<std::string> tokens;
    for (size_t i = 0; i < expr.size();)
    {
        const unsigned char c = static_cast<unsigned char>(expr[i]);
        if (std::isalpha(c) || expr[i] == '_')
        {
            size_t j = i + 1;
            while (j < expr.size())
            {
                const unsigned char cj = static_cast<unsigned char>(expr[j]);
                if (!std::isalnum(cj) && expr[j] != '_') break;
                ++j;
            }
            tokens.push_back(expr.substr(i, j - i));
            i = j;
        }
        else
        {
            ++i;
        }
    }
    return tokens;
}

bool Is_Allowed_Combine_Builtin(const std::string& token)
{
    static const std::set<std::string> kAllowed = {
        "abs", "acos", "asin", "atan", "ceil", "cos", "exp", "fabs", "floor",
        "log", "max", "min", "pow", "sin", "sqrt", "tan",
    };
    return kAllowed.count(token) > 0;
}

class CombineExpressionParser
{
public:
    CombineExpressionParser(const std::string& expr, const std::map<std::string, double>& vars)
        : expr_(expr), vars_(vars), pos_(0) {}

    double Parse()
    {
        const double v = ParseExpression();
        SkipSpace();
        if (pos_ != expr_.size())
        {
            throw std::runtime_error("unexpected token near `" + expr_.substr(pos_) + "`");
        }
        if (!std::isfinite(v))
        {
            throw std::runtime_error("non-finite cv_combine mainpath result");
        }
        return v;
    }

private:
    double ParseExpression()
    {
        double v = ParseTerm();
        while (true)
        {
            SkipSpace();
            if (Match('+'))
            {
                v += ParseTerm();
            }
            else if (Match('-'))
            {
                v -= ParseTerm();
            }
            else
            {
                return v;
            }
        }
    }

    double ParseTerm()
    {
        double v = ParseFactor();
        while (true)
        {
            SkipSpace();
            if (Match('*'))
            {
                v *= ParseFactor();
            }
            else if (Match('/'))
            {
                const double denom = ParseFactor();
                if (std::fabs(denom) < 1.0e-20)
                {
                    throw std::runtime_error("division by zero in cv_combine mainpath expression");
                }
                v /= denom;
            }
            else
            {
                return v;
            }
        }
    }

    double ParseFactor()
    {
        SkipSpace();
        if (Match('+')) return ParseFactor();
        if (Match('-')) return -ParseFactor();
        if (Match('('))
        {
            const double v = ParseExpression();
            SkipSpace();
            if (!Match(')'))
            {
                throw std::runtime_error("missing `)` in cv_combine mainpath expression");
            }
            return v;
        }
        if (pos_ < expr_.size())
        {
            const char c = expr_[pos_];
            const unsigned char uc = static_cast<unsigned char>(c);
            if (std::isdigit(uc) || c == '.')
            {
                return ParseNumber();
            }
            if (std::isalpha(uc) || c == '_')
            {
                return ParseIdentifierOrFunction();
            }
        }
        throw std::runtime_error("unexpected token in cv_combine mainpath expression");
    }

    double ParseNumber()
    {
        const char* start = expr_.c_str() + pos_;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end == start)
        {
            throw std::runtime_error("failed to parse number in cv_combine mainpath expression");
        }
        pos_ = static_cast<size_t>(end - expr_.c_str());
        if (!std::isfinite(v))
        {
            throw std::runtime_error("non-finite numeric literal in cv_combine mainpath expression");
        }
        return v;
    }

    std::string ParseIdentifier()
    {
        const size_t begin = pos_;
        ++pos_;
        while (pos_ < expr_.size())
        {
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
        std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        SkipSpace();
        if (Match('('))
        {
            std::vector<double> args;
            SkipSpace();
            if (!Match(')'))
            {
                while (true)
                {
                    args.push_back(ParseExpression());
                    SkipSpace();
                    if (Match(')')) break;
                    if (!Match(','))
                    {
                        throw std::runtime_error("expected `,` or `)` in function call");
                    }
                }
            }
            return EvalBuiltin(lower_token, args);
        }

        if (lower_token == "pi") return 3.14159265358979323846;
        if (lower_token == "e") return 2.71828182845904523536;
        const auto it = vars_.find(token);
        if (it == vars_.end())
        {
            throw std::runtime_error("unknown identifier `" + token + "` in cv_combine mainpath expression");
        }
        return it->second;
    }

    double EvalBuiltin(const std::string& name, const std::vector<double>& args)
    {
        auto check_argc = [&](size_t n)
        {
            if (args.size() != n)
            {
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
        throw std::runtime_error("unsupported function `" + name + "` in cv_combine mainpath expression");
    }

    void SkipSpace()
    {
        while (pos_ < expr_.size() && std::isspace(static_cast<unsigned char>(expr_[pos_])))
        {
            ++pos_;
        }
    }

    bool Match(char c)
    {
        if (pos_ < expr_.size() && expr_[pos_] == c)
        {
            ++pos_;
            return true;
        }
        return false;
    }

    const std::string& expr_;
    const std::map<std::string, double>& vars_;
    size_t pos_;
};

double Evaluate_Combine_Expression(const std::string& expr, const std::map<std::string, double>& vars)
{
    CombineExpressionParser parser(expr, vars);
    return parser.Parse();
}

}  // namespace

#ifdef SPONGE_CPU_PHASE1
struct CV_PHASE2_SCOPED_OBSERVABLE : public COLLECTIVE_VARIABLE_PROTOTYPE
{
    float base_value = 0.0f;
    float step_scale = 0.0f;
    float cycle_scale = 0.0f;

    void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
    {
        float* p_base = manager->Ask_For_Float_Parameter(module_name, "base_value", 1, 2, false, 0.0f);
        float* p_step = manager->Ask_For_Float_Parameter(module_name, "step_scale", 1, 2, false, 0.0f);
        float* p_cycle = manager->Ask_For_Float_Parameter(module_name, "cycle_scale", 1, 2, false, 0.0f);
        base_value = p_base[0];
        step_scale = p_step[0];
        cycle_scale = p_cycle[0];
        free(p_base);
        free(p_step);
        free(p_cycle);
        Super_Initial(manager, atom_numbers, module_name);
        if (manager->controller != NULL)
        {
            manager->controller->printf(
                "PHASE2-CV-COMBINE-MAINPATH: observable cv=%s base=%f step_scale=%f cycle_scale=%f\n",
                module_name, base_value, step_scale, cycle_scale);
        }
    }

    void Compute(int atom_numbers, UNSIGNED_INT_VECTOR*, VECTOR, VECTOR*, VECTOR, int need, int step)
    {
        need = Check_Whether_Computed_At_This_Step(step, need);
        if (!need)
        {
            Record_Update_Step_Of_Fast_Computing_CV(step, need);
            return;
        }
        const float cycle = static_cast<float>(step) * 0.01f;
        value = base_value + step_scale * static_cast<float>(step) + cycle_scale * std::sin(cycle);
        cudaMemcpyAsync(d_value, &value, sizeof(float), cudaMemcpyHostToDevice, this->cuda_stream);
        cudaMemset(crd_grads, 0, sizeof(VECTOR) * atom_numbers);
        cudaMemset(box_grads, 0, sizeof(VECTOR));
        Record_Update_Step_Of_Fast_Computing_CV(step, need);
    }
};

REGISTER_CV_STRUCTURE(CV_PHASE2_SCOPED_OBSERVABLE, "cpu_phase2_scoped_observable", 0);
#endif

REGISTER_CV_STRUCTURE(CV_COMBINE, "combination", 0);

void CV_COMBINE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
    cv_lists = manager->Ask_For_CV(module_name, -1, 0, 2);
    if (!manager->Command_Exist(module_name, "function"))
    {
        manager->Throw_SPONGE_Error(spongeErrorMissingCommand, "CV_COMBINE::Initial",
            "Reason:\n\tNeed to specify the function of the CV combination");
    }

    void** temp_ptr;
    Malloc_Safely((void**)&temp_ptr, sizeof(void*) * cv_lists.size());
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->d_value;
    }
    Cuda_Malloc_And_Copy_Safely((void**)&d_cv_values, temp_ptr, sizeof(float*) * cv_lists.size());
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->crd_grads;
    }
    Cuda_Malloc_And_Copy_Safely((void**)&cv_crd_grads, temp_ptr, sizeof(VECTOR*) * cv_lists.size());
    for (int i = 0; i < cv_lists.size(); i++)
    {
        temp_ptr[i] = cv_lists[i]->box_grads;
    }
    Cuda_Malloc_And_Copy_Safely((void**)&cv_box_grads, temp_ptr, sizeof(VECTOR*) * cv_lists.size());
    free(temp_ptr);
    Cuda_Malloc_Safely((void**)&df_dcv, sizeof(float) * cv_lists.size());

    cpu_function_expr = manager->Original_Command(module_name, "function");
    Super_Initial(manager, atom_numbers, module_name);

#ifdef SPONGE_CPU_PHASE1
    cpu_phase2_mainpath_mode =
        manager->controller != NULL &&
        manager->controller->Command_Exist("phase2_cv_combine_mainpath_mode") &&
        manager->controller->Get_Bool("phase2_cv_combine_mainpath_mode", "CV_COMBINE::Initial");
    if (cpu_phase2_mainpath_mode)
    {
        cpu_atom_numbers = atom_numbers;
        cpu_cv_names.clear();
        std::set<std::string> allowed_identifiers;
        for (size_t i = 0; i < cv_lists.size(); ++i)
        {
            cpu_cv_names.push_back(cv_lists[i]->module_name);
            allowed_identifiers.insert(cv_lists[i]->module_name);
        }
        for (size_t i = 0; i < cpu_function_expr.size(); ++i)
        {
            if (!Is_Allowed_Combine_Expr_Char(cpu_function_expr[i]))
            {
                manager->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CV_COMBINE::Initial",
                    "Reason:\n\tcv_combine mainpath expression has unsupported character\n"
                    "\tTrack: PHASE2-REENABLE-CV-COMBINE");
            }
        }
        const std::vector<std::string> identifiers = Extract_Identifier_Tokens(cpu_function_expr);
        for (size_t i = 0; i < identifiers.size(); ++i)
        {
            if (allowed_identifiers.count(identifiers[i]) > 0) continue;
            std::string lower = identifiers[i];
            std::transform(lower.begin(), lower.end(), lower.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower == "pi" || lower == "e" || Is_Allowed_Combine_Builtin(lower)) continue;
            std::string reason = "Reason:\n\tcv_combine mainpath unknown identifier `";
            reason += identifiers[i];
            reason += "`\n\tTrack: PHASE2-REENABLE-CV-COMBINE";
            manager->Throw_SPONGE_Error(spongeErrorBadFileFormat, "CV_COMBINE::Initial", reason.c_str());
        }
        if (manager->controller != NULL)
        {
            manager->controller->printf(
                "PHASE2-CV-COMBINE-MAINPATH: scoped runtime init section=%s terms=%d function=%s\n",
                module_name, static_cast<int>(cv_lists.size()), cpu_function_expr.c_str());
        }
        return;
    }
    manager->Throw_SPONGE_Error(
        spongeErrorNotImplemented,
        "CV_COMBINE::Initial",
        "Reason:\n\tFeature GATE-JIT-CV-COMBINE is gated in CPU Phase 1.\n"
        "\tSet phase2_cv_combine_mainpath_mode = 1 for scoped main-path runtime coverage.\n"
        "\tIssue: PHASE2-REENABLE-CV-COMBINE");
#endif

    std::string function_code = string_replace(cpu_function_expr, ")(", ", ");
    std::string sadf = "SADfloat<" + std::to_string(cv_lists.size()) + ">";
    std::string source_code = R"JIT(
#include "common.h"
extern "C" __global__ void cv_combine_first_step( const float **CV_values, const VECTOR **cv_box_grads,
float* out_value, float* out_df_dcv, VECTOR* box_grads)
{
    %PARM_DEC%
    %sadf% %NAME% = %FUNC_CODE%;
    out_value[0] = %NAME%.val;
    VECTOR local_box_grads = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < %N%; i++)
    {
        out_df_dcv[i] = %NAME%.dval[i];
        local_box_grads = local_box_grads + %NAME%.dval[i] * cv_box_grads[i][0];
    }
    box_grads[0] = local_box_grads;
}
)JIT";
    StringVector cv_names;
    for (auto cv : cv_lists)
    {
        cv_names.push_back(cv->module_name);
    }
    std::string endl = "\n    ";
    std::string PARM_DEC = string_join(sadf + " %0%(CV_values[%INDEX%][0], %INDEX%);", endl, { cv_names });
    source_code = string_format(source_code, { {"sadf", sadf}, {"PARM_DEC", PARM_DEC}, {"FUNC_CODE", function_code},
        {"NAME", module_name}, {"N", std::to_string(cv_lists.size())} });

    first_step.Compile(source_code);
    if (!first_step.error_reason.empty())
    {
        first_step.error_reason = "Reason:\n" + first_step.error_reason;
        manager->Throw_SPONGE_Error(spongeErrorMallocFailed, "CV_COMBINE::Initial (first step)", first_step.error_reason.c_str());
    }

    source_code = string_format(R"JIT(
#include "common.h"
extern "C" __global__ void cv_combine_second_step(const int atom_numbers, const VECTOR **CV_crd_grads, const float* out_df_dcv, VECTOR* crd_grads)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR local_crd_grads = {0.0f, 0.0f, 0.0f};
        %ADD_GRADS%
        crd_grads[atom_i] = local_crd_grads;
    }
}
)JIT", { {"ADD_GRADS", string_join("local_crd_grads = local_crd_grads + out_df_dcv[%INDEX%] * CV_crd_grads[%INDEX%][atom_i];",
    endl, { cv_names }) } });
    second_step.Compile(source_code);
    if (!second_step.error_reason.empty())
    {
        second_step.error_reason = "Reason:\n" + second_step.error_reason;
        manager->Throw_SPONGE_Error(spongeErrorMallocFailed, "CV_COMBINE::Initial (second step)", second_step.error_reason.c_str());
    }
}

void CV_COMBINE::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need)
    {
        for (auto cv : cv_lists)
        {
            cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, CV_NEED_ALL, step);
        }
        for (auto cv : cv_lists)
        {
            cudaStreamSynchronize(cv->cuda_stream);
        }

#ifdef SPONGE_CPU_PHASE1
        if (cpu_phase2_mainpath_mode)
        {
            const int n = static_cast<int>(cv_lists.size());
            std::vector<double> values;
            values.reserve(n);
            for (int i = 0; i < n; ++i)
            {
                values.push_back(cv_lists[i]->value);
            }

            auto eval_with_values = [&](const std::vector<double>& in_values) -> double
            {
                std::map<std::string, double> vars;
                for (int i = 0; i < n; ++i)
                {
                    vars[cpu_cv_names[i]] = in_values[i];
                }
                return Evaluate_Combine_Expression(cpu_function_expr, vars);
            };

            std::vector<double> dfdx(n, 0.0);
            try
            {
                const double f0 = eval_with_values(values);
                value = static_cast<float>(f0);
                for (int i = 0; i < n; ++i)
                {
                    const double eps = 1.0e-6 * (1.0 + std::fabs(values[i]));
                    std::vector<double> vp = values;
                    std::vector<double> vm = values;
                    vp[i] += eps;
                    vm[i] -= eps;
                    const double fp = eval_with_values(vp);
                    const double fm = eval_with_values(vm);
                    dfdx[i] = (fp - fm) / (2.0 * eps);
                }
            }
            catch (const std::exception&)
            {
                value = NAN;
                std::fill(dfdx.begin(), dfdx.end(), 0.0);
            }

            for (int i = 0; i < n; ++i)
            {
                const float dval = static_cast<float>(dfdx[i]);
                cudaMemcpyAsync(&df_dcv[i], &dval, sizeof(float), cudaMemcpyHostToDevice, this->cuda_stream);
            }
            cudaMemset(crd_grads, 0, sizeof(VECTOR) * cpu_atom_numbers);
            cudaMemset(box_grads, 0, sizeof(VECTOR));
            for (int i = 0; i < n; ++i)
            {
                const float dval = static_cast<float>(dfdx[i]);
                for (int atom_i = 0; atom_i < cpu_atom_numbers; ++atom_i)
                {
                    crd_grads[atom_i].x += dval * cv_lists[i]->crd_grads[atom_i].x;
                    crd_grads[atom_i].y += dval * cv_lists[i]->crd_grads[atom_i].y;
                    crd_grads[atom_i].z += dval * cv_lists[i]->crd_grads[atom_i].z;
                }
                box_grads[0].x += dval * cv_lists[i]->box_grads[0].x;
                box_grads[0].y += dval * cv_lists[i]->box_grads[0].y;
                box_grads[0].z += dval * cv_lists[i]->box_grads[0].z;
            }
            cudaMemcpyAsync(d_value, &value, sizeof(float), cudaMemcpyHostToDevice, this->cuda_stream);
            Record_Update_Step_Of_Fast_Computing_CV(step, need);
            return;
        }
#endif

        CUresult res1 = first_step({ 1, 1, 1 }, { 1, 1, 1 }, this->cuda_stream, 0, { &d_cv_values, &cv_box_grads, &d_value, &df_dcv, &box_grads });
        CUresult res2 = second_step({ atom_numbers + 1023u / 1024u, 1, 1 }, { 1024, 1, 1 }, this->cuda_stream, 0,
            { &atom_numbers, &cv_crd_grads, &df_dcv, &crd_grads });
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS)
        {
            value = NAN;
            cudaMemcpyAsync(d_value, &value, sizeof(float), cudaMemcpyHostToDevice, this->cuda_stream);
        }
        else
        {
            cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
        }
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
