#include "main.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define SUBPACKAGE_HINT "SPONGE, for normal molecular dynamics simulations"

CONTROLLER controller;
MD_INFORMATION md_info;
MIDDLE_Langevin_INFORMATION middle_langevin;
Langevin_MD_INFORMATION langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
LISTED_FORCES listed_forces;
BOND bond;
ANGLE angle;
UREY_BRADLEY urey_bradley;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
CMAP cmap;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
SOLVENT_LENNARD_JONES solvent_lj;
Particle_Mesh_Ewald pme;
LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SIMPLE_CONSTRAIN simple_constrain;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
MC_BAROSTAT_INFORMATION mc_baro;
BERENDSEN_BAROSTAT_INFORMATION bd_baro;
ANDERSEN_BAROSTAT_INFORMATION ad_baro;
LJ_SOFT_CORE lj_soft;
PAIRWISE_FORCE pairwise_force;
BOND_SOFT bond_soft;
SITS_INFORMATION sits;
DIHEDRAL sits_dihedral;
RESTRAIN_CV restrain_cv;
STEER_CV steer_cv;
COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
META1D meta;
HARD_WALL hard_wall;
SOFT_WALLS soft_walls;
SPONGE_PLUGIN plugin;

static bool Pressure_Split_Debug_On()
{
    const char* value = std::getenv("SPONGE_DEBUG_PRESSURE_SPLIT");
    return value != NULL && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void Pressure_Split_Debug_Print(const char* stage, int step, int need_pressure, const float* d_pressure)
{
    if (!Pressure_Split_Debug_On() || need_pressure <= 0 || d_pressure == NULL)
    {
        return;
    }
    float pressure_internal = 0.0f;
    cudaMemcpy(&pressure_internal, d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
    std::fprintf(
        stderr,
        "DEBUG_PRESSURE_SPLIT step=%d stage=%s pressure_internal=%e\n",
        step,
        stage,
        (double)pressure_internal);
}

#ifdef SPONGE_CPU_PHASE1
static bool Cpu_Debug_Env_On(const char* name)
{
    const char* value = std::getenv(name);
    return value != NULL && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static int Cpu_Debug_Trace_Atom()
{
    static int inited = 0;
    static int atom = -1;
    if (!inited)
    {
        const char* value = std::getenv("SPONGE_CPU_DEBUG_TRACE_ATOM");
        if (value != NULL && value[0] != '\0')
        {
            atom = std::atoi(value);
        }
        inited = 1;
    }
    return atom;
}

static void Cpu_Debug_Check_Vector_Finite(
    CONTROLLER* controller,
    const char* env_name,
    const char* stage,
    VECTOR* d_vec,
    int atom_numbers)
{
    if (!Cpu_Debug_Env_On(env_name) || d_vec == NULL || atom_numbers <= 0)
    {
        return;
    }
    static VECTOR* h_vec = NULL;
    static int h_vec_cap = 0;
    if (atom_numbers > h_vec_cap)
    {
        free(h_vec);
        h_vec = (VECTOR*)malloc(sizeof(VECTOR) * atom_numbers);
        h_vec_cap = atom_numbers;
    }
    if (h_vec == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed,
            "Cpu_Debug_Check_Vector_Finite",
            "Reason:\n\tfailed to allocate host debug buffer\n");
    }
    cudaMemcpy(h_vec, d_vec, sizeof(VECTOR) * atom_numbers, cudaMemcpyDeviceToHost);
    double max_abs = 0.0;
    double max_norm2 = -1.0;
    int max_norm_atom = -1;
    for (int i = 0; i < atom_numbers; ++i)
    {
        const VECTOR v = h_vec[i];
        const double ax = std::fabs((double)v.x);
        const double ay = std::fabs((double)v.y);
        const double az = std::fabs((double)v.z);
        if (ax > max_abs) max_abs = ax;
        if (ay > max_abs) max_abs = ay;
        if (az > max_abs) max_abs = az;
        const double norm2 = (double)v.x * (double)v.x + (double)v.y * (double)v.y + (double)v.z * (double)v.z;
        if (norm2 > max_norm2)
        {
            max_norm2 = norm2;
            max_norm_atom = i;
        }
        if (!(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)))
        {
            char reason[512];
            std::snprintf(
                reason, sizeof(reason),
                "Reason:\n\tnon-finite vector found at stage '%s', atom %d, value=(%e,%e,%e)\n",
                stage, i, v.x, v.y, v.z);
            controller->Throw_SPONGE_Error(spongeErrorSimulationBreakDown, "Cpu_Debug_Check_Vector_Finite", reason);
        }
    }
    if (Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_VECTOR_STATS"))
    {
        std::fprintf(
            stderr,
            "CPU_DEBUG_VECTOR stage=%s max_abs=%e max_norm=%e max_norm_atom=%d\n",
            stage,
            max_abs,
            std::sqrt(max_norm2 < 0.0 ? 0.0 : max_norm2),
            max_norm_atom);
    }
    const int trace_atom = Cpu_Debug_Trace_Atom();
    if (trace_atom >= 0 && trace_atom < atom_numbers)
    {
        const VECTOR v = h_vec[trace_atom];
        std::fprintf(
            stderr,
            "CPU_DEBUG_TRACE stage=%s atom=%d value=(%e,%e,%e) norm=%e\n",
            stage,
            trace_atom,
            v.x,
            v.y,
            v.z,
            std::sqrt((double)v.x * (double)v.x + (double)v.y * (double)v.y + (double)v.z * (double)v.z));
    }
}

static __forceinline__ float Cpu_Debug_Minimum_Image(float delta, float box_length)
{
    if (box_length <= 0.0f)
    {
        return delta;
    }
    return delta - roundf(delta / box_length) * box_length;
}

static void Cpu_Debug_Report_Constraint_Error(
    CONTROLLER* controller,
    const CONSTRAIN* constrain,
    VECTOR* d_crd,
    int atom_numbers,
    VECTOR box_length,
    int step)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_CONSTRAIN_ERR") ||
        constrain == NULL ||
        d_crd == NULL ||
        atom_numbers <= 0 ||
        constrain->constrain_pair_numbers <= 0 ||
        constrain->h_constrain_pair == NULL)
    {
        return;
    }

    static VECTOR* h_crd = NULL;
    static int h_crd_cap = 0;
    if (atom_numbers > h_crd_cap)
    {
        free(h_crd);
        h_crd = (VECTOR*)malloc(sizeof(VECTOR) * atom_numbers);
        h_crd_cap = atom_numbers;
    }
    if (h_crd == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed,
            "Cpu_Debug_Report_Constraint_Error",
            "Reason:\n\tfailed to allocate host coordinate buffer\n");
    }
    cudaMemcpy(h_crd, d_crd, sizeof(VECTOR) * atom_numbers, cudaMemcpyDeviceToHost);

    float max_abs_err = -1.0f;
    int worst_pair = -1;
    int worst_i = -1;
    int worst_j = -1;
    float worst_target = 0.0f;
    float worst_dist = 0.0f;

    for (int i = 0; i < constrain->constrain_pair_numbers; ++i)
    {
        const CONSTRAIN_PAIR& cp = constrain->h_constrain_pair[i];
        if (cp.atom_i_serial < 0 || cp.atom_i_serial >= atom_numbers ||
            cp.atom_j_serial < 0 || cp.atom_j_serial >= atom_numbers)
        {
            continue;
        }
        float dx = h_crd[cp.atom_i_serial].x - h_crd[cp.atom_j_serial].x;
        float dy = h_crd[cp.atom_i_serial].y - h_crd[cp.atom_j_serial].y;
        float dz = h_crd[cp.atom_i_serial].z - h_crd[cp.atom_j_serial].z;

        dx = Cpu_Debug_Minimum_Image(dx, box_length.x);
        dy = Cpu_Debug_Minimum_Image(dy, box_length.y);
        dz = Cpu_Debug_Minimum_Image(dz, box_length.z);

        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(dist))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorSimulationBreakDown,
                "Cpu_Debug_Report_Constraint_Error",
                "Reason:\n\tnon-finite distance in constrain pair diagnostics\n");
        }
        const float abs_err = fabsf(dist - cp.constant_r);
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            worst_pair = i;
            worst_i = cp.atom_i_serial;
            worst_j = cp.atom_j_serial;
            worst_target = cp.constant_r;
            worst_dist = dist;
        }
    }

    std::fprintf(
        stderr,
        "CPU_DEBUG_CONSTRAIN step=%d max_abs_err=%e pair_idx=%d atom_i=%d atom_j=%d target=%e dist=%e\n",
        step,
        max_abs_err,
        worst_pair,
        worst_i,
        worst_j,
        worst_target,
        worst_dist);
}

static void Cpu_Debug_Report_Bond_Error(
    CONTROLLER* controller,
    const BOND* bond,
    VECTOR* d_crd,
    int atom_numbers,
    VECTOR box_length,
    int step)
{
    if (!Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_BOND_ERR") ||
        bond == NULL ||
        d_crd == NULL ||
        atom_numbers <= 0 ||
        bond->bond_numbers <= 0 ||
        bond->h_atom_a == NULL ||
        bond->h_atom_b == NULL ||
        bond->h_r0 == NULL)
    {
        return;
    }

    static VECTOR* h_crd = NULL;
    static int h_crd_cap = 0;
    if (atom_numbers > h_crd_cap)
    {
        free(h_crd);
        h_crd = (VECTOR*)malloc(sizeof(VECTOR) * atom_numbers);
        h_crd_cap = atom_numbers;
    }
    if (h_crd == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed,
            "Cpu_Debug_Report_Bond_Error",
            "Reason:\n\tfailed to allocate host coordinate buffer\n");
    }
    cudaMemcpy(h_crd, d_crd, sizeof(VECTOR) * atom_numbers, cudaMemcpyDeviceToHost);

    float max_abs_err = -1.0f;
    int worst_bond = -1;
    int worst_i = -1;
    int worst_j = -1;
    float worst_target = 0.0f;
    float worst_dist = 0.0f;

    for (int i = 0; i < bond->bond_numbers; ++i)
    {
        const int ai = bond->h_atom_a[i];
        const int aj = bond->h_atom_b[i];
        if (ai < 0 || ai >= atom_numbers || aj < 0 || aj >= atom_numbers)
        {
            continue;
        }
        float dx = h_crd[ai].x - h_crd[aj].x;
        float dy = h_crd[ai].y - h_crd[aj].y;
        float dz = h_crd[ai].z - h_crd[aj].z;
        dx = Cpu_Debug_Minimum_Image(dx, box_length.x);
        dy = Cpu_Debug_Minimum_Image(dy, box_length.y);
        dz = Cpu_Debug_Minimum_Image(dz, box_length.z);
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(dist))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorSimulationBreakDown,
                "Cpu_Debug_Report_Bond_Error",
                "Reason:\n\tnon-finite distance in bond diagnostics\n");
        }
        const float abs_err = fabsf(dist - bond->h_r0[i]);
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            worst_bond = i;
            worst_i = ai;
            worst_j = aj;
            worst_target = bond->h_r0[i];
            worst_dist = dist;
        }
    }

    std::fprintf(
        stderr,
        "CPU_DEBUG_BOND step=%d max_abs_err=%e bond_idx=%d atom_i=%d atom_j=%d target=%e dist=%e\n",
        step,
        max_abs_err,
        worst_bond,
        worst_i,
        worst_j,
        worst_target,
        worst_dist);
}

static void Cpu_Debug_Report_Scalar_Stats(
    CONTROLLER* controller,
    const char* env_name,
    const char* stage,
    const float* d_list,
    int element_numbers,
    int step)
{
    if (!Cpu_Debug_Env_On(env_name) || d_list == NULL || element_numbers <= 0)
    {
        return;
    }
    static float* h_list = NULL;
    static int h_list_cap = 0;
    if (element_numbers > h_list_cap)
    {
        free(h_list);
        h_list = (float*)malloc(sizeof(float) * element_numbers);
        h_list_cap = element_numbers;
    }
    if (h_list == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed,
            "Cpu_Debug_Report_Scalar_Stats",
            "Reason:\n\tfailed to allocate host scalar debug buffer\n");
    }
    cudaMemcpy(h_list, d_list, sizeof(float) * element_numbers, cudaMemcpyDeviceToHost);

    double sum = 0.0;
    double sumsq = 0.0;
    float minv = FLT_MAX;
    float maxv = -FLT_MAX;
    int nan_count = 0;
    for (int i = 0; i < element_numbers; ++i)
    {
        const float v = h_list[i];
        if (!std::isfinite(v))
        {
            ++nan_count;
            continue;
        }
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += (double)v;
        sumsq += (double)v * (double)v;
    }
    std::fprintf(
        stderr,
        "CPU_DEBUG_SCALAR step=%d stage=%s n=%d nan=%d min=%e max=%e sum=%e sumsq=%e\n",
        step,
        stage,
        element_numbers,
        nan_count,
        minv,
        maxv,
        sum,
        sumsq);
}

static void Cpu_Debug_Report_Virial_Stats(
    CONTROLLER* controller,
    const char* atom_stage,
    const char* sys_stage,
    int need_pressure,
    const float* d_atom_virial,
    int atom_numbers,
    const float* d_sys_virial,
    int step)
{
    if (need_pressure <= 0)
    {
        return;
    }
    Cpu_Debug_Report_Scalar_Stats(
        controller,
        "SPONGE_CPU_DEBUG_VIRIAL_STATS",
        atom_stage,
        d_atom_virial,
        atom_numbers,
        step);
    Cpu_Debug_Report_Scalar_Stats(
        controller,
        "SPONGE_CPU_DEBUG_VIRIAL_STATS",
        sys_stage,
        d_sys_virial,
        1,
        step);
}
#endif

int main(int argc, char *argv[])
{
    Main_Initial(argc, argv);
    for (md_info.sys.steps = 1; md_info.sys.steps <= md_info.sys.step_limit; md_info.sys.steps++)
    {
        Main_Calculate_Force();
        Main_Iteration();
        Main_Print();
    }

    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char* argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    cv_controller.Initial(&controller, &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    controller.Step_Print_Initial("potential", "%.2f");
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);

    if (md_info.mode >= md_info.NVT && !controller.Command_Exist("thermostat"))
    {
        controller.Throw_SPONGE_Error(spongeErrorMissingCommand, "Main_Initial", "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if  (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "middle_langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "langevin"))
    {
        langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "andersen_thermostat"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature, md_info.atom_numbers, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.sys.target_temperature);
    }

    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
        lj_soft.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
        pairwise_force.Initial(&controller);
        solvent_lj.Initial(&controller, &lj, &lj_soft, md_info.res.residue_numbers, md_info.res.h_res_start, md_info.res.h_res_end, md_info.mode >= md_info.NVT);
        pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff);
        sits.Initial(&controller, md_info.atom_numbers);
        if (sits.is_initialized)
        {
            sits_dihedral.Initial(&controller, "sits_dihedral");
        }
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);
    }
    else
    {
        LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
        CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
        if (controller.Command_Exist("gb", "in_file"))
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B, LJ_NOPBC.h_atom_LJ_type);
    }

    listed_forces.Initial(&controller, &md_info.sys.connectivity, &md_info.sys.connected_distance);

    bond.Initial(&controller, &md_info.sys.connectivity, &md_info.sys.connected_distance);
    bond_soft.Initial(&controller);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    cmap.Initial(&controller);

    hard_wall.Initial(&controller, md_info.sys.target_temperature, md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    restrain_cv.Initial(&controller, &cv_controller);
    steer_cv.Initial(&controller, &cv_controller);
    meta.Initial(&controller, &cv_controller);

    if (controller.Command_Exist("constrain_mode"))
    {    
        constrain.Initial_List(&controller, md_info.sys.connectivity, md_info.sys.connected_distance, md_info.h_mass);
        if (middle_langevin.is_initialized)
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, middle_langevin.exp_gamma, 0, md_info.h_mass, &md_info.sys.freedom);
        else
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, 1.0, md_info.mode == md_info.MINIMIZATION, md_info.h_mass, &md_info.sys.freedom);
        if (!(controller.Command_Exist("settle_disable") && controller.Get_Bool("settle_disable", "Main_Initial")))
        {
            settle.Initial(&controller, &constrain, md_info.h_mass);
        }
        if (controller.Command_Choice("constrain_mode", "simple_constrain"))
        {
            simple_constrain.Initial_Simple_Constrain(&controller, &constrain);
        }
        if (controller.Command_Choice("constrain_mode", "shake"))
        {
            shake.Initial_Simple_Constrain(&controller, &constrain);
        }
    }

    if (md_info.mode == md_info.NPT && !controller.Command_Exist("barostat"))
    {
        controller.Throw_SPONGE_Error(spongeErrorMissingCommand, "Main_Initial", "Reason:\n\tbarostat is required for NPT simulations\n");
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers, md_info.sys.target_pressure, md_info.sys.box_length, md_info.res.is_initialized);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "berendsen_barostat"))
    {
        bd_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "andersen_barostat"))
    {
        ad_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }

    vatom.Initial(&controller, &cv_controller, md_info.atom_numbers, md_info.no_direct_interaction_virtual_atom_numbers,
        cv_controller.cv_vatom_name, md_info.h_mass, &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
    md_info.mol.Initial(&controller);
    md_info.mol.Molecule_Crd_Map();
    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
        neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
            md_info.nb.d_excluded_numbers, neighbor_list.FORCED_UPDATE);
    }
    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();
    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_initial_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_initial_coordinate", md_info.crd, md_info.atom_numbers);
#endif
    controller.core_time.Start();
}

void Main_Clear()
{
    controller.core_time.Stop();
    controller.printf("Core Run Wall Time: %f seconds\n", controller.core_time.time);
    
    if (md_info.mode == md_info.MINIMIZATION)
    {
        controller.simulation_speed = md_info.sys.steps / controller.core_time.time;
        controller.printf("Core Run Speed: %f step/second\n", controller.simulation_speed);
    }
    else if (md_info.mode == md_info.RERUN)
    {
        controller.simulation_speed = md_info.sys.steps / controller.core_time.time;
        controller.printf("Core Run Speed: %f frame/second\n", controller.simulation_speed);
    }
    else
    {
        controller.simulation_speed = md_info.sys.steps * md_info.dt / CONSTANT_TIME_CONVERTION / controller.core_time.time * 86.4;
        controller.printf("Core Run Speed: %f ns/day\n", controller.simulation_speed);
    }
    fcloseall();

    if (controller.Command_Exist("end_pause"))
    {
        if (atoi(controller.Command("end_pause")) == 1)
        {
            printf("End Pause\n");
            getchar();
        }
    }
}

void Main_Calculate_Force()
{
    md_info.MD_Information_Crd_To_Uint_Crd();
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Report_Scalar_Stats(
        &controller,
        "SPONGE_CPU_DEBUG_CHARGE_STATS",
        "before_force_charge",
        md_info.d_charge,
        md_info.atom_numbers,
        md_info.sys.steps);
#endif
    if (md_info.mode == md_info.RERUN)
    {
        return;
    }
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    if (pme.is_initialized)
    {
        // PME direct/correction atom-energy work buffers are accumulated with atomicAdd in force kernels.
        // Reset them per MD step to avoid step-to-step carry-over in PME energy decomposition.
        cudaMemset(pme.d_direct_atom_energy, 0, sizeof(float) * md_info.atom_numbers);
        cudaMemset(pme.d_correction_atom_energy, 0, sizeof(float) * md_info.atom_numbers);
        cudaMemset(pme.d_direct_ene, 0, sizeof(float));
        cudaMemset(pme.d_correction_ene, 0, sizeof(float));
#ifdef SPONGE_CPU_PHASE1
        Cpu_Debug_Report_Scalar_Stats(
            &controller,
            "SPONGE_CPU_DEBUG_DIRECT_STATS",
            "after_reset_direct_atom_energy",
            pme.d_direct_atom_energy,
            md_info.atom_numbers,
            md_info.sys.steps);
#endif
    }
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps, &md_info.need_potential);
    bd_baro.Ask_For_Calculate_Pressure(md_info.sys.steps, &md_info.need_pressure);
    ad_baro.Ask_For_Calculate_Pressure(md_info.sys.steps, &md_info.need_pressure);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_need_pressure_decision_atom_virial",
        "after_need_pressure_decision_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif

    sits.Reset_Force_Energy(md_info.need_potential);
    
    if (sits.is_initialized)
    {
        sits_dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, sits.pw_select.select_force[0], sits.pw_select.select_atom_energy[0]);
        sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, &lj, md_info.frc,
            neighbor_list.d_nl, md_info.nb.cutoff, pme.beta, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
        sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, &lj_soft, md_info.frc,
            neighbor_list.d_nl, md_info.nb.cutoff, pme.beta, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    }
    else
    {
        lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
            neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
        lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
            neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    }
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_lj_and_lj_soft", md_info.frc, md_info.atom_numbers);
#endif
    solvent_lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers, md_info.res.residue_numbers, md_info.res.d_res_start, md_info.res.d_res_end, md_info.uint_crd, md_info.d_charge,
        md_info.frc, neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_nonbond_short_range", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_nonbond_short_range_atom_virial",
        "after_nonbond_short_range_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
    Cpu_Debug_Report_Scalar_Stats(
        &controller,
        "SPONGE_CPU_DEBUG_DIRECT_STATS",
        "after_nonbond_short_range_direct_atom_energy",
        pme.d_direct_atom_energy,
        md_info.atom_numbers,
        md_info.sys.steps);
#endif

    lj.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial,
        md_info.need_potential, md_info.sys.d_potential);
    lj_soft.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial, md_info.need_potential, md_info.sys.d_potential);
    
    pairwise_force.Compute_Force(neighbor_list.d_nl, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.nb.cutoff, pme.beta,
        md_info.d_charge, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Report_Scalar_Stats(
        &controller,
        "SPONGE_CPU_DEBUG_DIRECT_STATS",
        "after_pairwise_direct_atom_energy",
        pme.d_direct_atom_energy,
        md_info.atom_numbers,
        md_info.sys.steps);
#endif

    pme.PME_Excluded_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.d_charge,
        md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, md_info.frc, pme.d_correction_atom_energy);

    pme.PME_Reciprocal_Force_With_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.frc, 
        md_info.need_pressure, md_info.need_potential, md_info.sys.d_virial, md_info.sys.d_potential, md_info.sys.steps);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_pme_total", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_pme_total_atom_virial",
        "after_pme_total_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif

    LJ_NOPBC.LJ_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    CF_NOPBC.Coulomb_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.d_charge, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    gb.Get_Effective_Born_Radius(md_info.pbc.nopbc_crd);
    gb.GB_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.d_charge, md_info.frc, md_info.d_atom_energy);

    nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);

    bond.Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    bond_soft.Soft_Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    angle.Angle_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    improper.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    cmap.CMAP_Force_with_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    listed_forces.Compute_Force(md_info.crd, md_info.sys.box_length, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    soft_walls.Compute_Force(md_info.atom_numbers, md_info.crd, md_info.frc, md_info.d_atom_energy);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_bonded_and_walls", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_bonded_and_walls_atom_virial",
        "after_bonded_and_walls_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif

    plugin.Calculate_Force();

    restrain.Restraint(md_info.crd, md_info.sys.box_length, md_info.d_atom_energy, md_info.d_atom_virial, md_info.frc);
    restrain_cv.Restraint(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
                md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps, 
        md_info.sys.d_potential, md_info.sys.d_virial, md_info.frc, md_info.need_potential, md_info.need_pressure);
    steer_cv.Steer(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
                md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps, 
        md_info.sys.d_potential, md_info.sys.d_virial, md_info.frc, md_info.need_potential, md_info.need_pressure);
    meta.Do_Metadynamics(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
        md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length,
        md_info.sys.steps, md_info.need_potential, md_info.need_pressure, md_info.frc, md_info.sys.d_potential, md_info.sys.d_virial);

    sits.Update_And_Enhance(md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure, md_info.sys.d_virial, md_info.frc, 1.0 / (CONSTANT_kB * md_info.sys.target_temperature));
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_restraint_meta_sits", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_restraint_meta_sits_atom_virial",
        "after_restraint_meta_sits_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif

#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_force_accumulate", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_force_accumulate_atom_virial",
        "after_force_accumulate_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif

    vatom.Force_Redistribute(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_FORCE_FINITE", "after_vatom_force_redistribute", md_info.frc, md_info.atom_numbers);
    Cpu_Debug_Report_Virial_Stats(
        &controller,
        "after_vatom_force_redistribute_atom_virial",
        "after_vatom_force_redistribute_sys_virial",
        md_info.need_pressure,
        md_info.d_atom_virial,
        md_info.atom_numbers,
        md_info.sys.d_virial,
        md_info.sys.steps);
#endif
    md_info.Calculate_Pressure_And_Potential_If_Needed();
#ifdef SPONGE_CPU_PHASE1
    if (md_info.need_pressure > 0)
    {
        Cpu_Debug_Report_Scalar_Stats(
            &controller,
            "SPONGE_CPU_DEBUG_VIRIAL_STATS",
            "after_pressure_calc_sys_pressure",
            md_info.sys.d_pressure,
            1,
            md_info.sys.steps);
    }
#endif
}

void Main_Iteration()
{
    if (md_info.mode == md_info.RERUN)
    {
        vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
        md_info.MD_Information_Crd_To_Uint_Crd();
        return;
    }

    if (mc_baro.is_initialized && md_info.sys.steps % mc_baro.update_interval == 0)
    {
        mc_baro.energy_old = md_info.sys.h_potential;
        cudaMemcpy(mc_baro.frc_backup, md_info.frc, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mc_baro.crd_backup, md_info.crd, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);

        mc_baro.Volume_Change_Attempt(md_info.sys.box_length);

        if (mc_baro.scale_coordinate_by_molecule)
        {
            md_info.mol.Molecule_Crd_Map(mc_baro.crd_scale_factor);
        }
        else
        {
            mc_baro.Scale_Coordinate_Atomically(md_info.atom_numbers, md_info.crd);
        }

        Main_Box_Length_Change(mc_baro.crd_scale_factor);

        Main_Calculate_Force();
        mc_baro.energy_new = md_info.sys.h_potential;

        if (mc_baro.scale_coordinate_by_molecule)
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.mol.molecule_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);
        else
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.atom_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);

        if (mc_baro.couple_dimension != mc_baro.NO && mc_baro.couple_dimension != mc_baro.XYZ)
            mc_baro.extra_term -= mc_baro.surface_number * mc_baro.surface_tension * mc_baro.DeltaS;

        mc_baro.accept_possibility = mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility = expf(-mc_baro.accept_possibility / (CONSTANT_kB * md_info.sys.target_temperature));

        if (mc_baro.Check_MC_Barostat_Accept())
        {
            mc_baro.crd_scale_factor = 1.0 / mc_baro.crd_scale_factor;
            cudaMemcpy(md_info.crd, mc_baro.crd_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
            Main_Box_Length_Change(mc_baro.crd_scale_factor);
            neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
            cudaMemcpy(md_info.frc, mc_baro.frc_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
            md_info.sys.h_potential = mc_baro.energy_old;
        }
        if ((!mc_baro.reject && (mc_baro.newV > 1.331 * mc_baro.V0 || mc_baro.newV < 0.729 * mc_baro.V0)))
        {
            Main_Volume_Change_Largely();
            mc_baro.V0 = mc_baro.newV;
        }
        mc_baro.Delta_Box_Length_Max_Update();
    }
    
    settle.Remember_Last_Coordinates(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    simple_constrain.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    shake.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);

    if (md_info.mode == md_info.NVE)
    {
        md_info.nve.Leap_Frog();
    }
    else if (md_info.mode == md_info.MINIMIZATION)
    {
        md_info.min.Gradient_Descent();
    }
    else if (middle_langevin.is_initialized)
    {
        middle_langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.vel, md_info.acc, md_info.crd);
    }
    else if (langevin.is_initialized)
    {
        langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.crd, md_info.vel, md_info.acc);
    }
    else if (bd_thermo.is_initialized)
    {
        bd_thermo.Record_Temperature(md_info.sys.Get_Atom_Temperature(), md_info.sys.freedom);
        md_info.nve.Leap_Frog();
        bd_thermo.Scale_Velocity(md_info.atom_numbers, md_info.vel);
    }
    else if (ad_thermo.is_initialized)
    {
        if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
        {
            ad_thermo.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt);
            constrain.v_factor = FLT_MIN;
            constrain.x_factor = 0.5;
        }
        else
        {
            md_info.nve.Leap_Frog();
            constrain.v_factor = 1.0;
            constrain.x_factor = 1.0;
        }
    }
    else if (nhc.is_initialized)
    {
        nhc.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt, md_info.sys.Get_Total_Atom_Ek(), md_info.sys.freedom);
    }
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_integrator_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_integrator_coordinate", md_info.crd, md_info.atom_numbers);
#endif
    Pressure_Split_Debug_Print("before_settle", md_info.sys.steps, md_info.need_pressure, md_info.sys.d_pressure);

    settle.Do_SETTLE(md_info.d_mass, md_info.crd, md_info.sys.box_length, md_info.vel, md_info.need_pressure, md_info.sys.d_pressure);
    Pressure_Split_Debug_Print("after_settle", md_info.sys.steps, md_info.need_pressure, md_info.sys.d_pressure);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_settle_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_settle_coordinate", md_info.crd, md_info.atom_numbers);
#endif
    simple_constrain.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    Pressure_Split_Debug_Print("after_simple_constrain", md_info.sys.steps, md_info.need_pressure, md_info.sys.d_pressure);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_simple_constrain_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_simple_constrain_coordinate", md_info.crd, md_info.atom_numbers);
#endif
    shake.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    Pressure_Split_Debug_Print("after_shake", md_info.sys.steps, md_info.need_pressure, md_info.sys.d_pressure);
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_shake_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_shake_coordinate", md_info.crd, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_constrain_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_constrain_coordinate", md_info.crd, md_info.atom_numbers);
    Cpu_Debug_Report_Constraint_Error(&controller, &constrain, md_info.crd, md_info.atom_numbers, md_info.sys.box_length, md_info.sys.steps);
    Cpu_Debug_Report_Bond_Error(&controller, &bond, md_info.crd, md_info.atom_numbers, md_info.sys.box_length, md_info.sys.steps);
#endif
    

    if (bd_baro.is_initialized && md_info.sys.steps % bd_baro.update_interval == 0)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        bd_baro.crd_scale_factor = 1 - bd_baro.update_interval * bd_baro.compressibility * bd_baro.dt / bd_baro.taup / 3 * (md_info.sys.target_pressure - md_info.sys.h_pressure);
        if (bd_baro.stochastic_term)
        {
            bd_baro.crd_scale_factor += sqrtf(2 * CONSTANT_kB * md_info.sys.target_temperature* bd_baro.compressibility / bd_baro.taup / bd_baro.newV)
                / 3 * bd_baro.n(bd_baro.e);
            Scale_List((float*)md_info.vel, 1.0f / bd_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        }
        md_info.Scale_Position_To_Center(bd_baro.crd_scale_factor);
        Main_Volume_Change(bd_baro.crd_scale_factor);
        bd_baro.newV = md_info.sys.Get_Volume();
        if (bd_baro.newV > 1.331 * bd_baro.V0 || bd_baro.newV < 0.729 * bd_baro.V0)
        {
            Main_Volume_Change_Largely();
            bd_baro.V0 = bd_baro.newV;
        }
    }

    if (ad_baro.is_initialized && md_info.sys.steps % ad_baro.update_interval == 0)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        ad_baro.dV_dt += (md_info.sys.h_pressure - md_info.sys.target_pressure) * ad_baro.h_mass_inverse * md_info.dt * ad_baro.update_interval;
        ad_baro.Control_Velocity_Of_Box(md_info.dt * ad_baro.update_interval, md_info.sys.target_temperature);
        ad_baro.new_V = md_info.sys.Get_Volume() + ad_baro.dV_dt * md_info.dt * ad_baro.update_interval;
        ad_baro.crd_scale_factor = cbrt(ad_baro.new_V / md_info.sys.Get_Volume());
        md_info.Scale_Position_To_Center(ad_baro.crd_scale_factor);
        Scale_List((float*)md_info.vel, 1.0f / ad_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        Main_Volume_Change(ad_baro.crd_scale_factor);

        if (ad_baro.new_V > 1.331 * ad_baro.V0 || ad_baro.new_V < 0.729 * ad_baro.V0)
        {
            Main_Volume_Change_Largely();
            ad_baro.V0 = ad_baro.new_V;
        }
    }
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_VEL_FINITE", "after_barostat_velocity", md_info.vel, md_info.atom_numbers);
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_barostat_coordinate", md_info.crd, md_info.atom_numbers);
#endif

    if (md_info.mode == md_info.MINIMIZATION)
    {
        md_info.min.Check_Nan();
    }

    hard_wall.Reflect(md_info.atom_numbers, md_info.crd, md_info.vel);
    md_info.mol.Molecule_Crd_Map();
    md_info.MD_Information_Crd_To_Uint_Crd();

    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
    md_info.MD_Information_Crd_To_Uint_Crd();
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Check_Vector_Finite(&controller, "SPONGE_CPU_DEBUG_CRD_FINITE", "after_vatom_coordinate_refresh", md_info.crd, md_info.atom_numbers);
#endif
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
}

void Main_Print()
{
#ifdef SPONGE_CPU_PHASE1
    Cpu_Debug_Report_Scalar_Stats(
        &controller,
        "SPONGE_CPU_DEBUG_CHARGE_STATS",
        "before_print_charge",
        md_info.d_charge,
        md_info.atom_numbers,
        md_info.sys.steps);
#endif
    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0)
    {
        if (pme.is_initialized)
        {
            // The energy-only paths below reuse PME direct/correction staging buffers.
            // Reset them to make per-step print decomposition independent from force-path history.
            cudaMemset(pme.d_direct_atom_energy, 0, sizeof(float) * md_info.atom_numbers);
            cudaMemset(pme.d_correction_atom_energy, 0, sizeof(float) * md_info.atom_numbers);
            cudaMemset(pme.d_direct_ene, 0, sizeof(float));
            cudaMemset(pme.d_correction_ene, 0, sizeof(float));
        }
        md_info.Step_Print(&controller);
        if (!md_info.pbc.pbc)
        {
            controller.Step_Print("Coulomb", CF_NOPBC.Get_Energy(md_info.pbc.nopbc_crd, md_info.d_charge, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers), true);
            controller.Step_Print("gb", gb.Get_Energy(md_info.pbc.nopbc_crd, md_info.d_charge), true);
            controller.Step_Print("LJ", LJ_NOPBC.Get_Energy(md_info.pbc.nopbc_crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers), true);
        }
        else if (!sits.is_initialized)
        {
            controller.Step_Print("LJ", lj.Get_Energy(md_info.uint_crd, neighbor_list.d_nl, pme.beta, md_info.d_charge, pme.d_direct_atom_energy), true);
            controller.Step_Print("LJ_soft", lj_soft.Get_Energy(md_info.uint_crd, neighbor_list.d_nl, pme.beta, md_info.d_charge, pme.d_direct_atom_energy), true);
        }
        else
        {
            sits_dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, false);
            sits.Step_Print(&controller, 1.0 / (CONSTANT_kB * md_info.sys.target_temperature),
                &lj, &lj_soft, md_info.atom_numbers, md_info.uint_crd, neighbor_list.d_nl,
                pme.beta, md_info.d_charge, pme.d_direct_atom_energy, md_info.sys.steps, sits_dihedral.d_sigma_of_dihedral_ene);
        }
        controller.Step_Print(pairwise_force.force_name.c_str(),
            pairwise_force.Get_Energy(neighbor_list.d_nl, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.nb.cutoff,
                pme.beta, md_info.d_charge, pme.d_direct_atom_energy),
            true);
        soft_walls.Step_Print(&controller, md_info.atom_numbers, md_info.crd);
        pme.Step_Print(&controller, md_info.uint_crd, md_info.d_charge, neighbor_list.d_nl, md_info.pbc.uint_dr_to_dr_cof,
            md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

        const float nb14_lj_energy = nb14.Get_14_LJ_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
        const float nb14_ee_energy = nb14.Get_14_CF_Energy(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof);
        const float bond_energy = bond.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
        controller.Step_Print("nb14_LJ", nb14_lj_energy, true);
        controller.Step_Print("nb14_EE", nb14_ee_energy, true);
        controller.Step_Print("bond", bond_energy, true);
        controller.Step_Print("bond_soft", bond_soft.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        listed_forces.Step_Print(&controller, md_info.crd, md_info.sys.box_length);
        const float angle_energy = angle.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
        controller.Step_Print("angle", angle_energy, true);
        controller.Step_Print("urey_bradley", urey_bradley.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("restrain", restrain.Get_Energy(md_info.crd, md_info.sys.box_length), true);
        const float dihedral_energy = dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
        controller.Step_Print("dihedral", dihedral_energy, true);
        controller.Step_Print("improper_dihedral", improper.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("cmap", cmap.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
#ifdef SPONGE_CPU_PHASE1
        if (Cpu_Debug_Env_On("SPONGE_CPU_DEBUG_TERM_VALUES"))
        {
            static int sum_selfcheck_inited = 0;
            static float* sum_selfcheck_list = NULL;
            static float* sum_selfcheck_sum = NULL;
            float sum_selfcheck_value = NAN;
            if (!sum_selfcheck_inited)
            {
                float seed_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                cudaMalloc((void**)&sum_selfcheck_list, sizeof(float) * 4);
                cudaMalloc((void**)&sum_selfcheck_sum, sizeof(float));
                cudaMemcpy(sum_selfcheck_list, seed_values, sizeof(float) * 4, cudaMemcpyHostToDevice);
                sum_selfcheck_inited = 1;
            }
            if (sum_selfcheck_list != NULL && sum_selfcheck_sum != NULL)
            {
                Sum_Of_List(sum_selfcheck_list, sum_selfcheck_sum, 4);
                cudaMemcpy(&sum_selfcheck_value, sum_selfcheck_sum, sizeof(float), cudaMemcpyDeviceToHost);
            }
            std::fprintf(
                stderr,
                "CPU_DEBUG_TERMS step=%d nb14_LJ=%e nb14_EE=%e bond=%e angle=%e dihedral=%e sum_selfcheck=%e\n",
                md_info.sys.steps,
                nb14_lj_energy,
                nb14_ee_energy,
                bond_energy,
                angle_energy,
                dihedral_energy,
                sum_selfcheck_value);
        }
#endif
        controller.Step_Print("density", md_info.sys.Get_Density());
        controller.Step_Print("pressure", md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION);
        controller.Step_Print("dV_dt", ad_baro.dV_dt);
        controller.Step_Print("sits_dihedral", sits_dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        if (meta.is_initialized)
        {
            meta.cv->Compute(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, CV_NEED_CPU_VALUE, md_info.sys.steps + 1);
            controller.Step_Print("meta1d", meta.Potential(meta.cv->value), true);
        }
        controller.Step_Print(restrain_cv.module_name, 
            restrain_cv.Get_Energy(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps + 1),
            true);
        controller.Step_Print(steer_cv.module_name, 
            steer_cv.Get_Energy(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps + 1),
            true);
        cv_controller.Step_Print(md_info.sys.steps, md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length);
        plugin.Mdout_Print();

        controller.Step_Print("potential", controller.printf_sum);
        controller.Print_To_Screen_And_Mdout();
        neighbor_list.Check_Overflow(&controller);
        controller.Check_Error(md_info.sys.h_potential);
        if (md_info.mode == md_info.RERUN)
        {
            md_info.rerun.Iteration();
            Main_Box_Length_Change(md_info.rerun.box_length_change_factor);
        }
    }
    if (md_info.output.write_trajectory_interval && md_info.sys.steps % md_info.output.write_trajectory_interval == 0)
    {
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Box_Traj_File();
        meta.Write_Potential();
        if (md_info.output.is_vel_traj)
        {
            md_info.output.Append_Vel_Traj_File();
        }
        if (md_info.output.is_frc_traj)
        {
            md_info.output.Append_Frc_Traj_File();
        }
        nhc.Save_Trajectory_File();
    }
    if (md_info.output.write_restart_file_interval && md_info.sys.steps % md_info.output.write_restart_file_interval == 0)
    {
        md_info.output.Export_Restart_File();
        nhc.Save_Restart_File();
    }
}

void Main_Volume_Change(double factor)
{
    md_info.Update_Volume(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Volume(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    md_info.mol.Molecule_Crd_Map();
}

void Main_Box_Length_Change(VECTOR factor)
{
    md_info.Update_Box_Length(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Box_Length(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    md_info.mol.Molecule_Crd_Map();
}

void Main_Volume_Change_Largely()
{
    controller.printf("Some modules are based on the meshing methods, and it is more precise to re-initialize these modules now for a long time or a large volume change.\n");
    neighbor_list.Clear();
    pme.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, 1);
    pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length ,md_info.nb.cutoff );
    controller.printf("------------------------------------------------------------------------------------------------------------\n"); 
}

