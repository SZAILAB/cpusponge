#include "main_ti.h"

#define SUBPACKAGE_HINT "SPONGE_TI, for the analysis of FEP molecular dynamics simulation results by thermodynamic integration"

CONTROLLER controller;
MD_INFORMATION md_info;
NEIGHBOR_LIST neighbor_list;
LJ_SOFT_CORE lj_soft;
Particle_Mesh_Ewald pme;
CROSS_PME cross_pme;

int main(int argc, char* argv[])
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

void Main_Initial(int argc, char *argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    controller.Set_Command("mode", "rerun");
    md_info.Initial(&controller);
    controller.Step_Print_Initial("dH_dlambda", "%.2f");

    lj_soft.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
    pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff);
    cross_pme.Initial(&controller, md_info.atom_numbers, pme.PME_Nall, lj_soft.lambda);

    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.FORCED_UPDATE);

    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
}

void Main_Calculate_Force()
{
    md_info.MD_Information_Crd_To_Uint_Crd();
    return;
}

void Main_Iteration()
{
    return;
}



void Main_Print()
{
    md_info.Step_Print(&controller);
    controller.Step_Print("LJ_soft", lj_soft.Get_Partial_H_Partial_Lambda_With_Columb_Direct(md_info.uint_crd, md_info.d_charge, neighbor_list.d_nl,
        cross_pme.d_charge_B_A, pme.beta, cross_pme.charge_perturbated, true), true);
    controller.Step_Print("PME", cross_pme.Get_Partial_H_Partial_Lambda(&pme, md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.d_charge, neighbor_list.d_nl,
        md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, lj_soft.d_sigma_of_dH_dlambda_direct), true);
    controller.Step_Print("dH_dlambda", controller.printf_sum);
    neighbor_list.Check_Overflow(&controller);
    controller.Check_Error(md_info.sys.h_potential);
    controller.Print_To_Screen_And_Mdout();
    md_info.rerun.Iteration();
    Main_Box_Length_Change(md_info.rerun.box_length_change_factor);
}

void Main_Box_Length_Change(VECTOR factor)
{
    md_info.Update_Box_Length(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Box_Length(md_info.sys.box_length);
}



void Main_Clear()
{
    controller.core_time.Stop();
    controller.printf("Core Run Wall Time: %f seconds\n", controller.core_time.time);
    controller.simulation_speed = md_info.sys.steps / controller.core_time.time;
    controller.printf("Core Run Speed: %f frame/second\n", controller.simulation_speed);
    fcloseall();
}
