#ifndef MAIN_RUN_CUH
#define MAIN_RUN_CUH


#include "common.h"
#include "control.h"
#include "MD_core/MD_core.h"
#include "bond/bond.h"
#include "bond/listed_forces.h"
#include "angle/angle.h"
#include "angle/Urey_Bradley_force.h"
#include "dihedral/dihedral.h"
#include "dihedral/improper_dihedral.h"
#include "nb14/nb14.h"
#include "cmap/cmap.h"
#include "neighbor_list/neighbor_list.h"
#include "Lennard_Jones_force/Lennard_Jones_force.h"
#include "Lennard_Jones_force/solvent_LJ.h"
#include "PME_force/PME_force.h"
#include "thermostats/Middle_Langevin_MD.h"
#include "thermostats/Langevin_MD.h"
#include "thermostats/Andersen_thermostat.h"
#include "thermostats/Berendsen_thermostat.h"
#include "thermostats/nose_hoover_chain.h"
#include "barostats/MC_barostat.h"
#include "barostats/Berendsen_barostat.h"
#include "barostats/andersen_barostat.h"
#include "restrain/restrain.h"
#include "constrain/constrain.h"
#include "constrain/SETTLE.h"
#include "constrain/SHAKE.h"
#include "constrain/simple_constrain.h"
#include "virtual_atoms/virtual_atoms.h"
#include "No_PBC/Lennard_Jones_force_No_PBC.h"
#include "No_PBC/Coulomb_Force_No_PBC.h"
#include "No_PBC/generalized_Born.h"
#include "Lennard_Jones_force/LJ_soft_core.h"
#include "Lennard_Jones_force/pairwise_force.h"
#include "bond/bond_soft.h"
#include "SITS/SITS.h"
#include "collective_variable/collective_variable.h"
#include "collective_variable/simple_cv.h"
#include "bias/restrain_cv.h"
#include "bias/steer.h"
#include "bias/Meta1D.h"
#include "wall/hard_wall.h"
#include "wall/soft_wall.h"
#include "plugin/plugin.h"

void Main_Initial(int argc, char *argv[]);
void Main_Clear();

void Main_Calculate_Force();
void Main_Iteration();
void Main_Print();

void Main_Volume_Change(double factor);
void Main_Box_Length_Change(VECTOR factor);
void Main_Volume_Change_Largely();

#endif //MAIN_CUH
