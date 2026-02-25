#ifndef MAIN_CUH
#define MAIN_CUH


#include "common.h"
#include "control.h"
#include "MD_core/MD_core.h"
#include "Lennard_Jones_force/LJ_soft_core.h"
#include "PME_force/cross_PME.h"
#include "neighbor_list/neighbor_list.h"

void Main_Initial(int argc, char* argv[]);
void Main_Clear();

void Main_Calculate_Force();
void Main_Iteration();
void Main_Print();

void Main_Volume_Change(double factor);
void Main_Box_Length_Change(VECTOR factor);

#endif