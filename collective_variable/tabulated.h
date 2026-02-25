//Tabulated CV
#ifndef CV_TABULATE_CUH
#define CV_TABULATE_CUH

#include "collective_variable.h"

struct CV_TABULATED : public COLLECTIVE_VARIABLE_PROTOTYPE
{
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    float* parameters;
    float cv_min;
    float cv_max;
    float delta;
    void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name);
    void Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step);
};

#endif


