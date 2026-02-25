#include "tabulated.h"

REGISTER_CV_STRUCTURE(CV_TABULATED, "tabulated", 0);

static __global__ void BSpline_interpolate1d(int atom_numbers,
    const float* points, const float min, const float max, const float delta,
    const float* x, const VECTOR* x_crd_grads, const VECTOR *x_box_grads,
    float* y, VECTOR* y_crd_grads, VECTOR* y_box_grads)
{
    int ti = blockDim.x * blockIdx.x + threadIdx.x;
    float y0 = NAN;
    float dy_dx = NAN;
    if (ti == 0)
    {
        float x0 = x[0] - min;
        if (x0 >= 0 && x0 <= max - min)
        {
            x0 /= delta;
            int i = x0 + 1;
            x0 = (float)i - x0;
            y0 = BSpline_4_1(x0) * points[i - 1] + BSpline_4_2(x0) * points[i] + BSpline_4_3(x0) * points[i + 1] + BSpline_4_4(x0) * points[i + 2];
            dy_dx = dBSpline_4_1(x0) * points[i - 1] + dBSpline_4_2(x0) * points[i] + dBSpline_4_3(x0) * points[i + 1] + dBSpline_4_4(x0) * points[i + 2];
        }
        y[0] = y0;
        y_box_grads[0] = dy_dx * x_box_grads[0];
    }
    __syncthreads();
    if (ti < atom_numbers)
    {
        y_crd_grads[ti] = dy_dx * x_crd_grads[ti];
    }
}

void CV_TABULATED::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
    CV_LIST cv_lists = manager->Ask_For_CV(module_name, 1, 0, 2);
    cv = cv_lists[0];
    std::vector<float> cpu_parameters = manager->Ask_For_Indefinite_Length_Float_Parameter(module_name, "parameter");
    float* temp;
    temp = manager->Ask_For_Float_Parameter(module_name, "min", 1, 2);
    cv_min = temp[0];
    free(temp);
    temp = manager->Ask_For_Float_Parameter(module_name, "max", 1, 2);
    cv_max = temp[0];
    free(temp);
    delta = (cv_max - cv_min) / (cpu_parameters.size() - 1);
    float padding = cpu_parameters.front();
    if (manager->Command_Exist(module_name, "min_padding"))
    {
        manager->Check_Float(module_name, "min_padding", "CV_TABULATED::Initial");
        padding = atof(manager->Command(module_name, "min_padding"));
    }
    cpu_parameters.insert(cpu_parameters.begin(), padding);
    padding = cpu_parameters.back();
    if (manager->Command_Exist(module_name, "max_padding"))
    {
        manager->Check_Float(module_name, "max_padding", "CV_TABULATED::Initial");
        padding = atof(manager->Command(module_name, "max_padding"));
    }
    cpu_parameters.push_back(padding);
    Cuda_Malloc_And_Copy_Safely((void**)&parameters, &cpu_parameters[0], sizeof(float) * cpu_parameters.size());
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_TABULATED::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need)
    {
        cv->Compute(atom_numbers, uint_crd, scaler, crd, box_length, CV_NEED_ALL, step);
        cudaStreamSynchronize(cv->cuda_stream);
        CUDA_LAUNCH_STREAM(BSpline_interpolate1d,
            dim3((atom_numbers + 1023) / 1024), dim3(1024), 0, this->cuda_stream,
            atom_numbers, parameters, cv_min, cv_max, delta,
            cv->d_value, cv->crd_grads, cv->box_grads, this->d_value, this->crd_grads, this->box_grads);
        cudaMemcpyAsync(&this->value, this->d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
