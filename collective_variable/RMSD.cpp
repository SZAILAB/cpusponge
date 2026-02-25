#include "RMSD.h"

REGISTER_CV_STRUCTURE(CV_RMSD, "rmsd", 0);

static __global__ void Get_Center_of_Atoms(int atom_numbers, int *atoms, VECTOR* crd, VECTOR *points)
{
    __shared__ VECTOR crd_sum[1024];
    crd_sum[threadIdx.x] = { 0, 0, 0 };
    VECTOR tempc;
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
    {
        tempc = crd[atoms[i]];
        points[i] = tempc;
        crd_sum[threadIdx.x] = crd_sum[threadIdx.x] + tempc;
    }
    __syncthreads();
    for (int i = 512; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            crd_sum[threadIdx.x] = crd_sum[threadIdx.x] + crd_sum[i + threadIdx.x];
        }
        __syncthreads();
    }
    tempc = 1.0f / atom_numbers * crd_sum[0];
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
    {
        points[i] = points[i] - tempc;
    }
}

static __global__ void Get_Coordinate_Covariance(const int atom_numbers, const VECTOR* A, const VECTOR* B, float* covariance)
{
    int atom_i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
    if (atom_i < 9)
    {
        covariance[atom_i] = 0;
    }
    __syncthreads();
    float local_covariance11 = 0;
    float local_covariance12 = 0;
    float local_covariance13 = 0;
    float local_covariance21 = 0;
    float local_covariance22 = 0;
    float local_covariance23 = 0;
    float local_covariance31 = 0;
    float local_covariance32 = 0;
    float local_covariance33 = 0;
    VECTOR a;
    VECTOR b;
    if (atom_i < atom_numbers)
    {
        a = A[atom_i];
        b = B[atom_i];
        local_covariance11 += a.x * b.x;
        local_covariance12 += a.x * b.y;
        local_covariance13 += a.x * b.z;
        local_covariance21 += a.y * b.x;
        local_covariance22 += a.y * b.y;
        local_covariance23 += a.y * b.z;
        local_covariance31 += a.z * b.x;
        local_covariance32 += a.z * b.y;
        local_covariance33 += a.z * b.z;
    }
    Warp_Sum_To(covariance, local_covariance11);
    Warp_Sum_To(covariance + 1, local_covariance21);
    Warp_Sum_To(covariance + 2, local_covariance31);
    Warp_Sum_To(covariance + 3, local_covariance12);
    Warp_Sum_To(covariance + 4, local_covariance22);
    Warp_Sum_To(covariance + 5, local_covariance32);
    Warp_Sum_To(covariance + 6, local_covariance13);
    Warp_Sum_To(covariance + 7, local_covariance23);
    Warp_Sum_To(covariance + 8, local_covariance33);
}

static __global__ void Get_Rotated_Reference(const int atom_numbers, const VECTOR* reference, const float* R, VECTOR* rotated_reference)
{
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR inp_crd = reference[atom_i];
        VECTOR out_crd;
        out_crd.x = inp_crd.x * R[0] + inp_crd.y * R[3] + inp_crd.z * R[6];
        out_crd.y = inp_crd.x * R[1] + inp_crd.y * R[4] + inp_crd.z * R[7];
        out_crd.z = inp_crd.x * R[2] + inp_crd.y * R[5] + inp_crd.z * R[8];
        rotated_reference[atom_i] = out_crd;
    }
}

static __global__ void Get_Rotation_Matrix(float *m, float *R)
{
    __shared__ float mDmT[6], UT[9], S[3], VT[9];
    //float determinant =  m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
    //                     - m[0] * m[5] * m[7] - m[1] * m[3] * m[8] - m[2] * m[4] * m[6];
    //determinant = copysignf(1, determinant);
    float norm_factor = fabsf(m[2] * m[0] + m[5] * m[3] + m[8] * m[6]);
    norm_factor = 1.0F / sqrtf(norm_factor);
    for (int i = 0; i < 9; i++)
    {
        m[i] *= norm_factor;
    }
    mDmT[0] = (m[0] * m[0] + m[3] * m[3] + m[6] * m[6]);
    mDmT[1] = (m[1] * m[0] + m[4] * m[3] + m[7] * m[6]);
    mDmT[2] = (m[2] * m[0] + m[5] * m[3] + m[8] * m[6]);
    mDmT[3] = (m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
    mDmT[4] = (m[2] * m[1] + m[5] * m[4] + m[8] * m[7]);
    mDmT[5] = (m[2] * m[2] + m[5] * m[5] + m[8] * m[8]);
    get_eigen(mDmT, S, (VECTOR*)VT);

    UT[0] = (m[0] * VT[0] + m[1] * VT[1] + m[2] * VT[2]);
    UT[1] = (m[3] * VT[0] + m[4] * VT[1] + m[5] * VT[2]);
    UT[2] = (m[6] * VT[0] + m[7] * VT[1] + m[8] * VT[2]);
    norm_factor = rnorm3df(UT[0], UT[1], UT[2]);
    UT[0] *= norm_factor;
    UT[1] *= norm_factor;
    UT[2] *= norm_factor;

    UT[3] = (m[0] * VT[3] + m[1] * VT[4] + m[2] * VT[5]);
    UT[4] = (m[3] * VT[3] + m[4] * VT[4] + m[5] * VT[5]);
    UT[5] = (m[6] * VT[3] + m[7] * VT[4] + m[8] * VT[5]);
    norm_factor = rnorm3df(UT[3], UT[4], UT[5]);
    UT[3] *= norm_factor;
    UT[4] *= norm_factor;
    UT[5] *= norm_factor;

    UT[6] = UT[1] * UT[5] - UT[2] * UT[4];
    UT[7] = UT[2] * UT[3] - UT[0] * UT[5];
    UT[8] = UT[0] * UT[4] - UT[1] * UT[3];

    R[0] = UT[0] * VT[0] + UT[3] * VT[3] + UT[6] * VT[6];
    R[3] = UT[0] * VT[1] + UT[3] * VT[4] + UT[6] * VT[7];
    R[6] = UT[0] * VT[2] + UT[3] * VT[5] + UT[6] * VT[8];
    R[1] = UT[1] * VT[0] + UT[4] * VT[3] + UT[7] * VT[6];
    R[4] = UT[1] * VT[1] + UT[4] * VT[4] + UT[7] * VT[7];
    R[7] = UT[1] * VT[2] + UT[4] * VT[5] + UT[7] * VT[8];
    R[2] = UT[2] * VT[0] + UT[5] * VT[3] + UT[8] * VT[6];
    R[5] = UT[2] * VT[1] + UT[5] * VT[4] + UT[8] * VT[7];
    R[8] = UT[2] * VT[2] + UT[5] * VT[5] + UT[8] * VT[8];
}

static __global__ void get_diff_and_rmsd(int atom_numbers, const VECTOR* points, const VECTOR* rotated_reference, 
    float *d_value, int *atom, VECTOR *crd_grads, VECTOR *box_grads, VECTOR box_length)
{
    __shared__ float rmsd[1024];
    rmsd[threadIdx.x] = 0;
    __shared__ VECTOR drmsd_dx[1024];
    drmsd_dx[threadIdx.x] = { 0, 0, 0 };
    VECTOR diff;
    float rmsd0;
    int atom_i;
    for (int i = threadIdx.x; i < atom_numbers; i += 1024)
    {
        diff = points[i] - rotated_reference[i];
        rmsd[threadIdx.x] += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }
    __syncthreads();
    for (int i = 512; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            rmsd[threadIdx.x] += rmsd[i + threadIdx.x];
        }
        __syncthreads();
    }
    rmsd0 = sqrtf(rmsd[0] / atom_numbers);
    *d_value = rmsd0;
    __syncthreads();
    rmsd0 *= atom_numbers;
    for (int i = threadIdx.x; i < atom_numbers; i += 1024)
    {
        atom_i = atom[i];
        diff = points[i] - rotated_reference[i];
        diff.x /= rmsd0;
        diff.y /= rmsd0;
        diff.z /= rmsd0;
        drmsd_dx[threadIdx.x].x += diff.x * points[i].x / box_length.x;
        drmsd_dx[threadIdx.x].y += diff.y * points[i].y / box_length.y;
        drmsd_dx[threadIdx.x].z += diff.z * points[i].z / box_length.z;
        crd_grads[atom_i] = diff;
    }
    __syncthreads();
    for (int i = 512; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            drmsd_dx[threadIdx.x] = drmsd_dx[threadIdx.x] + drmsd_dx[i + threadIdx.x];
        }
        __syncthreads();
    }
    box_grads[0] = drmsd_dx[0];
}

void CV_RMSD::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
    std::vector<int> cpu_atom = manager->Ask_For_Indefinite_Length_Int_Parameter(module_name, "atom");
    if (cpu_atom.size() == 0)
    {
        std::string error_reason = "Reason:\n\tatoms are required for the CV ";
        error_reason += module_name;
        error_reason += " (atom or atom_in_file)\n";
        manager->Throw_SPONGE_Error(spongeErrorMissingCommand, "CV_RMSD::Initial", error_reason.c_str());
    }
    this->atom_numbers = cpu_atom.size();
    manager->printf("        atom_numbers is %d\n", this->atom_numbers);
    std::vector<float> cpu_reference = manager->Ask_For_Indefinite_Length_Float_Parameter(module_name, "coordinate");
    if (cpu_reference.size() == 0)
    {
        std::string error_reason = "Reason:\n\tcoordinates are required for the CV ";
        error_reason += module_name;
        error_reason += " (coordinate or coordinate_in_file)\n";
        manager->Throw_SPONGE_Error(spongeErrorMissingCommand, "CV_RMSD::Initial", error_reason.c_str());
    }
    if (3 * this->atom_numbers != cpu_reference.size())
    {
        std::string error_reason = "Reason:\n\tthe number of coordinates (";
        error_reason += cpu_reference.size();
        error_reason += ") != 3 * the number of atoms (";
        error_reason += this->atom_numbers;
        error_reason += ")for the CV ";
        error_reason += module_name;
        error_reason += "\n";
        manager->Throw_SPONGE_Error(spongeErrorConflictingCommand, "CV_RMSD::Initial", error_reason.c_str());
    }
    Cuda_Malloc_And_Copy_Safely((void**)&atom, &cpu_atom[0], sizeof(int) * this->atom_numbers);
    Cuda_Malloc_Safely((void**)&points, sizeof(VECTOR) * this->atom_numbers);
    Cuda_Malloc_Safely((void**)&covariance_matrix, sizeof(float) * 9);
    Cuda_Malloc_Safely((void**)&rotated_ref, sizeof(VECTOR) * this->atom_numbers);
    Cuda_Malloc_Safely((void**)&R, sizeof(float) * 9);
    VECTOR center = {0, 0, 0};
    for (int i = 0; i < 3 * this->atom_numbers; i += 3)
    {
        center.x += cpu_reference[i];
        center.y += cpu_reference[i + 1];
        center.z += cpu_reference[i + 2];
    }
    center = 1.0f / this->atom_numbers * center;
    for (int i = 0; i < 3 * this->atom_numbers; i += 3)
    {
        cpu_reference[i] -= center.x;
        cpu_reference[i + 1] -= center.y;
        cpu_reference[i + 2] -= center.z;
    }
    Cuda_Malloc_And_Copy_Safely((void**)&references, &cpu_reference[0], sizeof(VECTOR) * this->atom_numbers);
    rotated_comparing = 1;
    if (manager->Command_Exist(module_name, "rotate"))
    {
        rotated_comparing = manager->Get_Bool(module_name, "rotate", "CV_RMSD::Initial");
    }
    Super_Initial(manager, atom_numbers, module_name);
}

void CV_RMSD::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
    need = Check_Whether_Computed_At_This_Step(step, need);
    if (need)
    {
        CUDA_LAUNCH_STREAM(Get_Center_of_Atoms, dim3(1), dim3(1024), 0, this->cuda_stream,
            this->atom_numbers, this->atom, crd, this->points);
        if (rotated_comparing)
        {
            CUDA_LAUNCH_STREAM(Get_Coordinate_Covariance,
                dim3((atom_numbers + 1023) / 1024), dim3(32, 32), 0, this->cuda_stream,
                this->atom_numbers, this->references, this->points, this->covariance_matrix);
            CUDA_LAUNCH_STREAM(Get_Rotation_Matrix, dim3(1), dim3(1), 0, this->cuda_stream, covariance_matrix, R);
            CUDA_LAUNCH_STREAM(Get_Rotated_Reference,
                dim3((atom_numbers + 1023) / 1024), dim3(1024), 0, this->cuda_stream,
                this->atom_numbers, this->references, this->R, this->rotated_ref);
            CUDA_LAUNCH_STREAM(get_diff_and_rmsd, dim3(1), dim3(1024), 0, this->cuda_stream,
                this->atom_numbers, this->points, this->rotated_ref, d_value, atom, crd_grads, box_grads, box_length);
        }
        else
        {
            CUDA_LAUNCH_STREAM(get_diff_and_rmsd, dim3(1), dim3(1024), 0, this->cuda_stream,
                this->atom_numbers, this->points, this->references, d_value, atom, crd_grads, box_grads, box_length);
        }
        cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
    }
    Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
