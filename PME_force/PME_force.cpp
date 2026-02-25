#include "PME_force.h"


//constants
#define PI 3.1415926
#define INVSQRTPI  0.56418958835977
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446
__constant__ float PME_Ma[4] = { 1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0 };
__constant__ float PME_Mb[4] = { 0, 0.5, -1, 0.5 };
__constant__ float PME_Mc[4] = { 0, 0.5, 0, -0.5 };
__constant__ float PME_Md[4] = { 0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0 };
__constant__ float PME_dMa[4] = { 0.5, -1.5, 1.5, -0.5 };
__constant__ float PME_dMb[4] = { 0, 1, -2, 1 };
__constant__ float PME_dMc[4] = { 0, 0.5, 0, -0.5 };

//local functions
static float M_(float u, int n)
{
    if (n == 2)
    {
        if (u > 2 || u < 0)
            return 0;
        return 1 - abs(u - 1);
    }
    else
        return  u / (n - 1) * M_(u, n - 1) + (n - u) / (n - 1) * M_(u - 1, n - 1);    
}

static cufftComplex expc(cufftComplex z)
{
    cufftComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

static float getb(int k, int NFFT, int B_order)
{
    cufftComplex tempc,tempc2,res;
    float tempf;
    tempc2.x=0;
    tempc2.y=0;
    
    tempc.x=0;
    tempc.y = 2 * (B_order - 1) * PI * k / NFFT;
    res = expc(tempc);
    
    for (int kk = 0; kk<(B_order - 1); kk++)
    {
        tempc.x = 0;
        tempc.y=2*PI*k/NFFT*kk;
        tempc = expc(tempc);
        tempf = M_(kk + 1, B_order);
        tempc2.x += tempf*tempc.x;
        tempc2.y += tempf*tempc.y;
    }
    res = cuCdivf(res, tempc2);
    return res.x * res.x + res.y * res.y;
}

static __global__ void Build_PMC_IZ_C(const int PME_Nfft, int fftx, int ffty, int fftz, 
    float box_length_inverse_x_square, float box_length_inverse_y_square, float grid_length_of_z,
    float beta, float scalor, cufftComplex *C)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < PME_Nfft)
    {
        int ffta = (fftx / 2 + 1);
        int grid_x = tid % ffta;
        int grid_y = (tid % (ffta * ffty)) / ffta;
        int grid_z = tid / ffty / ffta;
        if (grid_x >= fftx / 2)
        {
            grid_x = fftx - grid_x;
        }
        if (grid_y >= ffty / 2)
        {
            grid_y = ffty - grid_y;
        }
        if (grid_z >= fftz / 2)
        {
            grid_z = fftz - grid_z;
        }
        float z = grid_length_of_z * grid_z;
        float A = 2.0f * CONSTANT_Pi * sqrtf(grid_x * grid_x * box_length_inverse_x_square 
            + grid_y * grid_y * box_length_inverse_y_square);
        float AB = A / beta / 2.0f;
        float zb2 = z * beta;
        float AB_minus_zb2 = AB - zb2;
        float AB_plus_zb2 = AB + zb2;
        float temp_f = expf(-A * z) * (erfcf(AB_minus_zb2) + expf(2.0f * A * z - AB_plus_zb2 * AB_plus_zb2) * erfcxf(AB_plus_zb2));
        temp_f = temp_f / A;
        if (grid_x == 0 && grid_y == 0)
        {
            temp_f = 2.0f / sqrtf(CONSTANT_Pi) / beta * (1.0f - expf(-zb2 * zb2)) - 2.0f * z * erff(zb2);
        }
        C[tid].x = scalor * temp_f;
        C[tid].y = 0.;
    }
}

static __global__ void Build_PMC_IZ_BC_Final(const int Nfft, int fftx, int ffty, int fftz, 
    const cufftComplex* C, const cufftComplex* B, float *BC)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Nfft)
    {
        int fftc = fftz / 2 + 1;
        int ffta = fftx / 2 + 1;
        int zi = tid % fftc;
        int yi = (tid / fftc) % ffty;
        int xi = tid / fftc / ffty;
        if (xi >= fftx / 2)
        {
            xi = fftx - xi;
        }
        int ti = zi * ffta * ffty + yi * ffta + xi;
        float b = B[ti].x;
        BC[tid] = C[ti].x / (b * b);
    }
}

static void Build_PMC_IZ_BC(CONTROLLER* controller,
    int fftx, int ffty, int fftz,
    int PME_Nfft, int PME_Nall, int PME_Nin,
    float box_length_inverse_x_square, float box_length_inverse_y_square, float grid_length_of_z,
    float beta, float scalor, float **BC)
{
    Cuda_Malloc_Safely((void**)BC, sizeof(float) * PME_Nfft);
    int n2d[2] = { ffty, fftx };
    cufftResult result;
    cufftHandle plan_2d_many_c2r, plan_3d_temp_r2c;
    result = cufftPlanMany(&plan_2d_many_c2r, 2, n2d, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, fftz);
    if (result != CUFFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "Build_PMC_IZ_Efficient_Potential", "Reason:\n\tFail to create the batched 2D FFT plan");
    }
    result = cufftPlan3d(&plan_3d_temp_r2c, fftz, ffty, fftx, CUFFT_R2C);
    if (result != CUFFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "Build_PMC_IZ_Efficient_Potential", "Reason:\n\tFail to create the temporary 3D Real to Complex FFT plan");
    }
    cufftComplex* B, * C;
    float* d_FB, * h_FB, * FC;
    int temp_Nfft = (fftx / 2 + 1) * ffty * fftz;
    Cuda_Malloc_Safely((void**)&B, sizeof(cufftComplex) * temp_Nfft);
    Cuda_Malloc_Safely((void**)&C, sizeof(cufftComplex) * temp_Nfft);
    Cuda_Malloc_Safely((void**)&FC, sizeof(float) * PME_Nall);
    Cuda_Malloc_Safely((void**)&d_FB, sizeof(float) * PME_Nall);
    Malloc_Safely((void**)&h_FB, sizeof(float) * PME_Nall);

    for (int i = 0; i < PME_Nall; i = i + 1)
    {
        h_FB[i] = 0.;
    }
    float temp_b_spline[3] = { 1. / 6., 2. / 3., 1. / 6. };
    for (int k = -1; k <= 1; k = k + 1)
    {
        for (int j = -1; j <= 1; j = j + 1)
        {
            for (int i = -1; i <= 1; i = i + 1)
            {
                float weight = temp_b_spline[k + 1] * temp_b_spline[j + 1] * temp_b_spline[i + 1];
                int kk, jj, ii;
                if (k < 0)
                {
                    kk = k + fftz;
                }
                else
                {
                    kk = k;
                }
                if (j < 0)
                {
                    jj = j + ffty;
                }
                else
                {
                    jj = j;
                }
                if (i < 0)
                {
                    ii = i + fftx;
                }
                else
                {
                    ii = i;
                }
                h_FB[ii + jj * fftx + kk * fftx * ffty] = weight;
            }
        }
    }
    cudaMemcpy(d_FB, h_FB, sizeof(float) * PME_Nall, cudaMemcpyHostToDevice);
    cufftExecR2C(plan_3d_temp_r2c, d_FB, B);
    CUDA_LAUNCH(Build_PMC_IZ_C,
        dim3((temp_Nfft + 1023) / 1024),
        dim3(1024),
        temp_Nfft, fftx, ffty, fftz,
        box_length_inverse_x_square, box_length_inverse_y_square, grid_length_of_z,
        beta, scalor, C);
    cufftExecC2R(plan_2d_many_c2r, C, FC);
    cufftExecR2C(plan_3d_temp_r2c, FC, C);
    CUDA_LAUNCH(Build_PMC_IZ_BC_Final,
        dim3((PME_Nfft + 1023) / 1024),
        dim3(1024),
        PME_Nfft, fftx, ffty, fftz, C, B, BC[0]);

    cudaFree(FC);
    cudaFree(C);
    cudaFree(B);
    cudaFree(d_FB);
    free(h_FB);
    cufftDestroy(plan_2d_many_c2r);
    cufftDestroy(plan_3d_temp_r2c);
}

static float Get_Beta(float cutoff, float tolerance)
{
    float beta, low, high, tempf;
    int ilow, ihigh;
    
    high = 1.0;
    ihigh = 1;
    
    while (1)
    {
        tempf = erfc(high * cutoff) / cutoff;
        if (tempf <= tolerance)
            break;
        high *= 2;
        ihigh++;
    }
    
    ihigh += 50;
    low = 0.0;
    for (ilow =1; ilow < ihigh; ilow++)
    {
        beta = (low + high) / 2;
        tempf = erfc(beta * cutoff) / cutoff;
        if (tempf >= tolerance)
            low = beta;
        else
            high = beta;
    }
    return beta;
}

static __global__ void device_add(float *ene, float factor, float *charge_sum)
{
    ene[0] += factor * charge_sum[0] * charge_sum[0];
}

#ifdef SPONGE_CPU_PHASE1
static inline float Cpu_Sum_Product_Serial(const float* list1, const float* list2, int n)
{
    if (list1 == NULL || list2 == NULL || n <= 0)
    {
        return 0.0f;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum += (double)list1[i] * (double)list2[i];
    }
    return (float)sum;
}

static inline float Cpu_Sum_List_Serial(const float* list, int n)
{
    if (list == NULL || n <= 0)
    {
        return 0.0f;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum += (double)list[i];
    }
    return (float)sum;
}
#endif

//////////////////////////////
void Particle_Mesh_Ewald::Initial(CONTROLLER *controller, int atom_numbers, VECTOR boxlength,float cutoff, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "PME");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    controller[0].printf("START INITIALIZING PME:\n");
    this->cutoff = cutoff;

    tolerance = 0.00001;
    if (controller[0].Command_Exist(this->module_name, "Direct_Tolerance"))
    {
        controller->Check_Float(this->module_name, "Direct_Tolerance", "Particle_Mesh_Ewald::Initial");
        tolerance = atof(controller[0].Command(this->module_name, "Direct_Tolerance"));
    }

    fftx = -1;
    ffty = -1;
    fftz = -1;
    if (controller[0].Command_Exist(this->module_name, "fftx"))
    {
        controller->Check_Int(this->module_name, "fftx", "Particle_Mesh_Ewald::Initial");
        fftx = atoi(controller[0].Command(this->module_name, "fftx"));
    }
    if (controller[0].Command_Exist(this->module_name, "ffty"))
    {
        controller->Check_Int(this->module_name, "ffty", "Particle_Mesh_Ewald::Initial");
        ffty = atoi(controller[0].Command(this->module_name, "ffty"));
    }
    if (controller[0].Command_Exist(this->module_name, "fftz"))
    {
        controller->Check_Int(this->module_name, "fftz", "Particle_Mesh_Ewald::Initial");
        fftz = atoi(controller[0].Command(this->module_name, "fftz"));
    }
        

    this->atom_numbers = atom_numbers;
    this->boxlength = boxlength;

    float volume = boxlength.x * boxlength.y * boxlength.z;


    if (fftx < 0)
        fftx = Get_Fft_Patameter(boxlength.x);

    if (ffty < 0)
        ffty = Get_Fft_Patameter(boxlength.y);

    if (fftz < 0)
        fftz = Get_Fft_Patameter(boxlength.z);



    controller[0].printf("    fftx: %d\n", fftx);
    controller[0].printf("    ffty: %d\n", ffty);
    controller[0].printf("    fftz: %d\n", fftz);

    PME_Nall = fftx * ffty * fftz;
    PME_Nin = ffty * fftz;
    PME_Nfft = fftx * ffty * (fftz / 2 + 1);
    PME_inverse_box_vector.x = (float)fftx / boxlength.x;
    PME_inverse_box_vector.y = (float)ffty / boxlength.y;
    PME_inverse_box_vector.z = (float)fftz / boxlength.z;



    beta = Get_Beta(cutoff, tolerance);
    controller[0].printf("    beta: %f\n", beta);

    neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
    Cuda_Malloc_Safely((void**)&charge_sum, sizeof(float));

    int i, kx, ky, kz, kxrp, kyrp, kzrp, index;
    cufftResult errP1, errP2;
    update_interval = 1;
    if (controller->Command_Exist("PME", "update_interval"))
    {
        controller->Check_Int("PME", "update_interval", "Particle_Mesh_Ewald::Initial");
        update_interval = atoi(controller->Command("PME", "update_interval"));
    }
    Cuda_Malloc_Safely((void**)&force_backup, sizeof(VECTOR) * atom_numbers);
    cudaMemset(force_backup, 0, sizeof(VECTOR) * atom_numbers);
    Cuda_Malloc_Safely((void**)&PME_uxyz, sizeof(UNSIGNED_INT_VECTOR)* atom_numbers);
    Cuda_Malloc_Safely((void**)&PME_frxyz, sizeof(VECTOR)* atom_numbers);
    CUDA_LAUNCH(Reset_List,
        dim3(static_cast<unsigned int>(3 * atom_numbers / 32 + 1)),
        dim3(32),
        3 * atom_numbers,
        (int*)PME_uxyz,
        1 << 30);

    Cuda_Malloc_Safely((void**)&PME_Q, sizeof(float)* PME_Nall);
    Cuda_Malloc_Safely((void**)&PME_FQ, sizeof(cufftComplex)* PME_Nfft);
    Cuda_Malloc_Safely((void**)&PME_FBCFQ, sizeof(float)* PME_Nall);

    int **atom_near_cpu = NULL;
    Malloc_Safely((void**)&atom_near_cpu, sizeof(int*)* atom_numbers);
    Cuda_Malloc_Safely((void**)&PME_atom_near, sizeof(int*)* atom_numbers);
    for (i = 0; i < atom_numbers; i++)
    {
        Cuda_Malloc_Safely((void**)&atom_near_cpu[i], sizeof(int)* 64);
    }
    cudaMemcpy(PME_atom_near, atom_near_cpu, sizeof(int*)* atom_numbers, cudaMemcpyHostToDevice);
    free(atom_near_cpu);


    errP1 = cufftPlan3d(&PME_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);
    errP2 = cufftPlan3d(&PME_plan_c2r, fftx, ffty, fftz, CUFFT_C2R);
    if (errP1 != CUFFT_SUCCESS || errP2 != CUFFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "Particle_Mesh_Ewald::Initial",
            "Reason:\n\tError occurs when create fft plan of PME");
    }

    Cuda_Malloc_Safely((void**)&d_reciprocal_ene, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_self_ene, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_direct_ene, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_direct_atom_energy, sizeof(float)* atom_numbers);
    Cuda_Malloc_Safely((void**)&d_correction_atom_energy, sizeof(float)* atom_numbers);
    Cuda_Malloc_Safely((void**)&d_correction_ene, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_ee_ene, sizeof(float));


    UNSIGNED_INT_VECTOR *PME_kxyz_cpu = NULL;
    Cuda_Malloc_Safely((void**)&PME_kxyz, sizeof(UNSIGNED_INT_VECTOR)* 64);
    Malloc_Safely((void**)&PME_kxyz_cpu, sizeof(UNSIGNED_INT_VECTOR)* 64);

    for (kx = 0; kx < 4; kx++)
    for (ky = 0; ky < 4; ky++)
    for (kz = 0; kz < 4; kz++)
    {
        index = kx * 16 + ky * 4 + kz;
        PME_kxyz_cpu[index].uint_x = kx;
        PME_kxyz_cpu[index].uint_y = ky;
        PME_kxyz_cpu[index].uint_z = kz;
    }
    cudaMemcpy(PME_kxyz, PME_kxyz_cpu, sizeof(UNSIGNED_INT_VECTOR)* 64, cudaMemcpyHostToDevice);
    free(PME_kxyz_cpu);

    calculate_reciprocal_part = true;
    if (controller->Command_Exist("PME", "calculate_reciprocal_part"))
    {
        calculate_reciprocal_part = controller->Get_Bool("PME", "calculate_reciprocal_part", "Particle_Mesh_Ewald::Initial");
    }
    calculate_excluded_part = true;
    if (controller->Command_Exist("PME", "calculate_excluded_part"))
    {
        calculate_excluded_part = controller->Get_Bool("PME", "calculate_excluded_part", "Particle_Mesh_Ewald::Initial");
    }
    bool use_pmc_iz = false;
    if (controller->Command_Exist("PME", "replaced_by_PMC_IZ"))
    {
        use_pmc_iz = controller->Get_Bool("PME", "replaced_by_PMC_IZ", "Particle_Mesh_Ewald::Initial");
    }

    if (calculate_reciprocal_part)
    {
        if (use_pmc_iz)
        {
            controller->printf("    PMC-IZ will be used instead of PME\n");
            if (controller->Command_Choice("mode", "npt"))
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "Particle_Mesh_Ewald::Initial", "Reason:\n\tPMC-IZ can not be used in NPT mode");
            }
            Build_PMC_IZ_BC(controller, fftx, ffty, fftz,
                PME_Nfft, PME_Nall, PME_Nin, 
                1.0f / boxlength.x/ boxlength.x, 1.0f / boxlength.y / boxlength.y, boxlength.z / fftz,
                beta, CONSTANT_Pi / PME_Nall / boxlength.x / boxlength.y, &PME_BC);
        }
        else
        {
            float* B1 = NULL, * B2 = NULL, * B3 = NULL, * h_PME_BC = NULL, * h_PME_BC0 = NULL;
            B1 = (float*)malloc(sizeof(float) * fftx);;
            B2 = (float*)malloc(sizeof(float) * ffty);
            B3 = (float*)malloc(sizeof(float) * fftz);
            h_PME_BC0 = (float*)malloc(sizeof(float) * PME_Nfft);
            h_PME_BC = (float*)malloc(sizeof(float) * PME_Nfft);
            if (B1 == NULL || B2 == NULL || B3 == NULL || h_PME_BC0 == NULL || h_PME_BC == NULL)
            {
                controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "Particle_Mesh_Ewald::Initial", "Reason:\n\tError occurs when malloc PME_BC of PME");
            }
            for (kx = 0; kx < fftx; kx++)
            {
                B1[kx] = getb(kx, fftx, 4);
            }

            for (ky = 0; ky < ffty; ky++)
            {
                B2[ky] = getb(ky, ffty, 4);
            }

            for (kz = 0; kz < fftz; kz++)
            {
                B3[kz] = getb(kz, fftz, 4);
            }

            float mprefactor = PI * PI / -beta / beta;
            float msq;
            for (kx = 0; kx < fftx; kx++)
            {
                kxrp = kx;
                if (kx > fftx / 2)
                    kxrp = fftx - kx;
                for (ky = 0; ky < ffty; ky++)
                {
                    kyrp = ky;
                    if (ky > ffty / 2)
                        kyrp = ffty - ky;
                    for (kz = 0; kz <= fftz / 2; kz++)
                    {
                        kzrp = kz;

                        msq = kxrp * kxrp / boxlength.x / boxlength.x
                            + kyrp * kyrp / boxlength.y / boxlength.y
                            + kzrp * kzrp / boxlength.z / boxlength.z;

                        index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                        if (kx + ky + kz == 0)
                            h_PME_BC[index] = 0;
                        else
                            h_PME_BC[index] = (float)1.0 / PI / msq * exp(mprefactor * msq) / volume;

                        h_PME_BC0[index] = B1[kx] * B2[ky] * B3[kz];
                        h_PME_BC[index] *= h_PME_BC0[index];


                    }
                }
            }

            Cuda_Malloc_Safely((void**)&PME_BC, sizeof(float) * PME_Nfft);
            Cuda_Malloc_Safely((void**)&PME_BC0, sizeof(float) * PME_Nfft);
            cudaMemcpy(PME_BC, h_PME_BC, sizeof(float) * PME_Nfft, cudaMemcpyHostToDevice);
            cudaMemcpy(PME_BC0, h_PME_BC0, sizeof(float) * PME_Nfft, cudaMemcpyHostToDevice);
            free(B1);
            free(B2);
            free(B3);
            free(h_PME_BC0);
            free(h_PME_BC);
        }
    }

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        if (controller->Command_Exist(this->module_name, "print_detail"))
        {
            print_detail = controller->Get_Bool(this->module_name, "print_detail", "Particle_Mesh_Ewald::Initial");
            if (print_detail)
            {
                controller->Step_Print_Initial("PME_direct", "%.2f");
                controller->Step_Print_Initial("PME_reciprocal", "%.2f");
                controller->Step_Print_Initial("PME_self", "%.2f");
                controller->Step_Print_Initial("PME_correction", "%.2f");
            }
        }
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING PME\n\n");
}

void Particle_Mesh_Ewald::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;
        cudaFree(PME_uxyz);
        cudaFree(PME_kxyz);
        cudaFree(PME_frxyz);
        cudaFree(PME_Q);
        cudaFree(PME_FQ);
        cudaFree(PME_FBCFQ);
        cudaFree(PME_BC);
        cudaFree(PME_BC0);
        cudaFree(charge_sum);

        PME_uxyz = NULL;
        PME_kxyz = NULL;
        PME_frxyz = NULL;
        PME_Q = NULL;
        PME_FQ = NULL;
        PME_FBCFQ = NULL;
        PME_BC = NULL;
        PME_BC0 = NULL;
        charge_sum = NULL;

        int **atom_near_cpu = NULL;
        Malloc_Safely((void**)&atom_near_cpu, sizeof(int*)* atom_numbers);
        cudaMemcpy(atom_near_cpu, PME_atom_near, sizeof(int*)* atom_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < atom_numbers; i++)
        {
            cudaFree(atom_near_cpu[i]);
        }
        cudaFree(PME_atom_near);
        PME_atom_near = NULL;
        free(atom_near_cpu);

        cufftDestroy(PME_plan_r2c);
        cufftDestroy(PME_plan_c2r);

        cudaFree(d_reciprocal_ene);
        cudaFree(d_self_ene);
        cudaFree(d_direct_ene);
        cudaFree(d_direct_atom_energy);
        cudaFree(d_correction_atom_energy);
        cudaFree(d_correction_ene);
        cudaFree(d_ee_ene);

        d_reciprocal_ene= NULL;
        d_self_ene= NULL;
        d_direct_ene= NULL;
        d_direct_atom_energy= NULL;
        d_correction_atom_energy= NULL;
        d_correction_ene= NULL;
        d_ee_ene= NULL;


    }
}
 
__global__ void PME_Atom_Near(const UNSIGNED_INT_VECTOR *uint_crd, int **PME_atom_near, const int PME_Nin,
    const float periodic_factor_inverse_x, const float periodic_factor_inverse_y, const float periodic_factor_inverse_z,
    const int atom_numbers, const int fftx, const int ffty, const int fftz,
    const UNSIGNED_INT_VECTOR *PME_kxyz, UNSIGNED_INT_VECTOR *PME_uxyz, VECTOR *PME_frxyz)
{  
    int atom = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom < atom_numbers)
    {       
        UNSIGNED_INT_VECTOR *temp_uxyz = &PME_uxyz[atom];
        int k, tempux, tempuy, tempuz;
        float tempf;
        tempf = (float)uint_crd[atom].uint_x * periodic_factor_inverse_x;
        tempux = (int)tempf;
        PME_frxyz[atom].x = tempf - tempux;

        tempf = (float)uint_crd[atom].uint_y * periodic_factor_inverse_y;
        tempuy = (int)tempf;
        PME_frxyz[atom].y = tempf - tempuy;

        tempf = (float)uint_crd[atom].uint_z * periodic_factor_inverse_z;
        tempuz = (int)tempf;
        PME_frxyz[atom].z = tempf - tempuz;

        if (tempux != (*temp_uxyz).uint_x || tempuy != (*temp_uxyz).uint_y || tempuz != (*temp_uxyz).uint_z)
        {
            (*temp_uxyz).uint_x = tempux;
            (*temp_uxyz).uint_y = tempuy;
            (*temp_uxyz).uint_z = tempuz;
            int *temp_near = PME_atom_near[atom];
            int kx, ky, kz;
            for (k = 0; k < 64; k++)
            {
                UNSIGNED_INT_VECTOR temp_kxyz = PME_kxyz[k];
                
                kx = tempux - temp_kxyz.uint_x;
                
                if (kx < 0)
                    kx += fftx;
                if (kx >= fftx)
                    kx -= fftx;
                ky = tempuy - temp_kxyz.uint_y;
                if (ky < 0)
                    ky += ffty;
                if (ky >= ffty)
                    ky -= ffty;
                kz = tempuz - temp_kxyz.uint_z;
                if (kz < 0)
                    kz += fftz;
                if (kz >= fftz)
                    kz -= fftz;
                temp_near[k] = kx * PME_Nin + ky * fftz + kz;
            }
        }
    }
}

__global__ void PME_Q_Spread
(int **PME_atom_near, const float *charge, const VECTOR *PME_frxyz, 
float *PME_Q, const UNSIGNED_INT_VECTOR *PME_kxyz, const int atom_numbers)
{
    int atom = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (atom < atom_numbers)
    {
        int k;
        float tempf, tempQ, tempf2;
        int *temp_near = PME_atom_near[atom];
        VECTOR temp_frxyz = PME_frxyz[atom];
        float tempcharge = charge[atom];

        UNSIGNED_INT_VECTOR temp_kxyz;
        unsigned int kx;

        for (k = threadIdx.y; k < 64; k = k + blockDim.y)
        {
            temp_kxyz = PME_kxyz[k];
            kx = temp_kxyz.uint_x;
            tempf = (temp_frxyz.x );
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            
            tempQ = tempcharge * tempf;

            kx = temp_kxyz.uint_y;
            tempf = (temp_frxyz.y);
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];

            tempQ = tempQ * tempf;

            kx = temp_kxyz.uint_z;
            tempf = (temp_frxyz.z);
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempQ = tempQ * tempf;

            atomicAdd(&PME_Q[temp_near[k]], tempQ);
        }
    }
}

__global__ void PME_BCFQ(cufftComplex *PME_FQ, float *PME_BC, int PME_Nfft)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < PME_Nfft)
    {
        float tempf = PME_BC[index];
        cufftComplex tempc = PME_FQ[index];
        PME_FQ[index].x = tempc.x * tempf;
        PME_FQ[index].y = tempc.y * tempf;
    }
}

static __global__ void PME_Final(int **PME_atom_near, const float *charge, const float *PME_Q, VECTOR *force,
    const VECTOR *PME_frxyz, const UNSIGNED_INT_VECTOR *PME_kxyz, const VECTOR PME_inverse_box_vector, const int atom_numbers)
{
    int atom = blockDim.y * blockIdx.y + threadIdx.y;
    if (atom < atom_numbers)
    {
        int kx;
        float tempdQx, tempdQy, tempdQz, tempdx, tempdy, tempdz, tempx, tempy, tempz, tempdQf;
        float tempf, tempf2;
        float tempnvdx = 0.0f;
        float tempnvdy = 0.0f;
        float tempnvdz = 0.0f;
        float temp_charge = charge[atom];
        int *temp_near = PME_atom_near[atom];
        UNSIGNED_INT_VECTOR temp_kxyz;
        VECTOR temp_frxyz = PME_frxyz[atom];
#ifdef SPONGE_CPU_PHASE1
        // CPU launch emulation does not provide warp shuffles; run one thread per atom
        // and accumulate all 64 PME stencil terms directly.
        if (threadIdx.x != 0)
        {
            return;
        }
        for (int k = 0; k < 64; ++k)
        {
            temp_kxyz = PME_kxyz[k];
            tempdQf = -PME_Q[temp_near[k]] * temp_charge;

            kx = temp_kxyz.uint_x;
            tempf = (temp_frxyz.x);
            tempf2 = tempf * tempf;
            tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = temp_kxyz.uint_y;
            tempf = (temp_frxyz.y);
            tempf2 = tempf * tempf;
            tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = temp_kxyz.uint_z;
            tempf = (temp_frxyz.z);
            tempf2 = tempf * tempf;
            tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            tempdQx = tempdx * tempy * tempz * PME_inverse_box_vector.x;
            tempdQy = tempdy * tempx * tempz * PME_inverse_box_vector.y;
            tempdQz = tempdz * tempx * tempy * PME_inverse_box_vector.z;

            tempnvdx += tempdQf * tempdQx;
            tempnvdy += tempdQf * tempdQy;
            tempnvdz += tempdQf * tempdQz;
        }
        force[atom].x = force[atom].x + tempnvdx;
        force[atom].y = force[atom].y + tempnvdy;
        force[atom].z = force[atom].z + tempnvdz;
#else
        int k;
        for (k = threadIdx.x; k < 64; k = k + blockDim.x)
        {
            temp_kxyz = PME_kxyz[k];
            tempdQf = -PME_Q[temp_near[k]] * temp_charge;

            kx = temp_kxyz.uint_x;
            tempf = (temp_frxyz.x);
            tempf2 = tempf * tempf;
            tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = temp_kxyz.uint_y;
            tempf = (temp_frxyz.y);
            tempf2 = tempf * tempf;
            tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = temp_kxyz.uint_z;
            tempf = (temp_frxyz.z);
            tempf2 = tempf * tempf;
            tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
            tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];


            tempdQx = tempdx * tempy * tempz * PME_inverse_box_vector.x;
            tempdQy = tempdy * tempx * tempz * PME_inverse_box_vector.y;
            tempdQz = tempdz * tempx * tempy * PME_inverse_box_vector.z;

            tempnvdx += tempdQf * tempdQx;
            tempnvdy += tempdQf * tempdQy;
            tempnvdz += tempdQf * tempdQz;


        }
        for (int offset = 4; offset > 0; offset /= 2)
        {
            tempnvdx += __shfl_xor_sync(0xFFFFFFFF, tempnvdx, offset, 8);
            tempnvdy += __shfl_xor_sync(0xFFFFFFFF, tempnvdy, offset, 8);
            tempnvdz += __shfl_xor_sync(0xFFFFFFFF, tempnvdz, offset, 8);
        }

        if (threadIdx.x == 0)
        {
            force[atom].x = force[atom].x + tempnvdx;
            force[atom].y = force[atom].y + tempnvdy;
            force[atom].z = force[atom].z + tempnvdz;
        }
#endif
    }
}


__global__ void PME_Energy_Product(const int element_number, const float* list1, const float* list2, float *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
    float lin = 0.0;
    for (int i = threadIdx.x; i < element_number; i = i + blockDim.x)
    {
        lin = lin + list1[i] * list2[i];
    }
    atomicAdd(sum, lin);
}

static __global__ void PME_Excluded_Force_With_Atom_Energy_Correction
(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR sacler,
const float *charge, const float pme_beta, const float sqrt_pi,
const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
VECTOR* frc, float *ene)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int excluded_numbers = excluded_atom_numbers[atom_i];
        if (excluded_numbers > 0)
        {

            int list_start = excluded_list_start[atom_i];
            int list_end = list_start + excluded_numbers;
            int atom_j;
            int int_x;
            int int_y;
            int int_z;

            float charge_i = charge[atom_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
            VECTOR dr;
            float dr2;

            float frc_abs = 0.;
            VECTOR frc_lin;
            VECTOR frc_record = { 0., 0., 0. };
            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                atom_j = excluded_list[i];
                r2 = uint_crd[atom_j];
                charge_j = charge[atom_j];

                int_x = r2.uint_x - r1.uint_x;
                int_y = r2.uint_y - r1.uint_y;
                int_z = r2.uint_z - r1.uint_z;
                dr.x = sacler.x*int_x;
                dr.y = sacler.y*int_y;
                dr.z = sacler.z*int_z;
                dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                //假设剔除表中的原子对距离总是小于cutoff的，正常体系


                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta*dr_abs;
                //sqrt_pi= 2/sqrt(3.141592654);
                frc_abs = beta_dr *sqrt_pi * expf(-beta_dr*beta_dr) + erfcf(beta_dr);
                frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
                frc_abs = -charge_i * charge_j*frc_abs;
                frc_lin.x = frc_abs*dr.x;
                frc_lin.y = frc_abs*dr.y;
                frc_lin.z = frc_abs*dr.z;
                ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;

                frc_record.x = frc_record.x + frc_lin.x;
                frc_record.y = frc_record.y + frc_lin.y;
                frc_record.z = frc_record.z + frc_lin.z;

                atomicAdd(&frc[atom_j].x, -frc_lin.x);
                atomicAdd(&frc[atom_j].y, -frc_lin.y);
                atomicAdd(&frc[atom_j].z, -frc_lin.z);
            }//atom_j cycle
            atomicAdd(&frc[atom_i].x, frc_record.x);
            atomicAdd(&frc[atom_i].y, frc_record.y);
            atomicAdd(&frc[atom_i].z, frc_record.z);
            atomicAdd(ene + atom_i, ene_lin);
        }//if need excluded
    }
}

void Particle_Mesh_Ewald::PME_Excluded_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR sacler, const float *charge,
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
    VECTOR* frc, float *atom_energy)
{
    if (is_initialized && calculate_excluded_part)
    {
        cudaMemset(d_correction_atom_energy, 0, sizeof(float) * atom_numbers);
        CUDA_LAUNCH(PME_Excluded_Force_With_Atom_Energy_Correction,
            dim3(static_cast<unsigned int>(ceilf((float)atom_numbers / 128))),
            dim3(128),
            atom_numbers, uint_crd, sacler,
            charge, beta, TWO_DIVIDED_BY_SQRT_PI,
            excluded_list_start, excluded_list, excluded_atom_numbers,
            frc, atom_energy);
    }
}

static __global__ void PME_Add_Energy_To_Virial(float *d_virial, float *d_direct_ene, float *d_correction_ene, float *d_self_ene, float *d_reciprocal_ene, float update_interval)
{
    d_virial[0] += d_direct_ene[0] + d_correction_ene[0] + d_self_ene[0] + update_interval * d_reciprocal_ene[0];
}

static __global__ void PME_Add_Energy_To_Potential(float *d_ene, float* d_direct_ene, float *d_correction_ene, float *d_self_ene, float *d_reciprocal_ene)
{
    d_ene[0] += d_direct_ene[0] + d_correction_ene[0] + d_self_ene[0] + d_reciprocal_ene[0];
}

static __global__ void device_add_force(const int atom_numbers, float update_interval, VECTOR* force, const VECTOR* force_backup)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        force[i] = force[i] + update_interval * force_backup[i];
    }
}

void Particle_Mesh_Ewald::PME_Reciprocal_Force_With_Energy_And_Virial(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR* force, 
    int need_virial, int need_energy, float *d_virial, float *d_potential, int step)
{
    if (is_initialized)
    {
        if (step % update_interval == 0 && calculate_reciprocal_part)
        {
            CUDA_LAUNCH(PME_Atom_Near,
                dim3(static_cast<unsigned int>(atom_numbers / 32 + 1)),
                dim3(32),
                uint_crd, PME_atom_near, PME_Nin,
                CONSTANT_UINT_MAX_INVERSED * fftx, CONSTANT_UINT_MAX_INVERSED * ffty, CONSTANT_UINT_MAX_INVERSED * fftz,
                atom_numbers, fftx, ffty, fftz,
                PME_kxyz, PME_uxyz, PME_frxyz);

            CUDA_LAUNCH(Reset_List,
                dim3(static_cast<unsigned int>(PME_Nall / 1024 + 1)),
                dim3(1024),
                PME_Nall,
                PME_Q,
                0);

            CUDA_LAUNCH(PME_Q_Spread,
                dim3(static_cast<unsigned int>(atom_numbers / thread_PME.x + 1)),
                dim3(thread_PME),
                PME_atom_near, charge, PME_frxyz,
                PME_Q, PME_kxyz, atom_numbers);

            cufftExecR2C(PME_plan_r2c, (float*)PME_Q, (cufftComplex*)PME_FQ);

            CUDA_LAUNCH(PME_BCFQ,
                dim3(static_cast<unsigned int>(PME_Nfft / 1024 + 1)),
                dim3(1024),
                PME_FQ, PME_BC, PME_Nfft);

            cufftExecC2R(PME_plan_c2r, (cufftComplex*)PME_FQ, (float*)PME_FBCFQ);

            cudaMemset(force_backup, 0, sizeof(VECTOR) * atom_numbers);
            CUDA_LAUNCH(PME_Final,
                dim3(1, static_cast<unsigned int>(atom_numbers / thread_PME.x + 1)),
                dim3(thread_PME),
                PME_atom_near, charge, PME_FBCFQ, force_backup,
                PME_frxyz, PME_kxyz, PME_inverse_box_vector, atom_numbers);
            CUDA_LAUNCH(device_add_force,
                dim3((atom_numbers + 1023) / 1024),
                dim3(1024),
                atom_numbers, update_interval, force, force_backup);
        }
        if (need_virial > 0 || need_energy > 0)
        {
#ifdef SPONGE_CPU_PHASE1
            float reciprocal_ene_local = 0.0f;
            if (step % update_interval == 0 && calculate_reciprocal_part)
            {
                reciprocal_ene_local = 0.5f * Cpu_Sum_Product_Serial(PME_Q, PME_FBCFQ, PME_Nall);
            }
            d_reciprocal_ene[0] = reciprocal_ene_local;

            const float charge_dot = Cpu_Sum_Product_Serial(charge, charge, atom_numbers);
            const float charge_total = Cpu_Sum_List_Serial(charge, atom_numbers);
            float self_ene_local = charge_dot * (-beta / sqrtf(PI));
            self_ene_local += neutralizing_factor * charge_total * charge_total;
            d_self_ene[0] = self_ene_local;

            d_direct_ene[0] = Cpu_Sum_List_Serial(d_direct_atom_energy, atom_numbers);
            d_correction_ene[0] = Cpu_Sum_List_Serial(d_correction_atom_energy, atom_numbers);

            if (need_energy > 0)
            {
                d_potential[0] += d_direct_ene[0] + d_correction_ene[0] + d_self_ene[0] + d_reciprocal_ene[0];
            }
            if (need_virial > 0)
            {
                d_virial[0] += d_direct_ene[0] + d_correction_ene[0] + d_self_ene[0] + update_interval * d_reciprocal_ene[0];
            }
#else
            if (step % update_interval == 0 && calculate_reciprocal_part)
            {
                CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), PME_Nall, PME_Q, PME_FBCFQ, d_reciprocal_ene);
                CUDA_LAUNCH(Scale_List, dim3(1), dim3(1), 1, d_reciprocal_ene, 0.5);
            }
            else
            {
                cudaMemset(d_reciprocal_ene, 0, sizeof(float));
            }
            CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), atom_numbers, charge, charge, d_self_ene);
            CUDA_LAUNCH(Scale_List, dim3(1), dim3(1), 1, d_self_ene, -beta / sqrtf(PI));

            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, charge, charge_sum);
            CUDA_LAUNCH(device_add, dim3(1), dim3(1), d_self_ene, neutralizing_factor, charge_sum);

            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, d_direct_atom_energy, d_direct_ene);
            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, d_correction_atom_energy, d_correction_ene);

            if (need_energy > 0)
                CUDA_LAUNCH(PME_Add_Energy_To_Potential, dim3(1), dim3(1), d_potential, d_direct_ene, d_correction_ene, d_self_ene, d_reciprocal_ene);
            if (need_virial > 0)
                CUDA_LAUNCH(PME_Add_Energy_To_Virial, dim3(1), dim3(1), d_virial, d_direct_ene, d_correction_ene, d_self_ene, d_reciprocal_ene, update_interval);
#endif
        }
    }
}

static __global__ void PME_Excluded_Energy_Correction
(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR sacler,
const float *charge, const float pme_beta, const float sqrt_pi,
const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
float *ene)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int excluded_number = excluded_atom_numbers[atom_i];
        if (excluded_number > 0)
        {
            int list_start = excluded_list_start[atom_i];
            int list_end = list_start + excluded_number;
            int atom_j;
            int int_x;
            int int_y;
            int int_z;

            float charge_i = charge[atom_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
            VECTOR dr;
            float dr2;

            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                atom_j = excluded_list[i];
                r2 = uint_crd[atom_j];
                charge_j = charge[atom_j];

                int_x = r2.uint_x - r1.uint_x;
                int_y = r2.uint_y - r1.uint_y;
                int_z = r2.uint_z - r1.uint_z;
                dr.x = sacler.x*int_x;
                dr.y = sacler.y*int_y;
                dr.z = sacler.z*int_z;
                dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                //假设剔除表中的原子对距离总是小于cutoff的，正常体系
                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta*dr_abs;

                ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;

            }//atom_j cycle
            atomicAdd(ene + atom_i, ene_lin);
        }//if need excluded
    }
}


float Particle_Mesh_Ewald::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
    const ATOM_GROUP *nl, const VECTOR scaler,
	const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers, int which_part, int is_download)
{
    if (is_initialized)
    {
        if (which_part & RECIPROCAL && calculate_reciprocal_part)
        {
            CUDA_LAUNCH(PME_Atom_Near,
                dim3(static_cast<unsigned int>(atom_numbers / 32 + 1)),
                dim3(32),
                uint_crd, PME_atom_near, PME_Nin,
                CONSTANT_UINT_MAX_INVERSED * fftx, CONSTANT_UINT_MAX_INVERSED * ffty, CONSTANT_UINT_MAX_INVERSED * fftz,
                atom_numbers, fftx, ffty, fftz,
                PME_kxyz, PME_uxyz, PME_frxyz);

            CUDA_LAUNCH(Reset_List,
                dim3(static_cast<unsigned int>(PME_Nall / 1024 + 1)),
                dim3(1024),
                PME_Nall, PME_Q, 0);

            CUDA_LAUNCH(PME_Q_Spread,
                dim3(static_cast<unsigned int>(atom_numbers / thread_PME.x + 1)),
                dim3(thread_PME),
                PME_atom_near, charge, PME_frxyz,
                PME_Q, PME_kxyz, atom_numbers);

            cufftExecR2C(PME_plan_r2c, (float*)PME_Q, (cufftComplex*)PME_FQ);


            CUDA_LAUNCH(PME_BCFQ,
                dim3(static_cast<unsigned int>(PME_Nfft / 1024 + 1)),
                dim3(1024),
                PME_FQ, PME_BC, PME_Nfft);

            cufftExecC2R(PME_plan_c2r, (cufftComplex*)PME_FQ, (float*)PME_FBCFQ);

#ifdef SPONGE_CPU_PHASE1
            d_reciprocal_ene[0] = 0.5f * Cpu_Sum_Product_Serial(PME_Q, PME_FBCFQ, PME_Nall);
#else
            CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), PME_Nall, PME_Q, PME_FBCFQ, d_reciprocal_ene);
            CUDA_LAUNCH(Scale_List, dim3(1), dim3(1), 1, d_reciprocal_ene, 0.5);
#endif
        }

        if (which_part & SELF)
        {
#ifdef SPONGE_CPU_PHASE1
            const float charge_dot = Cpu_Sum_Product_Serial(charge, charge, atom_numbers);
            const float charge_total = Cpu_Sum_List_Serial(charge, atom_numbers);
            d_self_ene[0] = charge_dot * (-beta / sqrtf(PI)) + neutralizing_factor * charge_total * charge_total;
#else
            CUDA_LAUNCH(PME_Energy_Product, dim3(1), dim3(1024), atom_numbers, charge, charge, d_self_ene);
            CUDA_LAUNCH(Scale_List, dim3(1), dim3(1), 1, d_self_ene, -beta / sqrtf(PI));

            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, charge, charge_sum);
            CUDA_LAUNCH(device_add, dim3(1), dim3(1), d_self_ene, neutralizing_factor, charge_sum);
#endif
        }

        if (which_part & DIRECT)
        {
#ifdef SPONGE_CPU_PHASE1
            d_direct_ene[0] = Cpu_Sum_List_Serial(d_direct_atom_energy, atom_numbers);
#else
            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, d_direct_atom_energy, d_direct_ene);
#endif
        }

        if (which_part & CORRECTION && calculate_excluded_part)
        {
            cudaMemset(d_correction_atom_energy, 0, sizeof(float) * atom_numbers);
            CUDA_LAUNCH(PME_Excluded_Energy_Correction,
                dim3(static_cast<unsigned int>(atom_numbers / 32 + 1)),
                dim3(32),
                atom_numbers, uint_crd, scaler,
                charge, beta, sqrtf(PI), excluded_list_start, excluded_list, excluded_atom_numbers, d_correction_atom_energy);
#ifdef SPONGE_CPU_PHASE1
            d_correction_ene[0] = Cpu_Sum_List_Serial(d_correction_atom_energy, atom_numbers);
#else
            CUDA_LAUNCH(Sum_Of_List, dim3(1), dim3(1024), atom_numbers, d_correction_atom_energy, d_correction_ene);
#endif
        }

        if (is_download)
        {
			ee_ene = 0;
			if (which_part & RECIPROCAL && calculate_reciprocal_part)
			{
				cudaMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float), cudaMemcpyDeviceToHost);
				ee_ene += reciprocal_ene;
			}
             
			if (which_part & SELF)
			{
				cudaMemcpy(&self_ene, d_self_ene, sizeof(float), cudaMemcpyDeviceToHost);
				ee_ene += self_ene;
			}
             
			if (which_part & DIRECT)
			{
				cudaMemcpy(&direct_ene, d_direct_ene, sizeof(float), cudaMemcpyDeviceToHost);
				ee_ene += direct_ene;
			}
             
			if (which_part & CORRECTION && calculate_excluded_part)
			{
                cudaMemcpy(&correction_ene, d_correction_ene, sizeof(float), cudaMemcpyDeviceToHost);
				ee_ene += correction_ene;
			}
			return ee_ene;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return NAN;
    }
}

void Particle_Mesh_Ewald::Update_Volume(VECTOR box_length)
{
    Update_Box_Length(boxlength);
}

__global__ void up_box_bc(int fftx, int ffty, int fftz, float *PME_BC, float *PME_BC0, float mprefactor, VECTOR boxlength, float volume)
{
    int kx, ky, kz, kxrp, kyrp, kzrp, index;
    float msq;
    for (kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx; kx += blockDim.x * gridDim.x)
    {
        kxrp = kx;
        if (kx > fftx / 2)
            kxrp = fftx - kx;
        for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty; ky += blockDim.y * gridDim.y)
        {
            kyrp = ky;
            if (ky > fftx / 2)
                kyrp = ffty - ky;
            for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z)
            {
                kzrp = kz;
                msq = kxrp * kxrp / boxlength.x / boxlength.x
                    + kyrp * kyrp / boxlength.y / boxlength.y
                    + kzrp * kzrp / boxlength.z / boxlength.z;

                index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                if (kx + ky + kz == 0)
                    PME_BC[index] = 0;
                else
                    PME_BC[index] = (float)1.0 / PI / msq * exp(mprefactor * msq) / volume * PME_BC0[index];
            }
        }
    }
}

void Particle_Mesh_Ewald::Update_Box_Length(VECTOR boxlength)
{
    float volume = boxlength.x * boxlength.y * boxlength.z;
    PME_inverse_box_vector.x = (float)fftx / boxlength.x;
    PME_inverse_box_vector.y = (float)ffty / boxlength.y;
    PME_inverse_box_vector.z = (float)fftz / boxlength.z;
    neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
    float mprefactor = PI * PI / -beta / beta;
    CUDA_LAUNCH(up_box_bc, dim3(20, 20), dim3(8, 8, 16), fftx, ffty, fftz, PME_BC, PME_BC0, mprefactor, boxlength, volume);
}

void Particle_Mesh_Ewald::Step_Print(CONTROLLER* controller, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, const ATOM_GROUP* nl, const VECTOR scaler, 
    const int* excluded_list_start, const int* excluded_list, const int* excluded_atom_numbers)
{
    controller->Step_Print("PME", Get_Energy(uint_crd, charge, nl, scaler,
        excluded_list_start, excluded_list, excluded_atom_numbers), true);
    if (print_detail)
    {
        controller->Step_Print("PME_direct", direct_ene);
        controller->Step_Print("PME_reciprocal", reciprocal_ene);
        controller->Step_Print("PME_self", self_ene);
        controller->Step_Print("PME_correction", correction_ene);
    }

}
