R"JIT(
//存储各种常数
//圆周率
#define CONSTANT_Pi 3.1415926535897932f
//自然对数的底
#define CONSTANT_e 2.7182818284590452f
//玻尔兹曼常量（kcal.mol^-1.K^ -1）
//使用kcal为能量单位，因此kB=8.31441(J.mol^-1.K^-1)/4.18407(J/cal)/1000
#define CONSTANT_kB 0.00198716f
//程序中使用的单位时间与物理时间的换算1/20.455*dt=1 ps
#define CONSTANT_TIME_CONVERTION 20.455f
//程序中使用的单位压强与物理压强的换算
// 压强单位: bar -> kcal/mol/A^3 
// (1 kcal/mol) * (4.184074e3 J/kcal) / (6.023e23 mol^-1) * (1e30 m^3/A^3) * (1e-5 bar/pa)
// 程序的压强/(kcal/mol/A^3 ) * CONSTANT_PRES_CONVERTION = 物理压强/bar
#define CONSTANT_PRES_CONVERTION 6.946827162543585e4f
// 物理压强/bar * CONSTANT_PRES_CONVERTION_INVERSE = 程序的压强/(kcal/mol/A^3 )
#define CONSTANT_PRES_CONVERTION_INVERSE 0.00001439506089041446f
//周期性盒子映射所使用的信息，最大的unsigned int
#define CONSTANT_UINT_MAX UINT_MAX
//周期性盒子映射所使用的信息，最大的unsigned int对应的float
#define CONSTANT_UINT_MAX_FLOAT 4294967296.0f
//周期性盒子映射所使用的信息，最大的unsigned int对应的倒数
#define CONSTANT_UINT_MAX_INVERSED 2.3283064365387e-10f

#define CHAR_LENGTH_MAX 512
#define FULL_MASK 0xffffffff

//用于计算边界循环所定义的结构体
struct UNSIGNED_INT_VECTOR
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
};

//用于计算边界循环或者一些三维数组大小所定义的结构体
struct INT_VECTOR
{
    int int_x;
    int int_y;
    int int_z;
};

__device__ __forceinline__ void Warp_Sum_To(float* y, float x, int delta=32);
__host__ __device__ __forceinline__ float BSpline_4_1(float x);
__host__ __device__ __forceinline__ float BSpline_4_2(float x);
__host__ __device__ __forceinline__ float BSpline_4_3(float x);
__host__ __device__ __forceinline__ float BSpline_4_4(float x);
__host__ __device__ __forceinline__ float dBSpline_4_1(float x);
__host__ __device__ __forceinline__ float dBSpline_4_2(float x);
__host__ __device__ __forceinline__ float dBSpline_4_3(float x);
__host__ __device__ __forceinline__ float dBSpline_4_4(float x);
//用于存储各种三维float矢量而定义的结构体
struct VECTOR
{
    float x;
    float y;
    float z;

    friend __device__ __host__ __forceinline__ VECTOR operator+ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR operator+ (const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }

    friend __device__ __host__  __forceinline__ float operator* (const VECTOR& veca, const VECTOR& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__  __forceinline__ VECTOR operator* (const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator- (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR operator- (const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator- (const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = -vecb.x;
        vec.y = -vecb.y;
        vec.z = -vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator/ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator/ (const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator^ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const UNSIGNED_INT_VECTOR uvec_a, const UNSIGNED_INT_VECTOR uvec_b, const VECTOR scaler)
    {
        VECTOR dr;
        dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
        dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
        dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
        return dr;
    }


    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x / box_length.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y / box_length.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z / box_length.z + 0.5f) * box_length.z;
        return dr;
    }

    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length, const VECTOR box_length_inverse)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x * box_length_inverse.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y * box_length_inverse.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z * box_length_inverse.z + 0.5f) * box_length.z;
        return dr;
    }

    friend __device__ __forceinline__ VECTOR Make_Vector_Not_Exceed_Value(VECTOR vector, const float value)
    {
        return fminf(1.0, value * rnorm3df(vector.x, vector.y, vector.z)) * vector;
    }

    friend __device__ __forceinline__ void atomicAdd(VECTOR* a, const VECTOR b)
    {
        atomicAdd(&a->x, b.x);
        atomicAdd(&a->y, b.y);
        atomicAdd(&a->z, b.z);
    }
    friend __device__ __forceinline__ void Warp_Sum_To(float* y, float x, int delta)
    {
        unsigned int mask = __ballot_sync(FULL_MASK, threadIdx.x < delta);
        for (delta >>= 1; delta > 0; delta >>= 1)
        {
            x += __shfl_down_sync(FULL_MASK, x, delta);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(y, x);
        }
    }
    friend __device__ __forceinline__ void Warp_Sum_To(VECTOR* y, VECTOR x, int delta = 32)
    {
        for (delta >>= 1; delta > 0; delta >>= 1)
        {
            x.x += __shfl_down_sync(FULL_MASK, x.x, delta);
            x.y += __shfl_down_sync(FULL_MASK, x.y, delta);
            x.z += __shfl_down_sync(FULL_MASK, x.z, delta);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(y, x);
        }
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_1(float x)
    {
        return 0.1666667f * x * x * x;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_2(float x)
    {
        return -0.5f * x * x * x + 0.5f * x * x + 0.5f * x + 0.16666667f;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_3(float x)
    {
        return 0.5f * x * x * x - x * x + 0.66666667f;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_4(float x)
    {
        return  -0.16666667f * x * x * x + 0.5f * x * x - 0.5f * x + 0.16666667f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_1(float x)
    {
        return -0.5f * x * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_2(float x)
    {
        return  1.5f * x * x - x - 0.5f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_3(float x)
    {
        return -1.5f * x * x + 2.0f * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_4(float x)
    {
        return  -0.5f * x * x + x - 0.5f;
    }
    //Reference: Eigenvalues and Eigenvectors for 3x3 Symmetric Matrices: An Analytical Approach
    friend __device__ __host__ __forceinline__ void get_single_eigen_vector(const float* m, const float eigen_value, VECTOR* eigen_vector)
    {
        float b11 = m[0] - eigen_value;
#define b12 m[1]
#define b13 m[2]
        float b22 = m[3] - eigen_value;
#define b23 m[4]
        float b33 = m[5] - eigen_value;
        float Q, P, n;
        VECTOR local_VECTOR;
        if (fabsf(b12 * b12 - b11 * b22) > 1e-6 && fabsf(b13) > 1e-6f)
        {
            Q = (b11 * b23 - b13 * b12) / (b12 * b12 - b11 * b22);
            P = -(b23 * Q + b33) / b13;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b12 * b13 - b11 * b23) > 1e-6 && fabsf(b12) > 1e-6f)
        {
            Q = (b11 * b33 - b13 * b13) / (b12 * b13 - b11 * b23);
            P = -(b22 * Q + b23) / b12;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b22 * b13 - b12 * b23) > 1e-6 && fabsf(b11) > 1e-6f)
        {
            Q = (b12 * b33 - b23 * b13) / (b22 * b13 - b12 * b23);
            P = -(b12 * Q + b13) / b11;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b22 - b12 * b12) > 1e-6 && fabsf(b23) > 1e-6f)
        {
            P = (b12 * b23 - b13 * b22) / (b11 * b22 - b12 * b12);
            Q = -(b13 * P + b33) / b23;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b23 - b12 * b13) > 1e-6 && fabsf(b22) > 1e-6f)
        {
            P = (b12 * b33 - b13 * b23) / (b11 * b23 - b12 * b13);
            Q = -(b12 * P + b23) / b22;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b12 * b23 - b22 * b13) > 1e-6 && fabsf(b12) > 1e-6f)
        {
            P = (b22 * b33 - b23 * b23) / (b12 * b23 - b22 * b13);
            Q = -(b11 * P + b13) / b12;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b23 - b13 * b12) > 1e-6 && fabsf(b33) > 1e-6f)
        {
            P = (b13 * b22 - b12 * b23) / (b11 * b23 - b13 * b12);
            Q = -(b13 * P + b23) / b33;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
        else if (fabsf(b11 * b33 - b13 * b13) > 1e-6 && fabsf(b23) > 1e-6f)
        {
            P = (b13 * b23 - b12 * b33) / (b11 * b33 - b13 * b13);
            Q = -(b12 * P + b22) / b23;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
        else
        {
            P = (b23 * b23 - b22 * b33) / (b12 * b33 - b23 * b13);
            Q = -(b11 * P + b12) / b13;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
#undef b12
#undef b13
#undef b23
        eigen_vector[0] = local_VECTOR;
    }

//Reference: 1. Eigenvalues and Eigenvectors for 3x3 Symmetric Matrices: An Analytical Approach
//Reference: 2. github: svd_3x3_cuda
    friend __device__ __host__ __forceinline__ void get_eigen(const float* m, float* eigen_values, VECTOR* eigen_vector)
    {
        float t1, t2;
        VECTOR v1, v2, v3;
        if (m[1] == 0 && m[2] == 0 && m[4] == 0)
        {
            eigen_values[0] = m[0];
            eigen_values[1] = m[3];
            eigen_values[2] = m[5];
            v1.x = 1;
            v1.y = 0;
            v1.z = 0;
            v2.x = 0;
            v2.y = 1;
            v2.z = 0;
        }
        else if (m[1] == 0 && m[2] == 0)
        {
            t1 = m[3] - m[5];
            t1 = sqrtf(t1 * t1 / 4 + m[4] * m[4]);
            t2 = (m[3] + m[5]) / 2;
            eigen_values[0] = m[0];
            eigen_values[1] = t2 + t1;
            eigen_values[2] = t2 - t1;
            v1.x = 1;
            v1.y = 0;
            v1.z = 0;
            t1 = m[3] - eigen_values[2];
            t1 = sqrtf(t1 * t1 + m[4] * m[4]);
            v2.x = 0;
            v2.y = -m[4] / t1;
            v2.z = (m[3] - eigen_values[2]) / t1;
        }
        else if (m[1] == 0 && m[4] == 0)
        {
            t1 = m[0] - m[5];
            t1 = sqrtf(t1 * t1 / 4 + m[2] * m[2]);
            t2 = (m[0] + m[5]) / 2;
            eigen_values[0] = t2 + t1;
            eigen_values[1] = m[3];
            eigen_values[2] = t2 - t1;

            v2.x = 0;
            v2.y = 1;
            v2.z = 0;
            t1 = m[0] - eigen_values[0];
            t1 = sqrtf(t1 * t1 + m[2] * m[2]);
            v1.x = -m[2] / t1;
            v1.y = 0;
            v1.z = (m[0] - eigen_values[0]) / t1;
        }
        else if (m[2] == 0 && m[4] == 0)
        {
            t1 = m[3] - m[0];
            t1 = sqrtf(t1 * t1 / 4 + m[1] * m[1]);
            t2 = (m[3] + m[0]) / 2;
            eigen_values[0] = t2 + t1;
            eigen_values[1] = t2 - t1;
            eigen_values[2] = m[5];
            t1 = m[0] - eigen_values[0];
            t1 = sqrtf(t1 * t1 + m[1] * m[1]);
            v1.x = -m[1] / t1;
            v1.y = (m[0] - eigen_values[0]) / t1;
            v1.z = 0;
            v2.x = -(m[0] - eigen_values[0]) / t1;
            v2.y = -m[1] / t1;
            v2.z = 0;
        }
        else
        {
            float m_ = 1.0f / 3.0f * (m[0] + m[3] + m[5]);
            float a11 = m[0] - m_;
            float a22 = m[3] - m_;
            float a33 = m[5] - m_;
            float a12_sqr = m[1] * m[1];
            float a13_sqr = m[2] * m[2];
            float a23_sqr = m[4] * m[4];
            float p = 1.0f / 6.0f * (a11 * a11 + a22 * a22 + a33 * a33 + 2 * (a12_sqr + a13_sqr + a23_sqr));
            float q = 0.5f * (a11 * (a22 * a33 - a23_sqr) - a22 * a13_sqr - a33 * a12_sqr) + m[1] * m[2] * m[4];
            float sqrt_p = sqrtf(p);
            float disc = p * p * p - q * q;
            float phi = 1.0f / 3.0f * atan2f(sqrtf(fmaxf(0.0f, disc)), q);
            float c = cosf(phi);
            float s = sinf(phi);
            float sqrt_p_cos = sqrt_p * c;
            float root_three_sqrt_p_sin = sqrtf(3.0f) * sqrt_p * s;
            eigen_values[0] = m_ + 2.0f * sqrt_p;
            eigen_values[1] = m_ - sqrt_p_cos - root_three_sqrt_p_sin;
            eigen_values[2] = m_ - sqrt_p_cos + root_three_sqrt_p_sin;
            get_single_eigen_vector(m, eigen_values[0], &v1);
            get_single_eigen_vector(m, eigen_values[1], &v2);
        }
        v3 = v2 ^ v1;
        if (eigen_values[0] < eigen_values[1])
        {
            t1 = eigen_values[0];
            eigen_values[0] = eigen_values[1];
            eigen_values[1] = t1;
            t1 = v1.x;
            v1.x = v2.x;
            v2.x = t1;
            t1 = v1.y;
            v1.y = v2.y;
            v2.y = t1;
            t1 = v1.z;
            v1.z = v2.z;
            v2.z = t1;
        }
        if (eigen_values[0] < eigen_values[2])
        {
            t1 = eigen_values[0];
            eigen_values[0] = eigen_values[2];
            eigen_values[2] = t1;
            t1 = v1.x;
            v1.x = v3.x;
            v3.x = t1;
            t1 = v1.y;
            v1.y = v3.y;
            v3.y = t1;
            t1 = v1.z;
            v1.z = v3.z;
            v3.z = t1;
        }
        if (eigen_values[1] < eigen_values[2])
        {
            t1 = eigen_values[1];
            eigen_values[1] = eigen_values[2];
            eigen_values[2] = t1;
            t1 = v2.x;
            v2.x = v3.x;
            v3.x = t1;
            t1 = v2.y;
            v2.y = v3.y;
            v3.y = t1;
            t1 = v2.z;
            v2.z = v3.z;
            v3.z = t1;
        }
        eigen_vector[0] = v1;
        eigen_vector[1] = v2;
        eigen_vector[2] = v3;
    }
};

//用于记录原子组
struct ATOM_GROUP
{
    int atom_numbers;
    int *atom_serial;
};


/*XYJ备注：SAD=simple auto diff，简单自动微分
实现原理：利用操作符重载，将f(x,y)的关系同时用链式法则链接到df(x,y)上。效率肯定会有影响，暂时未明具体会影响多少
使用方法：1. 确定该部分需要求偏微分的数量，假设有1个，则后面使用的类就为SADfloat<1>或SADvector<1>，2个则为SADfloat<2>或SADvector<2>
2. 将包含微分的变量和过程用上面确定的类声明变量，其中对于想求的变量初始化时需要两个参数：本身的值和第i个变量
3. 正常计算，那么最后结果中的dval[i]即为第i个变量的微分。
使用样例：（均在No_PNC/generalized_Born.cu中）
1. 求有效伯恩半径对距离的导数：不求导数的函数为Effective_Born_Radii_Factor_CUDA，求导数的函数为GB_accumulate_Force_Energy_CUDA
2. 求GB能量对距离和有效伯恩半径的导数：不求导数的函数为GB_inej_Energy_CUDA，求导数的函数为GB_inej_Force_Energy_CUDA
*/
template<int N>
struct SADfloat
{
    float val;
    float dval[N];
    __device__ __host__ __forceinline__ SADfloat<N>()
    {
        this->val = 0;
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = 0;
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N>(int f, int id = -1)
    {
        this->val = (float)f;
        for (int i = 0; i < N; i++)
        {
            if (i != id)
                this->dval[i] = 0;
            else
                this->dval[i] = 1;
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N>(float f, int id = -1)
    {
        this->val = f;
        for (int i = 0; i < N; i++)
        {
            if (i != id)
                this->dval[i] = 0;
            else
                this->dval[i] = 1;
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N>(const SADfloat<N>& f)
    {
        this->val = f.val;
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = f.dval[i];
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N> operator-()
    {
        val = -val;
        for (int i = 0; i < N; i++)
        {
            dval[i] = -dval[i];
        }
        return *this;
    }
    __device__ __host__ __forceinline__ void operator=(const SADfloat<N>& f1)
    {
        val = f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] = f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator+=(const SADfloat<N>& f1)
    {
        val += f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] += f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator-=(const SADfloat<N>& f1)
    {
        val -= f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] -= f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator*=(const SADfloat<N>& f1)
    {
        val *= f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] = f1.val * dval[i] + val * f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator/=(const SADfloat<N>& f1)
    {
        val /= f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] = dval[i] * f1.val - f1.dval[i] * val;
            dval[i] /= f1.val * f1.val;
        }
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val <= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val == f2;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val != f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val > f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val < f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val >= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val <= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const float& f1, const SADfloat<N>& f2)
    {
        return f1 == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const float& f1, const SADfloat<N>& f2)
    {
        return f1 > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const float& f1, const SADfloat<N>& f2)
    {
        return f1 < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 <= f2.val;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator+ (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val + f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] + f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator- (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val - f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] - f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator* (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val * f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f2.val * f1.dval[i] + f1.val * f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator/ (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val / f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] * f2.val - f2.dval[i] * f1.val;
            f.dval[i] /= f2.val * f2.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> powf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = powf(x.val, y.val);
        float df_dx = y.val * powf(x.val, y.val - 1.0f);
        float df_dy = f.val * logf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> powf(const float x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = powf(x, y.val);
        float df_dy = f.val * logf(x);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> powf(const SADfloat<N>& x, const float y)
    {
        SADfloat<N> f;
        f.val = powf(x.val, y);
        float df_dx = y * powf(x.val, y - 1.0f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> expf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = expf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f.val * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> erfcf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = erfcf(x.val);
        float df_dx = -2.0f / sqrtf(CONSTANT_Pi) * expf(-x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> logf (const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = logf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = x.dval[i] / x.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> sqrtf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = sqrtf(x.val);
        float df_dx = 0.5f / f.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cbrtf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = cbrtf(x.val);
        float df_dx = 0.333333333f * powf(x, -0.66666667f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cosf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = cosf(x.val);
        float df_dx = -sinf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> sinf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = sinf(x.val);
        float df_dx = cosf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> tanf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = tanf(x.val);
        float df_dx = 2.0f / (1.0f + cosf(2.0f * x.val));
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> acosf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = acosf(x.val);
        float df_dx = -1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }    
    friend __device__ __host__ __forceinline__ SADfloat<N> asinf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = asinf(x.val);
        float df_dx = 1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> atanf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = atanf(x.val);
        float df_dx = 1.0f / (1.0f + x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fabsf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = fabsf(x.val);
        float df_dx = copysignf(1.0f, x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fmaxf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = fmaxf(x.val, y.val);
        float df_dx = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fminf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = fminf(x.val, y.val);
        float df_dx = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
};

template<int N>
struct SADvector
{
    SADfloat<N> x, y, z;
    __device__ __host__ __forceinline__ SADvector<N>()
    {
        this->x = SADfloat<N>(0);
        this->y = SADfloat<N>(0);
        this->z = SADfloat<N>(0);
    }
    __device__ __host__ __forceinline__ SADvector<N>(VECTOR v, int idx = -1, int idy = -1, int idz = -1)
    {
        this->x = SADfloat<N>(v.x, idx);
        this->y = SADfloat<N>(v.y, idy);
        this->z = SADfloat<N>(v.z, idz);
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator+ (const SADvector<N> &veca, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ SADvector<N> operator+ (const SADvector<N> &veca, const SADfloat<N> &b)
    {
        SADvector<N> vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator* (const SADfloat<N> &a, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = a*vecb.x;
        vec.y = a*vecb.y;
        vec.z = a*vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADfloat<N> operator* (const SADvector<N> &veca, const SADvector<N> &vecb)
    {
        return veca.x*vecb.x + veca.y*vecb.y + veca.z*vecb.z;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator- (const SADvector<N> &veca, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator- (const SADvector<N> &veca, const SADfloat<N> &b)
    {
        SADvector<N> vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator/ (const SADvector<N> &veca, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator/ (const float &a, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator^ (const SADvector<N> &veca, const SADvector<N> &vecb)
    {
        SADvector<N> vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> Get_Periodic_Displacement(const SADvector<N> vec_a, const SADvector<N> vec_b, const SADvector<N> box_length)
    {
        SADvector<N> dr;
        dr = vec_a - vec_b;
        dr.x.val = dr.x.val - floorf(dr.x.val / box_length.x.val + 0.5) * box_length.x.val;
        dr.y.val = dr.y.val - floorf(dr.y.val / box_length.y.val + 0.5) * box_length.y.val;
        dr.z.val = dr.z.val - floorf(dr.z.val / box_length.z.val + 0.5) * box_length.z.val;
        for (int i = 0; i < N; i++)
        {
            dr.x.dval[i] = dr.x.val / box_length.x.val * box_length.x.dval[i] + dr.x.dval[i];
            dr.y.dval[i] = dr.y.val / box_length.y.val * box_length.y.dval[i] + dr.y.dval[i];
            dr.z.dval[i] = dr.z.val / box_length.z.val * box_length.z.dval[i] + dr.z.dval[i];
        }
        return dr;
    }
};
)JIT";
