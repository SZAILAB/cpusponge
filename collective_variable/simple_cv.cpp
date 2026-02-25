#include "simple_cv.h"


REGISTER_CV_STRUCTURE(CV_POSITION, "position_x", 0);
REGISTER_CV_STRUCTURE(CV_POSITION, "position_y", 1);
REGISTER_CV_STRUCTURE(CV_POSITION, "position_z", 2);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_x", 3);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_y", 4);
REGISTER_CV_STRUCTURE(CV_POSITION, "scaled_position_z", 5);

static __global__ void position_x_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads, VECTOR* box_grads)
{
	float x = crd[atom].x;
	value[0] = x;
	crd_grads[atom].x = 1;
	box_grads[0].x = x / box_length.x;
}

static __global__ void position_y_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads, VECTOR* box_grads)
{
	float y = crd[atom].y;
	value[0] = y;
	crd_grads[atom].y = 1;
	box_grads[0].y = y / box_length.y;
}

static __global__ void position_z_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads, VECTOR* box_grads)
{
	float z = crd[atom].z;
	value[0] = z;
	crd_grads[atom].z = 1;
	box_grads[0].z = z / box_length.z;
}

static __global__ void scaled_position_x_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads)
{
	float x = crd[atom].x;
	float temp = 1.0f / box_length.x;
	value[0] = x * temp;
	crd_grads[atom].x = temp;
}

static __global__ void scaled_position_y_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads)
{
	float y = crd[atom].y;
	float temp = 1.0f / box_length.y;
	value[0] = y * temp;
	crd_grads[atom].y = temp;
}

static __global__ void scaled_position_z_get_all(const int atom, const VECTOR* crd, const VECTOR box_length, float* value, VECTOR* crd_grads)
{
	float z = crd[atom].z;
	float temp = 1.0f / box_length.z;
	value[0] = z * temp;
	crd_grads[atom].z = temp;
}

void CV_POSITION::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
	atom = manager->Ask_For_Int_Parameter(module_name, "atom", 1, 2);
	Super_Initial(manager, atom_numbers, module_name);
}

void CV_POSITION::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
	need = Check_Whether_Computed_At_This_Step(step, need);
	if (need != CV_NEED_NONE)
	{
		if (strcmp(type_name, "position_x") == 0)
		{
				CUDA_LAUNCH_STREAM(position_x_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads, box_grads);
		}
		else if (strcmp(type_name, "position_y") == 0)
		{
				CUDA_LAUNCH_STREAM(position_y_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads, box_grads);
		}
		else if (strcmp(type_name, "position_z") == 0)
		{
				CUDA_LAUNCH_STREAM(position_z_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads, box_grads);
		}
		else if (strcmp(type_name, "scaled_position_x") == 0)
		{
				CUDA_LAUNCH_STREAM(scaled_position_x_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads);
		}
		else if (strcmp(type_name, "scaled_position_y") == 0)
		{
				CUDA_LAUNCH_STREAM(scaled_position_y_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads);
		}
		else if (strcmp(type_name, "scaled_position_z") == 0)
		{
				CUDA_LAUNCH_STREAM(scaled_position_z_get_all, dim3(1), dim3(1), 0, cuda_stream,
					*atom, crd, box_length, d_value, crd_grads);
		}
		cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
	}
	Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_x", 0);
REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_y", 1);
REGISTER_CV_STRUCTURE(CV_BOX_LENGTH, "box_length_z", 2);

static __global__ void box_length_x_get_all(const VECTOR box_length, float* value, VECTOR* box_grads)
{
	value[0] = box_length.x;
	box_grads[0].x = 1;
}

static __global__ void box_length_y_get_all(const VECTOR box_length, float* value, VECTOR* box_grads)
{
	value[0] = box_length.y;
	box_grads[0].y = 1;
}

static __global__ void box_length_z_get_all(const VECTOR box_length, float* value, VECTOR* box_grads)
{
	value[0] = box_length.z;
	box_grads[0].z = 1;
}

void CV_BOX_LENGTH::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
	Super_Initial(manager, atom_numbers, module_name);
}

void CV_BOX_LENGTH::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
	need = Check_Whether_Computed_At_This_Step(step, need);
	if (need != CV_NEED_NONE)
	{
		if (strcmp(type_name, "box_length_x") == 0)
		{
				CUDA_LAUNCH_STREAM(box_length_x_get_all, dim3(1), dim3(1), 0, cuda_stream,
					box_length, d_value, box_grads);
			value = box_length.x;
		}
		else if (strcmp(type_name, "box_length_y") == 0)
		{
				CUDA_LAUNCH_STREAM(box_length_y_get_all, dim3(1), dim3(1), 0, cuda_stream,
					box_length, d_value, box_grads);
			value = box_length.y;
		}
		else if (strcmp(type_name, "box_length_z") == 0)
		{
				CUDA_LAUNCH_STREAM(box_length_z_get_all, dim3(1), dim3(1), 0, cuda_stream,
					box_length, d_value, box_grads);
			value = box_length.z;
		}
	}
	Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_DISTANCE, "distance", 0);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_x", 1);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_y", 2);
REGISTER_CV_STRUCTURE(CV_DISTANCE, "displacement_z", 3);

static __global__ void distance_get_all(const int atom0, const int atom1, const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	VECTOR dr = Get_Periodic_Displacement(uint_crd[atom1], uint_crd[atom0], scaler);
	float dr_abs = norm3df(dr.x, dr.y, dr.z);
	float dr_1 = 1.0f / dr_abs;
	VECTOR drdx = dr_1 * dr;
	value[0] = dr_abs;
	crd_grads[atom1].x = drdx.x;
	crd_grads[atom1].y = drdx.y;
	crd_grads[atom1].z = drdx.z;
	crd_grads[atom0].x = -drdx.x;
	crd_grads[atom0].y = -drdx.y;
	crd_grads[atom0].z = -drdx.z;
	box_grads[0] = dr / box_length;
}

static __global__ void displacement_x_get_all(const int atom0, const int atom1, const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	VECTOR dr = Get_Periodic_Displacement(uint_crd[atom1], uint_crd[atom0], scaler);
	value[0] = dr.x;
	crd_grads[atom1].x = 1;
	crd_grads[atom0].x = -1;
	box_grads[0].x = dr.x / box_length.x;
}

static __global__ void displacement_y_get_all(const int atom0, const int atom1, const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	VECTOR dr = Get_Periodic_Displacement(uint_crd[atom1], uint_crd[atom0], scaler);
	value[0] = dr.y;
	crd_grads[atom1].y = 1;
	crd_grads[atom0].y = -1;
	box_grads[0].y = dr.y / box_length.y;
}

static __global__ void displacement_z_get_all(const int atom0, const int atom1, const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	VECTOR dr = Get_Periodic_Displacement(uint_crd[atom1], uint_crd[atom0], scaler);
	value[0] = dr.z;
	crd_grads[atom1].z = 1;
	crd_grads[atom0].z = -1;
	box_grads[0].z = dr.z / box_length.z;
}

void CV_DISTANCE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
	atom = manager->Ask_For_Int_Parameter(module_name, "atom", 2, 2);
	Super_Initial(manager, atom_numbers, module_name);
}

void CV_DISTANCE::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
	need = Check_Whether_Computed_At_This_Step(step, need);
	if (need != CV_NEED_NONE)
	{
		if (strcmp(type_name, "distance") == 0)
				CUDA_LAUNCH_STREAM(distance_get_all, dim3(1), dim3(1), 0, cuda_stream,
					atom[0], atom[1], uint_crd, scaler, box_length, d_value, crd_grads, box_grads);
		else if (strcmp(type_name, "displacement_x"))
				CUDA_LAUNCH_STREAM(displacement_x_get_all, dim3(1), dim3(1), 0, cuda_stream,
					atom[0], atom[1], uint_crd, scaler, box_length, d_value, crd_grads, box_grads);
		else if (strcmp(type_name, "displacement_y"))
				CUDA_LAUNCH_STREAM(displacement_y_get_all, dim3(1), dim3(1), 0, cuda_stream,
					atom[0], atom[1], uint_crd, scaler, box_length, d_value, crd_grads, box_grads);
		else if (strcmp(type_name, "displacement_z"))
				CUDA_LAUNCH_STREAM(displacement_z_get_all, dim3(1), dim3(1), 0, cuda_stream,
					atom[0], atom[1], uint_crd, scaler, box_length, d_value, crd_grads, box_grads);
		cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
	}
	Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_ANGLE, "angle", 0);

static __global__ void angle_get_all(const int atom0, const int atom1, const int atom2, 
	const VECTOR* crd, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	SADvector<12> box_length_with_grads(box_length, 0, 1, 2);
	SADvector<12> r0(crd[atom0], 3, 4, 5);
	SADvector<12> r1(crd[atom1], 6, 7, 8);
	SADvector<12> r2(crd[atom2], 9, 10, 11);

	//复用变量r0减少寄存器使用，表示dr01
	r0 = Get_Periodic_Displacement(r0, r1, box_length_with_grads);
	//复用变量r2减少寄存器使用，表示dr21
	r2 = Get_Periodic_Displacement(r2, r1, box_length_with_grads);
	SADfloat<12> temp = 1.0f / (r0 * r0) / (r2 * r2);
	
	temp = sqrtf(temp) * (r0 * r2);
	if (temp.val > 0.999999f)
		temp.val = 0.999999f;
	else if (temp.val < -0.999999f)
		temp.val = -0.999999f;
	temp = acosf(temp);
	value[0] = temp.val;
	box_grads[0] = { temp.dval[0], temp.dval[1], temp.dval[2] };
	crd_grads[atom0] = { temp.dval[3], temp.dval[4], temp.dval[5] };
	crd_grads[atom1] = { temp.dval[6], temp.dval[7], temp.dval[8] };
	crd_grads[atom2] = { temp.dval[9], temp.dval[10], temp.dval[11] };
}

void CV_ANGLE::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
	atom = manager->Ask_For_Int_Parameter(module_name, "atom", 3, 2);
	Super_Initial(manager, atom_numbers, module_name);
}

void CV_ANGLE::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
	need = Check_Whether_Computed_At_This_Step(step, need);
	if (need != CV_NEED_NONE)
	{
			CUDA_LAUNCH_STREAM(angle_get_all, dim3(1), dim3(1), 0, cuda_stream,
				atom[0], atom[1], atom[2], crd, box_length, d_value, crd_grads, box_grads);
		cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
	}
	Record_Update_Step_Of_Fast_Computing_CV(step, need);
}

REGISTER_CV_STRUCTURE(CV_DIHEDRAL, "dihedral", 0);

static __global__ void dihedral_get_all(const int atom0, const int atom1, const int atom2, const int atom3,
	const VECTOR* crd, const VECTOR box_length, float* value, VECTOR *crd_grads, VECTOR* box_grads)
{
	SADvector<15> box_length_with_grads(box_length, 0, 1, 2);
	SADvector<15> r0(crd[atom0], 3, 4, 5);
	SADvector<15> r1(crd[atom1], 6, 7, 8);
	SADvector<15> r2(crd[atom2], 9, 10, 11);
	SADvector<15> r3(crd[atom3], 12, 13, 14);
	//复用变量r0减少寄存器使用，表示dr01
	r0 = Get_Periodic_Displacement(r1, r0, box_length_with_grads);
	//复用变量r1减少寄存器使用，表示dr21
	r1 = Get_Periodic_Displacement(r2, r1, box_length_with_grads);
	//复用变量r2减少寄存器使用，表示dr23
	r2 = Get_Periodic_Displacement(r3, r2, box_length_with_grads);
	//复用变量r3减少寄存器使用，表示dr01叉乘dr21
	r3 = r0 ^ r1;
	//复用变量r0，表示dr23叉乘dr21
	r0 = r1 ^ r2;

	SADfloat<15> temp = 1.0f / (r3 * r3) / (r0 * r0);
	temp = sqrtf(temp) * (r0 * r3);
	if (temp.val > 0.999999f)
		temp.val = 0.999999f;
	else if (temp.val < -0.999999f)
		temp.val = -0.999999f;
	temp = acosf(temp);
    if ((r3 * r2).val < 0)
        temp = -temp;
	value[0] = temp.val;
	box_grads[0] = { temp.dval[0], temp.dval[1], temp.dval[2] };
	crd_grads[atom0] = { temp.dval[3], temp.dval[4], temp.dval[5] };
	crd_grads[atom1] = { temp.dval[6], temp.dval[7], temp.dval[8] };
	crd_grads[atom2] = { temp.dval[9], temp.dval[10], temp.dval[11] };
	crd_grads[atom3] = { temp.dval[12], temp.dval[13], temp.dval[14] };
}

void CV_DIHEDRAL::Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers, const char* module_name)
{
	atom = manager->Ask_For_Int_Parameter(module_name, "atom", 4, 2);
	Super_Initial(manager, atom_numbers, module_name);
}

void CV_DIHEDRAL::Compute(int atom_numbers, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, VECTOR* crd, VECTOR box_length, int need, int step)
{
	need = Check_Whether_Computed_At_This_Step(step, need);
	if (need != CV_NEED_NONE)
	{
			CUDA_LAUNCH_STREAM(dihedral_get_all, dim3(1), dim3(1), 0, cuda_stream,
				atom[0], atom[1], atom[2], atom[3], crd, box_length, d_value, crd_grads, box_grads);
		cudaMemcpyAsync(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost, this->cuda_stream);
	}
	Record_Update_Step_Of_Fast_Computing_CV(step, need);
}
