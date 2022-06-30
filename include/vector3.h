#pragma once

#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "matrix3.h"

struct vector3
{
	float x, y, z;

};

// vector with all 1
__inline__ __host__ __device__ vector3 Vector3One()
{
	return vector3{ 1.f, 1.f, 1.f };
}

// vector with all 0
__inline__ __host__ __device__ vector3 Vector3Zero()
{
	return vector3{ 0.f, 0.f, 0.f };
}

__inline__ __host__ __device__ vector3 operator+(vector3 v1, vector3 v2)
{
	return vector3{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

__inline__ __host__ __device__ vector3 operator-(vector3 v1, vector3 v2)
{
	return vector3{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

__inline__ __host__ __device__ vector3 operator*(vector3 v, float scalar)
{
	return vector3{ v.x * scalar, v.y * scalar, v.z * scalar };
}

__inline__ __host__ __device__ vector3 operator/(vector3 v, float scalar)
{
	return vector3{ v.x / scalar, v.y / scalar, v.z / scalar };
}

__inline__ __host__ __device__ vector3 operator*(float scalar, vector3 v)
{
	return vector3{ v.x * scalar, v.y * scalar, v.z * scalar };
}

__inline__ __device__ vector3 operator/(float scalar, vector3 v)
{
	return vector3{ scalar / v.x ,scalar / v.y, scalar / v.z };
}


// iloczyn wektorowy AxB (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
__inline__ __host__ __device__ vector3 VectorCross(vector3 v1, vector3 v2)
{
	return vector3{ v1.y * v2.z - v1.z * v2.y,
					v1.z * v2.x - v1.x * v2.z,
					v1.x * v2.y - v1.y * v2.x };
}

//dlugosc wektora V ( sqrt(v.x*v.x + v.y*v.y + v.z*v.z) )
__inline__ __device__ float VectorLenght(vector3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	//return norm3df(v.x, v.y, v.z);
}

//dlugosc wektora V ( sqrt(v.x*v.x + v.y*v.y + v.z*v.z) )
__inline__ __host__ float VectorLenghtHost(vector3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

//dlugosc wektora do kwadratu
__inline__ __device__ float VectorLenghtSquared(vector3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

//dlugosc wektora do kwadratu
__inline__ __host__ float VectorLenghtSquaredHost(vector3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

// zwraca wektor normalny, czyli wektor o dlugosci 1
__inline__ __device__ vector3 VectorNormalize(vector3 v)
{
	return v / VectorLenght(v);
	//return rnorm3df(v.x, v.y, v.z) * v;
}

// zwraca wektor normalny, czyli wektor o dlugosci 1
__inline__ __host__ vector3 VectorNormalizeHost(vector3 v)
{
	return v / VectorLenghtHost(v);
}

__inline__ __device__ float VectorMultiplyHV(vector3 v1, vector3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


__inline__ __device__ matrix3 VectorMultiplyVH(vector3 v1, vector3 v2)
{
	matrix3 m1;
	m1.a11 = v1.x * v2.x;
	m1.a12 = v1.x * v2.y;
	m1.a13 = v1.x * v2.z;

	m1.a21 = v1.y * v2.x;
	m1.a22 = v1.y * v2.y;
	m1.a23 = v1.y * v2.z;

	m1.a31 = v1.z * v2.x;
	m1.a32 = v1.z * v2.y;
	m1.a33 = v1.z * v2.z;

	return m1;
}

__inline__ __device__ vector3 MatrixMultiplyVector(matrix3 m, vector3 v)
{
	vector3 result;
	result.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z;
	result.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z;
	result.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z;

	return result;
}