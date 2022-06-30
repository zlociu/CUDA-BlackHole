#pragma once

#include <stdlib.h>

typedef struct
{
	float a11, a12, a13;
	float a21, a22, a23;
	float a31, a32, a33;
} matrix3;

__inline__ __device__ matrix3 MatrixZero()
{
	return matrix3{ 0,0,0, 0,0,0, 0,0,0 };
}

__inline__ __device__ matrix3 MatrixOne()
{
	return matrix3{ 1,1,1, 1,1,1, 1,1,1 };
}

/*
	mul [1  2  3]

	[2]	[2  4  6]
	[4] [4  8 12]
	[6] [6 12 18]
*/


__inline__ __device__ matrix3 MultiplyScalar(matrix3 m, float scalar)
{
	m.a11 *= scalar;
	m.a12 *= scalar;
	m.a13 *= scalar;

	m.a21 *= scalar;
	m.a22 *= scalar;
	m.a23 *= scalar;

	m.a31 *= scalar;
	m.a32 *= scalar;
	m.a33 *= scalar;

	return m;
}