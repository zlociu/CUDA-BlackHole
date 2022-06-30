#pragma once

#include <stdlib.h>
#include <math.h>
#include "constants.h"

//#include "cuda_runtime.h"

typedef struct
{
	float x, y;

} vector2;

// vector with all 1
__inline__ __host__ __device__ vector2 Vector2One()
{
	return vector2{ 1.f, 1.f };
}

// vector with all 0
__inline__ __host__ __device__ vector2 Vector2Zero()
{
	return vector2{ 0.f, 0.f };
}

__inline__ __host__ __device__ vector2 operator+(vector2 v1, vector2 v2)
{
	return vector2{ v1.x + v2.x, v1.y + v2.y };
}

__inline__ __host__ __device__ vector2 operator-(vector2 v1, vector2 v2)
{
	return vector2{ v1.x - v2.x, v1.y - v2.y };
}

__inline__ __host__ __device__ vector2 operator*(vector2 v, float scalar)
{
	return vector2{ v.x * scalar, v.y * scalar };
}

__inline__ __host__ __device__ vector2 operator/(vector2 v, float scalar)
{
	return vector2{ v.x / scalar, v.y / scalar };
}

__inline__ __host__ __device__ vector2 operator*(float scalar, vector2 v)
{
	return vector2{ v.x * scalar, v.y * scalar };
}

__inline__ __host__ __device__ vector2 operator/(float scalar, vector2 v)
{
	return vector2{ scalar / v.x, scalar / v.y };
}

__inline__ __device__ float VectorLenght(vector2 v)
{
	//return sqrtf(v.x * v.x + v.y * v.y);
	return hypotf(v.x, v.y);
}

__inline__ __device__ float VectorLenghtSquared(vector2 v)
{
	return v.x * v.x + v.y * v.y;
}

__inline__ __device__ vector2 VectorNormalize(vector2 v)
{
	//return v / VectorLenght(v);
	return rhypotf(v.x, v.y) * v;
}

__inline__ __device__ float VectorMultiplyHV(vector2 v1, vector2 v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}

/*
__device__ vector2 MatrixMultiplyVector(matrix3 m, vector2 v)
{
	vector2 result;
	result.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z;
	result.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z;
	result.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z;

	return result;
}
*/

