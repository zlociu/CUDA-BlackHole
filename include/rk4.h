#pragma once
# include <stdlib.h>
#include "vector3.h"
#include "constants.h"
#include "config.h"

// 'y' array has len: 2 
__inline__ __device__ void rk4step(vector3 y1, vector3 y2, float h2, vector3* result)
{
    result[0] = y2;
    result[1] = -1.5f * h2 * y1 / powf(VectorLenghtSquared(y1), 2.5f);
}

__inline__ __device__ void rk4(vector3 point, vector3 velocity, float h2, vector3* result)
{
    vector3 k[8];
    rk4step(point, velocity, h2, k);
    rk4step(point + (0.5f * STEP) * k[0],
        velocity + (0.5f * STEP) * k[1],
        h2, k + 2);
    rk4step(point + (0.5f * STEP) * k[2],
        velocity + (0.5f * STEP) * k[3],
        h2, k + 4);
    rk4step(point + STEP * k[4],
        velocity + STEP * k[5],
        h2, k + 6);

    result[0] = (STEP / 6.f) * (k[0] + 2.f * k[2] + 2.f * k[4] + k[6]);
    result[1] = (STEP / 6.f) * (k[1] + 2.f * k[3] + 2.f * k[5] + k[7]);
}

/*
vector3 y1, y2;
        y1 = point;
        y2 = velocity;
        vector3 k1[2], k2[2], k3[2], k4[2];
        rk4(y1, y2, h2, k1);
        rk4(y1 + (0.5f * rkstep) * k1[0], y2 + (0.5f * rkstep) * k1[1], h2, k2);
        rk4(y1 + (0.5f * rkstep) * k2[0], y2 + (0.5f * rkstep) * k2[1], h2, k3);
        rk4(y1 + rkstep * k3[0], y2 + rkstep * k3[1], h2, k4);

        vector3 increment0 = (rkstep / 6.f) * (k1[0] + 2.f * k2[0] + 2.f * k3[0] + k4[0]);
        vector3 increment1 = (rkstep / 6.f) * (k1[1] + 2.f * k2[1] + 2.f * k3[1] + k4[1]);

        velocity = velocity + increment1;
        point = point + increment0;
*/


