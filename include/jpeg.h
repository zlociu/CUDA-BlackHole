#pragma once

#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "nvjpeg.h"

#include <stdio.h>
#include <stdlib.h>

#include "config.h"

const char* nvjpegGetErrorString(nvjpegStatus_t status);

#define nvjpegAssert(result) if((result) != NVJPEG_STATUS_SUCCESS){printf("%s in line: %d in file: %s\n\n", nvjpegGetErrorString((result)), __LINE__, __FILE__); exit(-1);}

__global__ void InvertBitmap(unsigned char* bitmap);

void SaveJPEGtoFile(unsigned char* host_bitmap, const char* filename);