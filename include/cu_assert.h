#pragma once


#define cudaAssert(result) if((result) != cudaSuccess){printf("%s in line: %d in file: %s\n\n", cudaGetErrorString((result)), __LINE__, __FILE__); exit(1);}