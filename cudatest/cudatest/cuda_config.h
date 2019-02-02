#include <stack>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>

#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

#define __host__
#define __device__
#define __global__

typedef int* cudaStream_t;

inline int cudaMalloc(void** ppResult, std::size_t cb)
{
	*ppResult = malloc(cb);
	if (!*ppResult)
		return -1;
	return 0;
}

inline int cudaFree(void* ptr)
{
	free(ptr);
	return 0;
}


#pragma warning(disable:4996)

#define CU_BEGIN namespace cu { 
#define CU_END }

#endif // !CUDA_CONFIG

