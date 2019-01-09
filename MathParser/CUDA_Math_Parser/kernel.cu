//#define __device__
//#define __global__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_string.cuh"
//#include "cuda_list.cuh"
//#include "cuda_stack.cuh"
#include <stdio.h>
//#include "CudaParser.h"

__global__ void memset_expr(double* vec, std::size_t n, const char* pStr, std::size_t cbStr)
{
	auto i = threadIdx.x * blockIdx.x * blockDim.x;
	if (i < n)
		vec[i] = cu::stod(cu::cuda_string(pStr, pStr + cbStr));
}

int main()
{
	cudaError_t cudaStatus;
	const char pStr[] = "3.14";
	double V[1000];

	auto pStr_d = make_cuda_unique_ptr<char>(sizeof(pStr));
	auto V_d = make_cuda_unique_ptr<double>(sizeof(V) / sizeof(double));

	cudaStatus = cudaMemcpy(pStr_d.get(), pStr, sizeof(pStr) - 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -1;
	memset_expr<<<2, 500>>>(V_d.get(), sizeof(V) / sizeof(double), pStr_d.get(), sizeof(pStr) - 1);

	/*cuda_string expression = "f(x, y) = min(x, 5, y) + min(y, 5, x) + max(x, 5, y) + max(y, 5, x)";
	Mathexpr<double> mathexpr(expression);
	cuda_vector<double> v;
	v.push_back(1);
	v.push_back(10);
	mathexpr.init_variables(v);*/
	//std::cout << "Value: " << mathexpr.compute() << "\n";

	//cuda_list<double> l;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		return -2;
	}

	cudaStatus = cudaMemcpy(V, V_d.get(), sizeof(V), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return -3;

	//printf("%d", l.front());

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -4;
	}

	return 0;
}
