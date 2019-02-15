//#define __device__
//#define __global__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_string.cuh"
#include "cuda_list.cuh"
//#include "cuda_stack.cuh"
#include <stdio.h>
#include "CudaParser.cuh"

typedef thrust::complex<double> number_type;

__device__ cu::Mathexpr<number_type>* g_pExpr;

__global__ void memset_expr(number_type* vec, std::size_t n, const char* pStr, std::size_t cbStr)
{
	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i == 0)
		g_pExpr = new cu::Mathexpr<number_type>(pStr, cbStr);
	__syncthreads();
	if (i < n)
	{
		auto& m = *g_pExpr;
		vec[i] = m(10).value();
	}
	__syncthreads();
	if (!i)
		delete g_pExpr;
}

int main()
{
	cudaError_t cudaStatus;
	//const char pStr[] = "f(x) = 2*j1(0.1*3.14*sin(x)) / (0.1*3.14*sin(x))";
	const char pStr[] = "f(x) = abs(polar(2, 5)) * x";
	number_type V[50];
	std::size_t cbStack;

	cudaStatus = cudaDeviceGetLimit(&cbStack, cudaLimitStackSize);
	if (cudaStatus != 0)
		return -6;

	cudaStatus = cudaDeviceSetLimit(cudaLimitStackSize, 1 << 13);
	if (cudaStatus != 0)
		return -5;

	auto pStr_d = make_cuda_unique_ptr<char>(sizeof(pStr));
	auto V_d = make_cuda_unique_ptr<number_type>(sizeof(V) / sizeof(number_type));

	cudaStatus = cudaMemcpy(pStr_d.get(), pStr, sizeof(pStr) - 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -1;
	memset_expr<<<1, sizeof(V) / sizeof(number_type)>>>(V_d.get(), sizeof(V) / sizeof(number_type), pStr_d.get(), sizeof(pStr) - 1);

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

	for (auto elem:V)
		std::cout << elem << " ";
	std::cout << "\n";

	return 0;
}
