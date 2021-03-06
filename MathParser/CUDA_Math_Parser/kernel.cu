#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "CudaParser.cuh"

__device__ cu::Mathexpr<thrust::complex<double>>* g_pExpr;

__global__ void memset_expr(cu::CudaParserErrorCodes* pCode, thrust::complex<double>* vec, std::size_t n, const char* pStr, std::size_t cbStr)
{
	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i == 0)
		g_pExpr = new cu::Mathexpr<thrust::complex<double>>(pCode, pStr, cbStr);
	__syncthreads();
	if (*pCode == cu::CudaParserErrorCodes::Success && i < n)
	{
		auto& m = *g_pExpr;
		auto rv = m(thrust::complex<double>(i, i));
		*pCode = rv.return_code();
		if (bool(rv))
			vec[i] = rv.value();
	}
}

int main()
{
	cudaError_t cudaStatus;
	const char pStr[] = "f(x) = 2*j1(0.1*3.14*sin(x)) / (0.1*3.14*sin(x))";
	thrust::complex<double> V[2048];
	std::size_t cbStack;

	cudaStatus = cudaDeviceGetLimit(&cbStack, cudaLimitStackSize);
	if (cudaStatus != 0)
		return -6;

	cudaStatus = cudaDeviceSetLimit(cudaLimitStackSize, 1 << 13);
	if (cudaStatus != 0)
		return -5;

	auto pStr_d = make_cuda_unique_ptr<char>(sizeof(pStr));
	if (!pStr_d)
		return -100;
	auto V_d = make_cuda_unique_ptr<thrust::complex<double>>(sizeof(V) / sizeof(thrust::complex<double>));
	if (!V_d)
		return -100;
	auto pCode = make_cuda_unique_ptr<cu::CudaParserErrorCodes>();
	if (!pCode)
		return -100;

	cudaStatus = cudaMemcpy(pStr_d.get(), pStr, sizeof(pStr) - 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return -1;
	memset_expr<<<1, sizeof(V) / sizeof(thrust::complex<double>)>>>(pCode.get(), V_d.get(), sizeof(V) / sizeof(thrust::complex<double>), pStr_d.get(), sizeof(pStr) - 1);

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

	cu::CudaParserErrorCodes errc;
	cudaStatus = cudaMemcpy(&errc, pCode.get(), sizeof(cu::CudaParserErrorCodes), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return -3;
	if (errc == cu::CudaParserErrorCodes::Success)
	{
		cudaStatus = cudaMemcpy(V, V_d.get(), sizeof(V), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			return -3;
		for (auto elem:V)
			std::cout << elem << " ";
		std::cout << "\n";
	}else
	{
		printf("CUDA kernel returned code %d (%s)", int(errc), strerror(errc));
		return -50;
	}

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
