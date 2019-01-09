//#define __device__
//#define __global__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_string.cuh"
//#include "cuda_list.cuh"
//#include "cuda_stack.cuh"
#include <stdio.h>
#include "CudaParser.h"

namespace cu
{
	template <class T>
	class TestHeader
	{
		cuda_map<cuda_string, T> m_arguments;
		cuda_vector<cuda_string> m_parameters;
		cuda_string function_name;
		mutable return_wrapper_t<void> construction_success_code;
	public:
		__device__ TestHeader() = default;
		__device__ TestHeader(const char* expression, std::size_t expression_len, char** endPtr)
		{
			char* begPtr = (char*)(expression);
			cuda_list<cuda_string> params;
			construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::Success);

			bool isOpeningBracket = false;
			bool isClosingBracket = false;
			std::size_t commaCount = 0;

			while (*begPtr != '=' && begPtr < expression + expression_len)
			{
				if (isalpha(*begPtr))
				{
					auto l_endptr = begPtr + 1;
					for (; isalnum(*l_endptr); ++l_endptr);
					if (this->function_name.empty())
						this->function_name = cuda_string(begPtr, l_endptr);
					else
					{
						if (!isOpeningBracket)
							construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedToken);
						auto param_name = cuda_string(begPtr, l_endptr);
						//if (!m_arguments.insert(thrust::pair<cuda_string, T>(param_name, T())).second)
							construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotUnique);
						params.push_back(std::move(param_name));
					}
					begPtr = l_endptr;
				}

				if (*begPtr == ' ')
				{
					begPtr += 1;
					continue;
				}

				if (*begPtr == '(')
				{
					if (isOpeningBracket)
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotUnique);
					isOpeningBracket = true;
					begPtr += 1;
				}

				if (*begPtr == ',') //a-zA_Z0-9
				{
					commaCount += 1;
					begPtr += 1;
				}

				if (*begPtr == ')')
				{
					if (!isOpeningBracket)
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotUnique);
					if (isClosingBracket)
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotUnique);
					isClosingBracket = true;
					begPtr += 1;
				}
			}
			//m_parameters.reserve(params.size());
			for (auto& param : params)
				m_parameters.push_back(std::move(param.data));
			*endPtr = begPtr;
		}
		TestHeader(const TestHeader<T>&) = delete;
		TestHeader& operator=(const TestHeader<T>&) = delete;
		__device__ TestHeader(TestHeader&&) = default;
		__device__ TestHeader& operator=(TestHeader&&) = default;
	};
}

__global__ void memset_expr(double* vec, std::size_t n, const char* pStr, std::size_t cbStr)
{
	char* endptr;
	auto header = cu::TestHeader<double>("f(x) = x", 8, &endptr);
	//header.push_argument("x", 1, 12);
	auto someptr = make_cuda_device_unique_ptr<cu::cuda_string>();
	*someptr = cu::cuda_string("abc");
	auto someotherptr = std::move(someptr);
	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
	{
		auto rw = cu::return_wrapper_t<double>(123);// = header.get_argument("x", 1);
		if (rw.get() != nullptr)
			vec[i] = cu::stod(cu::cuda_string(pStr, pStr + cbStr));// + rw.value();
	}
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
	memset_expr<<<1, 1>>>(V_d.get(), sizeof(V) / sizeof(double), pStr_d.get(), sizeof(pStr) - 1);

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
