//#define __device__
//#define __global__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_string.cuh"
#include "cuda_list.cuh"
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
						auto param_name = cu::cuda_string(begPtr, l_endptr);
						auto res = m_arguments.insert(make_cuda_pair<cu::cuda_string, T>(param_name, T()));
						if (!res.second)
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
			m_parameters.reserve(params.size());
			for (auto& param : params)
				m_parameters.push_back(std::move(param.data));
			*endPtr = begPtr;
		}
		TestHeader(const TestHeader<T>&) = delete;
		TestHeader& operator=(const TestHeader<T>&) = delete;
		__device__ TestHeader(TestHeader&&) = default;
		__device__ TestHeader& operator=(TestHeader&&) = default;
		__device__ return_wrapper_t<void> push_argument(const char* name, std::size_t parameter_name_size, const T& value)
		{
			auto it = m_arguments.find(cuda_string(name, name + parameter_name_size));
			if (it == m_arguments.end())
			{
				construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotFound);
				return construction_success_code;
			}
			it->second = value;
			return construction_success_code;
		}
		__device__ return_wrapper_t<const T&> get_argument(const char* parameter_name, std::size_t parameter_name_size) const //call this from Variable::operator()().
		{
			auto it = m_arguments.find(cuda_string(parameter_name, parameter_name + parameter_name_size));
			if (it == m_arguments.end())
			{
				this->construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotFound);
				return return_wrapper_t<const T&>(CudaParserErrorCodes::ParameterIsNotFound);
			}
			return return_wrapper_t<const T&>(it->second, CudaParserErrorCodes::Success);
		}
		__device__ auto get_argument(const char* parameter_name, std::size_t parameter_name_size) //call this from Variable::operator()().
		{
			auto carg = const_cast<const TestHeader<T>*>(this)->get_argument(parameter_name, parameter_name_size);
			if (carg.return_code() != CudaParserErrorCodes::Success)
				return return_wrapper_t<T&>(carg.return_code());
			return return_wrapper_t<T&>(const_cast<T&>(carg.value()), carg.return_code());
		}
		__device__ return_wrapper_t<T&> get_argument_by_index(std::size_t index)  //call this from Variable::operator()().
		{
			return this->get_argument(m_parameters[index].c_str(), m_parameters[index].size());
		}
		__device__ std::size_t get_required_parameter_count() const
		{
			return m_parameters.size();
		}
		__device__ const char* get_function_name() const
		{
			return function_name.c_str();
		}
		__device__ size_t get_name_length() const
		{
			return function_name.size();
		}
		__device__ return_wrapper_t<std::size_t> get_param_index(const cuda_string& param_name)
		{
			for (std::size_t i = 0; i < this->m_parameters.size(); ++i)
			{
				if (this->m_parameters[i] == param_name)
					return return_wrapper_t<std::size_t>(i);
			}
			return return_wrapper_t<std::size_t>(CudaParserErrorCodes::ParameterIsNotFound);
		}
	};
}

__global__ void memset_expr(double* vec, std::size_t n, const char* pStr, std::size_t cbStr)
{
	//auto someptr = make_cuda_device_unique_ptr<cu::cuda_string>();
	//*someptr = cu::cuda_string("abc");
	//auto somecopy = *someptr;
	//auto someotherptr = std::move(someptr);

	char* endptr;
	auto header = cu::Header<double>("f(x) = x", 8, &endptr);
	header.push_argument("x", 1, 12);

	cu::Mathexpr<double> math("f(x) = x", 8);

	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
	{
		auto rw = header.get_argument("x", 1);
		auto k = rw.value();
		if (rw.get() != nullptr)
			vec[i] = rw.value();

			//vec[i] = cu::stod(cu::cuda_string(pStr, pStr + cbStr)) + rw.value();
	}

	/*auto header = cu::TestHeader<double>("f(x) = x", 8, &endptr);
	header.push_argument("x", 1, 12);*/
	//cuda_list<cu::cuda_string> lst;
	/*lst.push_back("1");
	lst.push_front("000daa");
	auto b = lst.back();
	auto f = lst.front();
	auto beg = lst.begin();
	auto end = lst.end();
	lst.erase(beg);*/
	//lst.pop_back();
	//lst.pop_front(); //not needed
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
