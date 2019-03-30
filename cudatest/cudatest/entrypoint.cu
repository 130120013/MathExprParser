#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <limits>

#ifdef __CUDA_ARCH__
typedef int value_type;
#else
typedef double value_type;
#endif

constexpr value_type val = 12;

__host__ __device__ value_type get_val()
{
	return value_type(val);
}

__global__ void get_device_data_kernel(double* pBuf)
{
	*pBuf = double(get_val());
}

double get_device_data()
{
	double h, *pD;
	auto err = cudaMalloc((void**) &pD, sizeof(double));
	if (err)
		return std::numeric_limits<double>::infinity();
	get_device_data_kernel<<<1, 1>>>(pD);
	cudaDeviceSynchronize();
	err = cudaMemcpy(&h, pD, sizeof(double), cudaMemcpyDeviceToHost);
	if (err)
		return std::numeric_limits<double>::infinity();
	cudaDeviceSynchronize();
	err = cudaFree(pD);
	if (err)
		return std::numeric_limits<double>::infinity();
	return h;
}

double get_host_data()
{
	return get_val();
}

#include <iostream>

int main(int, char**)
{
	std::cout << "Host copy " << get_host_data() << "\n";
	std::cout << "Device copy " << get_device_data() << "\n";
	return 0;
}