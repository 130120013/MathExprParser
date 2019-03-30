#include "cuda_config.cuh"
#include <type_traits>

#ifndef CUDA_PAIR
#define CUDA_PAIR

CU_BEGIN
template <class T1, class T2>
struct pair
{
	T1 first;
	T2 second;
	typedef typename T1 first_type;
	typedef typename T2 second_type;
	__device__ constexpr pair() = default;
	__device__ pair(const T1& x, const T2& y) : first(x), second(y) {}
	__device__ pair(T1&& x, T2&& y) : first(std::move(x)), second(std::move(y)) {}
	template <class U1, class U2>
	__device__ pair(pair<U1, U2>&& pr) : first(std::move(pr.first)), second(std::move(pr.second)) {}
	template <class U1, class U2>
	__device__ pair(const pair<U1, U2>& pr) : first(pr.first), second(pr.second) {}
};

template <class T1, class T2>
__device__ auto make_pair(T1&& t, T2&& u)
{
	return pair<std::decay_t<T1>, std::decay_t<T2>>{std::forward<T1>(t), std::forward<T2>(u)};
}
CU_END

#endif //CUDA_PAIR
