#ifndef CUDA_PAIR
#define CUDA_PAIR

namespace cu
{
	template <class T1, class T2>
	struct cuda_pair
	{
		T1 first;
		T2 second;
		typedef typename T1 first_type;
		typedef typename T2 second_type;
		__device__ constexpr cuda_pair() = default;
		__device__ cuda_pair(const T1& x, const T2& y) : first(x), second(y) {}
		__device__ cuda_pair(T1&& x, T2&& y) : first(std::move(x)), second(std::move(y)) {}
		template <class U1, class U2>
		__device__ cuda_pair(cuda_pair<U1, U2>&& pr) : first(std::move(pr.first)), second(std::move(pr.second)) {}
		template <class U1, class U2>
		__device__ cuda_pair(const cuda_pair<U1, U2>& pr) : first(pr.first), second(pr.second) {}
	};

	template <class T1, class T2>
	__device__ cuda_pair<T1, T2> make_cuda_pair(T1 t, T2 u)
	{
		return cuda_pair<T1, T2>{t, u};
	}

	
}
#endif //CUDA_PAIR
