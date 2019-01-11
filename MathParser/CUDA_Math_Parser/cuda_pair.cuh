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
		constexpr cuda_pair() = default;
		cuda_pair(const T1& x, const T2& y) : first(x), second(y) {}
		cuda_pair(T1&& x, T2&& y) : first(std::move(x)), second(std::move(y)) {}
		cuda_pair(const cuda_pair<T1, T2>& p)
		{
			*this = p;
		}
		cuda_pair(cuda_pair<T1, T2>&& p)
		{
			*this = std::move(p);
		}
		cuda_pair& operator= (const cuda_pair<T1, T2>& other)
		{
			this->first = other.first;
			this->second = other.second;
		}
		cuda_pair& operator= (cuda_pair<T1, T2>&& other)
		{
			this->first = std::move(other.first);
			this->second = std::move(other.second);
			return *this;
		}
	};

	template <class T1, class T2>
	cuda_pair<T1, T2> make_cuda_pair(T1 t, T2 u)
	{
		return cuda_pair<T1, T2>(t, u);
	}

	
}
#endif //CUDA_PAIR
