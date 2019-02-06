#include <utility>
#include <type_traits>
#include <iterator>
//#include <thrust/iterator/std::reverse_iterator.h>
//#include <cuda/config.h>

#include "cuda_config.cuh"

#ifndef CUDA_ITERATORS_H_
#define CUDA_ITERATORS_H_

CU_BEGIN

template<class Iterator>
class reverse_iterator :public std::reverse_iterator<Iterator>
{	// wrap iterator to run it backwards
	typedef reverse_iterator<Iterator> _MyType;
	typedef std::reverse_iterator<Iterator> _MyBase;

public:
	typedef typename _MyBase::value_type value_type;
	typedef typename _MyBase::difference_type difference_type;
	typedef typename _MyBase::reference reference;
	typedef typename _MyBase::pointer pointer;
	typedef typename _MyBase::iterator_category iterator_category;
	typedef Iterator iterator_type;

	using _MyBase::reverse_iterator;

	//template<class RightIterator>
	//__device__ _MyType& operator=(const reverse_iterator<RightIterator>& right)
	//{	// assign from compatible base
	//	current = right.base();
	//	return *this;
	//}

	//__device__ typename std::iterator_traits<Iterator>::pointer operator->() const
	//{	// return pointer to class object
	//	return &(**this);
	//}

	//__device__ _MyType& operator++()
	//{	// preincrement
	//	return static_cast<_MyType&>(++static_cast<_MyBase&>(*this));
	//}

	//__device__ _MyType operator++(int)
	//{	// postincrement
	//	auto old = this->base();
	//	++*this;
	//	return old;
	//}

	//__device__ _MyType& operator--()
	//{	// predecrement
	//	return static_cast<_MyType&>(--static_cast<_MyBase&>(*this));
	//}

	//__device__ _MyType operator--(int)
	//{	// postincrement
	//	auto old = this->base();
	//	--*this;
	//	return old;
	//}

	//// N.B. functions valid for random-access iterators only beyond this point

	//__device__ _MyType& operator+=(difference_type offset)
	//{	// increment by integer
	//	return static_cast<_MyType&>(static_cast<_MyBase&>(*this) += offset);
	//}

	//__device__ _MyType operator+(difference_type offset) const
	//{	// return this + integer
	//	return _MyType((static_cast<const _MyBase&>(*this) + offset).base());
	//}

	//__device__ _MyType& operator-=(difference_type _Off)
	//{	// increment by integer
	//	return static_cast<_MyType&>(static_cast<_MyBase&>(*this) -= offset);
	//}

	//__device__ _MyType operator-(difference_type _Off) const
	//{	// return this - integer
	//	return _MyType((static_cast<const _MyBase&>(*this) - offset).base());
	//}
};

//template<class Iterator>
//__device__ cuda_reverse_iterator<Iterator> operator+(typename cuda_reverse_iterator<Iterator>::difference_type _Off,
//	const reverse_iterator<Iterator>& right)
//{	// return this + integer
//	return right + _Off;
//}
//
//template<class Iterator>
//__device__ typename reverse_iterator<Iterator>::difference_type operator-(const reverse_iterator<Iterator>& left,
//	const reverse_iterator<Iterator>& right)
//{	// return this + integer
//	return static_cast<const std::reverse_iterator<Iterator>&>(left) - static_cast<const std::reverse_iterator<Iterator>&>(right);
//}

//#pragma hd_warning_disable
//#pragma nv_exec_check_disable
template <class It>
__host__ __device__ It next_it(It it)
{
	return ++it;
}

//#pragma hd_warning_disable
//#pragma nv_exec_check_disable
template <class It>
__host__ __device__ It prev_it(It it)
{
	return --it;
}

//#pragma hd_warning_disable
//#pragma nv_exec_check_disable
template <class Container>
__host__ __device__ auto begin_it(Container&& cont)
{
	return std::forward<Container>(cont).begin();
}

//#pragma hd_warning_disable
//#pragma nv_exec_check_disable
template <class Container>
__host__ __device__ auto end_it(Container&& cont)
{
	return std::forward<Container>(cont).end();
}

template <class T, std::size_t N>
__host__ __device__ auto begin_it(T(&array)[N])
{
	return &array[0];
}

template <class T, std::size_t N>
__host__ __device__ auto end_it(T(&array)[N])
{
	return &array[N];
}

CU_END

#endif //CUDA_ITERATORS_H_

