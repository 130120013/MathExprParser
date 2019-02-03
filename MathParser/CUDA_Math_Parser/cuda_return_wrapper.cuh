#include "cuda_config.cuh"

#ifndef CUDA_RETURN_WRAPPER_H
#define CUDA_RETURN_WRAPPER_H

CU_BEGIN

enum class CudaParserErrorCodes
{
	Success,
	NotReady,
	NotEnoughMemory,
	UnexpectedCall,
	InsufficientNumberParams,
	UnexpectedToken,
	ParameterIsNotUnique,
	InvalidArgument,
	ParameterIsNotFound,
	InvalidExpression
};
template <class LeftReturnWrapper, class RightReturnWrapper>
__device__ auto impl_assign_return_wrapper(LeftReturnWrapper& left, RightReturnWrapper&& right)
-> std::enable_if_t <
	std::is_convertible<decltype(std::declval<RightReturnWrapper&&>().value()), typename LeftReturnWrapper::value_type>::value,
	LeftReturnWrapper&
>
{
	typedef typename LeftReturnWrapper::value_type value_type;
	left.code_ref() = right.return_code();
	if (right.get() == nullptr)
	{
		if (left.value_ptr() != nullptr)
		{
			left.value_ptr()->~value_type();
			left.value_ptr() = nullptr;
		}
	} else
	{
		if (left.value_ptr() == nullptr)
			left.value_ptr() = left.get_buf_ptr();
		new (left.value_ptr()) value_type(std::forward<RightReturnWrapper>(right).value());
	}
	return left;
}

template <class LeftReturnWrapper, class RightReturnWrapper>
__device__ auto impl_construct_return_wrapper(LeftReturnWrapper& left, RightReturnWrapper&& right)
-> std::enable_if_t <
	std::is_convertible<decltype(std::declval<RightReturnWrapper&&>().value()), typename LeftReturnWrapper::value_type>::value,
	LeftReturnWrapper&
>
{
	typedef typename LeftReturnWrapper::value_type value_type;
	if (right.get() == nullptr)
		left.value_ptr() = nullptr;
	else
	{
		left.value_ptr() = left.get_buf_ptr();
		new (left.value_ptr()) value_type(right.value());
	}
	return left;
}

template <class Derived, class T, class = void>
struct impl_move_assignable_return_wrapper
{
	impl_move_assignable_return_wrapper() = default;
	impl_move_assignable_return_wrapper(const impl_move_assignable_return_wrapper&) = default;
	impl_move_assignable_return_wrapper(impl_move_assignable_return_wrapper&&) = default;
	impl_move_assignable_return_wrapper& operator=(const impl_move_assignable_return_wrapper&) = default;
	impl_move_assignable_return_wrapper& operator=(impl_move_assignable_return_wrapper&&) = delete;
};

template <class Derived, class T>
struct impl_move_assignable_return_wrapper<Derived, T, std::enable_if_t<std::is_move_assignable<T>::value>>
{
	impl_move_assignable_return_wrapper() = default;
	impl_move_assignable_return_wrapper(const impl_move_assignable_return_wrapper&) = default;
	impl_move_assignable_return_wrapper(impl_move_assignable_return_wrapper&&) = default;
	impl_move_assignable_return_wrapper& operator=(const impl_move_assignable_return_wrapper&) = default;
	__device__ impl_move_assignable_return_wrapper& operator=(impl_move_assignable_return_wrapper&& right)
	{
		impl_assign_return_wrapper(get_this(), std::move(right.get_this()));
		return *this;
	}
private:
	__device__ Derived& get_this() { return static_cast<Derived&>(*this); }
	__device__ const Derived& get_this() const { return static_cast<const Derived&>(*this); }
};

template <class Derived, class T, class = void>
struct impl_copy_assignable_return_wrapper
{
	impl_copy_assignable_return_wrapper() = default;
	impl_copy_assignable_return_wrapper(const impl_copy_assignable_return_wrapper&) = default;
	impl_copy_assignable_return_wrapper(impl_copy_assignable_return_wrapper&&) = default;
	impl_copy_assignable_return_wrapper& operator=(const impl_copy_assignable_return_wrapper&) = delete;
	__device__ impl_copy_assignable_return_wrapper& operator=(impl_copy_assignable_return_wrapper&&) = default;
};
template <class Derived, class T>
struct impl_copy_assignable_return_wrapper<Derived, T, std::enable_if_t<std::is_copy_assignable<T>::value>>
{
	impl_copy_assignable_return_wrapper() = default;
	impl_copy_assignable_return_wrapper(const impl_copy_assignable_return_wrapper&) = default;
	impl_copy_assignable_return_wrapper(impl_copy_assignable_return_wrapper&&) = default;
	__device__ impl_copy_assignable_return_wrapper& operator=(const impl_copy_assignable_return_wrapper& right)
	{
		impl_assign_return_wrapper(get_this(), right.get_this());
		return *this;
	}
	__device__ impl_copy_assignable_return_wrapper& operator=(impl_copy_assignable_return_wrapper&&) = default;
private:
	__device__ Derived& get_this() { return static_cast<Derived&>(*this); }
	__device__ const Derived& get_this() const { return static_cast<const Derived&>(*this); }
};

template <class Derived, class T>
struct impl_storage_wrapper :impl_copy_assignable_return_wrapper<Derived, T>, impl_move_assignable_return_wrapper<Derived, T>
{
	typedef T value_type;
	typedef T* pointer;
	__device__ impl_storage_wrapper() = default;
	template <class U, class = std::enable_if_t<std::is_constructible<T, U&&>::value>>
	__device__ impl_storage_wrapper(U&& value, CudaParserErrorCodes exit_code = CudaParserErrorCodes::Success) :m_code(exit_code)
	{
		static_assert(sizeof(T) <= sizeof(val_buf) /*&& alignof(T) <= alignof(decltype(val_buf))*/,
			"Sizes and alignments of T and U!");
		m_pVal = std::move(this->get_buf_ptr());
		new (m_pVal) T(std::forward<U>(value));
	}
	__device__ impl_storage_wrapper(CudaParserErrorCodes exit_code) :m_code(exit_code), m_pVal(nullptr) {}
	__device__ ~impl_storage_wrapper()
	{
		if (m_pVal)
			m_pVal->~T();
	}
	__device__ T* get()
	{
		return m_pVal;
	}
	__device__ const T* get() const
	{
		return m_pVal;
	}
	__device__ T* operator->()
	{
		return this->get();
	}
	__device__ const T* operator->() const
	{
		return this->get();
	}
	__device__ const T& value() const &
	{
		return *this->get();
	}
	__device__ T& value() &
	{
		return *this->get();
	}
	__device__ const T&& value() const &&
	{
		return std::move(this->value());
	}
	__device__ T&& value() &&
	{
		return std::move(this->value());
	}
	__device__ CudaParserErrorCodes return_code() const
	{
		return m_code;
	}
	__device__ explicit operator bool() const
	{
		return m_code == CudaParserErrorCodes::Success;
	}
	__device__ bool operator!() const
	{
		return m_code != CudaParserErrorCodes::Success;
	}
protected:
	__device__ Derived& get_this() { return static_cast<Derived&>(*this); }
	__device__ const Derived& get_this() const { return static_cast<const Derived&>(*this); }
private:
	CudaParserErrorCodes m_code = CudaParserErrorCodes::Success;
	pointer m_pVal = nullptr;

	alignas(T) char val_buf[sizeof(T)];
protected:
	__device__ T* get_buf_ptr()
	{
		return reinterpret_cast<T*>(val_buf);
	}
	__device__ const T* get_buf_ptr() const
	{
		return reinterpret_cast<const T*>(val_buf);
	}
	__device__ const CudaParserErrorCodes& code_ref() const
	{
		return m_code;
	}
	__device__ CudaParserErrorCodes& code_ref()
	{
		return m_code;
	}
	__device__ const pointer& value_ptr() const
	{
		return m_pVal;
	}
	__device__ pointer& value_ptr()
	{
		return m_pVal;
	}
};
template <class Derived, class T>
struct impl_move_constructible_return_wrapper :impl_storage_wrapper<Derived, T>
{
	using impl_storage_wrapper<Derived, T>::impl_storage_wrapper;
	__device__ impl_move_constructible_return_wrapper() = default;
	__device__ impl_move_constructible_return_wrapper(impl_move_constructible_return_wrapper&& right)
	{
		impl_construct_return_wrapper(this->get_this(), std::move(right.get_this()));
	}
	__device__ impl_move_constructible_return_wrapper& operator=(const impl_move_constructible_return_wrapper&) = default;
	__device__ impl_move_constructible_return_wrapper& operator=(impl_move_constructible_return_wrapper&&) = default;
};

template <class Derived, class T>
struct impl_copy_constructible_return_wrapper :impl_move_constructible_return_wrapper<Derived, T>
{
	using impl_move_constructible_return_wrapper<Derived, T>::impl_move_constructible_return_wrapper;
	__device__ impl_copy_constructible_return_wrapper() = default;
	__device__ impl_copy_constructible_return_wrapper(const impl_copy_constructible_return_wrapper& right)
	{
		impl_construct_return_wrapper(this->get_this(), right.get_this());
	}
	__device__ impl_copy_constructible_return_wrapper(impl_copy_constructible_return_wrapper&&) = default;
	__device__ impl_copy_constructible_return_wrapper& operator=(const impl_copy_constructible_return_wrapper&) = default;
	__device__ impl_copy_constructible_return_wrapper& operator=(impl_copy_constructible_return_wrapper&&) = default;
};

template <class T>
struct return_wrapper_t;

template <class T>
using impl_return_wrapper_proxy = std::conditional_t<
	std::is_copy_constructible<T>::value,
	impl_copy_constructible_return_wrapper<return_wrapper_t<T>, T>,
	std::conditional_t<
	std::is_move_constructible<T>::value,
	impl_move_constructible_return_wrapper<return_wrapper_t<T>, T>,
	impl_storage_wrapper<return_wrapper_t<T>, T>
	>
>;

template <class T>
struct return_wrapper_t :impl_return_wrapper_proxy<T>
{
	__device__ return_wrapper_t() = default;
	template <class U, class = std::enable_if_t<std::is_constructible<T, U&&>::value>>
	__device__ return_wrapper_t(U&& value, CudaParserErrorCodes exit_code = CudaParserErrorCodes::Success)
		:impl_return_wrapper_proxy<T>(std::forward<U>(value), exit_code) {}
	__device__ return_wrapper_t(CudaParserErrorCodes exit_code)
		:impl_return_wrapper_proxy<T>(exit_code) {}
	friend impl_copy_assignable_return_wrapper<return_wrapper_t<T>, T>;
	friend impl_move_assignable_return_wrapper<return_wrapper_t<T>, T>;
	template <class LeftReturnWrapper, class RightReturnWrapper>
	friend __device__ auto impl_assign_return_wrapper(LeftReturnWrapper& left, RightReturnWrapper&& right)
		->std::enable_if_t <
		std::is_convertible<decltype(std::declval<RightReturnWrapper&&>().value()), typename LeftReturnWrapper::value_type>::value,
		LeftReturnWrapper&
		>;

	template <class LeftReturnWrapper, class RightReturnWrapper>
	friend __device__ auto impl_construct_return_wrapper(LeftReturnWrapper& left, RightReturnWrapper&& right)
		->std::enable_if_t <
		std::is_convertible<decltype(std::declval<RightReturnWrapper&&>().value()), typename LeftReturnWrapper::value_type>::value,
		LeftReturnWrapper&
		>;
};

template <class T>
struct return_wrapper_t<T&>
{
	__device__ return_wrapper_t(T& value, CudaParserErrorCodes exit_code = CudaParserErrorCodes::Success) :m_code(exit_code)
	{
		m_pVal = &value;
	}
	__device__ explicit return_wrapper_t(CudaParserErrorCodes exit_code) :m_pVal(nullptr), m_code(exit_code) {}
	__device__ T* get()
	{
		return m_pVal;
	}
	__device__ const T* get() const
	{
		return m_pVal;
	}
	__device__ T* operator->()
	{
		return this->get();
	}
	__device__ const T* operator->() const
	{
		return this->get();
	}
	__device__ CudaParserErrorCodes return_code() const
	{
		return m_code;
	}
	__device__ T& value()
	{
		return *this->get();
	}
	__device__ explicit operator bool() const
	{
		return m_code == CudaParserErrorCodes::Success;
	}
	__device__ bool operator!() const
	{
		return m_code != CudaParserErrorCodes::Success;
	}
private:
	CudaParserErrorCodes m_code = CudaParserErrorCodes::Success;
	T* m_pVal = nullptr;
};

template <>
struct return_wrapper_t<void>
{
	__device__ return_wrapper_t() = default;
	//__device__ return_wrapper_t():m_code(CudaParserErrorCodes::Success) {}
	__device__ explicit return_wrapper_t(CudaParserErrorCodes exit_code) :m_code(exit_code) {}
	///*__device__*/ return_wrapper_t(const return_wrapper_t& ) = default;
	///*__device__*/ return_wrapper_t(return_wrapper_t&& ) = default;
	///*__device__*/  return_wrapper_t& operator= (const return_wrapper_t&) = default;
	///*__device__*/  return_wrapper_t& operator= (return_wrapper_t&& ) = default;

	__device__ void* get()
	{
		return nullptr;
	}
	__device__ const void* get() const
	{
		return nullptr;
	}
	__device__ void* operator->()
	{
		return this->get();
	}
	__device__ const void* operator->() const
	{
		return this->get();
	}
	__device__ CudaParserErrorCodes return_code() const
	{
		return m_code;
	}
	__device__ explicit operator bool() const
	{
		return m_code == CudaParserErrorCodes::Success;
	}
	__device__ bool operator!() const
	{
		return m_code != CudaParserErrorCodes::Success;
	}
private:
	CudaParserErrorCodes m_code = CudaParserErrorCodes::Success;
};

template <class T = void>
auto make_return_wrapper_error(CudaParserErrorCodes error) { return return_wrapper_t<T>(error); }

CU_END
#endif // !CUDA_RETURN_WRAPPER_H
