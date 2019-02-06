#include "cuda_config.cuh"
#include "cuda_memory.cuh"
#include "cuda_return_wrapper.cuh"

#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

CU_BEGIN

template <class T>
class vector
{
	static constexpr const std::size_t cReserveFrame = 128 / sizeof(T);
public:
	typedef T value_type, *pointer, &reference, *iterator;
	typedef const T *const_pointer, &const_reference, *const_iterator;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	__device__ pointer data()
	{
		return m_buf.get();
	}

	__device__ const_pointer data() const
	{
		return m_buf.get();
	}

	__device__ size_type size() const
	{
		return m_size;
	}
	__device__ size_type capacity() const
	{
		return m_capacity;
	}

	__device__ reference operator[] (size_type i)
	{
		return m_buf.get()[i];
	}

	__device__ const_reference operator[] (size_type i) const
	{
		return m_buf.get()[i];
	}

	__device__ iterator begin()
	{
		return &m_buf.get()[0];
	}

	__device__ const_iterator begin() const
	{
		return &m_buf.get()[0];
	}

	__device__ const_iterator cbegin() const
	{
		return &m_buf.get()[0];
	}

	__device__ iterator end()
	{
		return &m_buf.get()[m_size];
	}

	__device__ const_iterator end() const
	{
		return &m_buf.get()[m_size];
	}

	__device__ const_iterator cend() const
	{
		return &m_buf.get()[m_size];
	}

	//__device__ void push_back(const T& value)
	//{
	//	this->reserve(m_size + 1);
	//	this->m_buf.get()[m_size++] = value;
	//}
	__device__ inline cu::return_wrapper_t<void> push_back(T&& value)
	{
		return this->emplace_back(std::move(value));
	}
	template <class U>
	__device__ cu::return_wrapper_t<void> emplace_back(U&& value)
	{
		if (m_size + 1 > m_capacity)
		{
			auto rv = this->reserve(m_capacity + cReserveFrame);
			if (!rv)
				return rv;
		}
		new (&m_buf[m_size++]) T(std::forward<U>(value));
		return cu::return_wrapper_t<void>();
	}

	__device__ inline bool empty() const
	{
		return this->size() == 0;
	}

	__device__ cu::return_wrapper_t<void> reserve(size_type new_cap)
	{
		if (new_cap > this->capacity())
		{
			auto buf = make_cuda_device_unique_ptr_malloc<value_type>(new_cap);
			if (!buf)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			if (bool(buf))
			{
				for (std::size_t iElement = 0; iElement < this->size(); ++iElement)
					new (&buf[iElement]) T(std::move(m_buf[iElement]));
				this->m_capacity = new_cap;
				m_buf = std::move(buf);
			}
		}
		return cu::return_wrapper_t<void>();
	}
	__device__ cu::return_wrapper_t<void> shrink_to_fit()
	{
		if (m_capacity > m_size)
		{
			auto buf = make_cuda_device_unique_ptr_malloc<value_type>(m_size);
			if (!buf)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			if (bool(buf))
			{
				for (std::size_t iElement = 0; iElement < this->size(); ++iElement)
					new (&buf[iElement]) T(std::move(m_buf[iElement]));
				this->m_capacity = m_size;
				m_buf = std::move(buf);
			}
		}
		return cu::return_wrapper_t<void>();
	}
	__device__ vector() = default;
	__device__ vector(const vector& right)
	{
		*this = right;
	}
	__device__ vector(vector&&) = default;
	__device__ vector& operator=(const vector& right)
	{
		if (this != &right)
		{
			auto buf = make_cuda_device_unique_ptr_malloc<value_type>(right.size());
			if (bool(buf))
			{
				for (size_type i = 0; i < right.size(); ++i)
					new T(&buf[i]) &right[i];
				m_buf = std::move(buf);
				m_size = m_capacity = right.size();
			}
		}
		return *this;
	}
	__device__ vector& operator=(vector&&) = default;

private:
	cuda_device_unique_ptr_malloc<value_type> m_buf;
	size_type m_size = 0;
	size_type m_capacity = 0;
};

CU_END

#endif // !CUDA_VECTOR
