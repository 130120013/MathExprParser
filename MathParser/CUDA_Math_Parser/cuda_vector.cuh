#include "cuda_memory.cuh"
template <class T>
class cuda_vector
{
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
	__device__ void push_back(T&& value)
	{
		this->reserve(m_size + 1);
		this->m_buf.get()[m_size++] = std::move(value);
	}

	__device__ bool empty() const
	{
		return this->size() == 0;
	}

	__device__ void reserve(size_type new_size)
	{
		if (new_size > this->capacity())
		{
			auto buf = make_cuda_device_unique_ptr<value_type[]>(new_size);
			if (bool(buf))
			{
				if (bool(m_buf))
					memcpy(buf.get(), m_buf.get(), this->size() * sizeof(value_type));
				this->m_capacity = new_size;
				m_buf = std::move(buf);
			}
		}
	}
	__device__ void shrink_to_fit()
	{
		//TODO: implement
	}
	__device__ cuda_vector() = default;
	__device__ cuda_vector(const cuda_vector& right)
	{
		*this = right;
	}
	__device__ cuda_vector(cuda_vector&&) = default;
	__device__ cuda_vector& operator=(const cuda_vector& right)
	{
		if (this != &right)
		{
			auto buf = cuda_device_unique_ptr<value_type[]>(right.size());
			if (bool(buf))
			{
				memcpy(buf.get(), right.data(), right.size() * sizeof(value_type));
				m_buf = std::move(buf);
				m_size = m_capacity = right.size();
			}
		}
		return *this;
	}
	__device__ cuda_vector& operator=(cuda_vector&&) = default;

private:
	cuda_device_unique_ptr<value_type[]> m_buf;
	size_type m_size = 0;
	size_type m_capacity = 0;
};