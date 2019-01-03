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

	__device__ void push_back(const T& value);
	__device__ void push_back(T&& value);

	__device__ bool empty() const
	{
		return this->size() == 0;
	}

	__device__ void reserve(size_type size);

	template <class ... Args>
	__device__ iterator emplace(const_iterator pos, Args&& ... args)
	{
		auto m_new_buf = make_cuda_device_unique_ptr<value_type>(m_size + 1);
		auto idx = pos - this->begin();
		if (idx != 0)
			memcpy(m_new_buf.get(), m_buf.get(), sizeof(value_type) * idx);
		new (&m_new_buf.get()[idx]) T(std::forward<Args>(args)...);

		if (idx != m_size)
			memcpy(m_new_buf.get(), m_buf.get(), sizeof(value_type) * (m_size - idx));
		m_buf.reset(m_new_buf.release());
		m_size = m_size + 1;
		return &m_new_buf.get()[idx];

	}

	template <class ... Args>
	__device__ reference emplace_back(Args&& ... args)
	{
		return *emplace(this->end(), std::forward<Args>(args)...);
	}
private:
	cuda_device_unique_ptr<value_type> m_buf;
	size_type m_size = 0;
};