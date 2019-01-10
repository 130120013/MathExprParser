#include <memory>
#include <atomic>
#include <type_traits>
//#include <chsvlib/chsverr.h>
//#include <cuda/config.h>
//#include <cuda/except.cuh>

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

template <class elemType>
class cuda_unique_ptr
{
	template <class U>
	friend class cuda_unique_ptr;
public:
	typedef elemType* pointer;
	typedef elemType element_type;
private:
	struct Deleter
	{
#if __CUDA_ARCH__ >= 350
		__device__ __host__ 
#endif //__CUDA_ARCH__
		void operator()(pointer ptr) const
		{
#if __CUDA_ARCH__ >= 350
			deconstruct(ptr);
#endif
			cudaFree(ptr);
		}
	private:
		template <class T>
#if __CUDA_ARCH__ >= 350
		__device__ __host__
#endif
		static auto deconstruct(T* ptr) -> typename std::enable_if<std::is_destructible<T>::value>::type
		{
			ptr->~T();
		}
		__device__ 
#if __CUDA_ARCH__ >= 350
		__device__ __host__
#endif
			static auto deconstruct(...) -> void {}
	};
public:
	typedef Deleter deleter_type;

	cuda_unique_ptr() = default;
	cuda_unique_ptr(cuda_unique_ptr&& right):ptr(right.ptr)
	{
		right.ptr = nullptr;
	}
	template <class U, class = typename std::enable_if<std::is_convertible<U*, pointer>::value>::type>
	cuda_unique_ptr(cuda_unique_ptr<U>&& right):ptr(right.release()) {}
	cuda_unique_ptr& operator=(cuda_unique_ptr&& right)
	{
		if (this != &right)
		{
			if (ptr != nullptr)
				del(ptr);
			ptr = right.ptr;
			right.ptr = nullptr;
		}
		return *this;
	}
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		explicit cuda_unique_ptr(pointer p) :ptr(p) {};
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		~cuda_unique_ptr()
	{
		if (ptr != nullptr)
			del(ptr);
	};
	
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		pointer release()
	{
		pointer p = ptr;
		ptr = nullptr;
		return p;
	}
	
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		void reset(pointer p = pointer())
	{
		if (ptr == p)
			return;
		if (ptr != nullptr)
			del(ptr);
		ptr = p;
	}
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		pointer get() const
	{
		return ptr;
	}
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		operator bool()
	{
		return(ptr != nullptr);
	}
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		pointer operator->() const
	{
		return ptr;
	}
	template <class U = element_type, class = typename std::enable_if<!std::is_void<U>::value>::type>
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
		U& operator*() const
	{
		return *ptr;
	}
private:
	pointer ptr = pointer();
	Deleter del;
};

namespace _Implementation
{
	template <class T>
	struct Deleter
	{
		void operator()(T* ptr)
		{
			cudaFree(ptr);
		}
	};
};

template <class T>
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
cuda_unique_ptr<T> make_cuda_unique_ptr(std::size_t c = 1);

template <>
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
inline cuda_unique_ptr<void> make_cuda_unique_ptr<void>(std::size_t cb)
{
	void* ptr;
	auto err = cudaMalloc(&ptr, cb);
	if (err != cudaSuccess)
		return cuda_unique_ptr<void>();
//#if defined(__CUDA_ARCH__)
//		cuda_abort_with_error(CHSVERROR_OUTOFMEMORY);
////#else
////		throw cuda_exception(err);
//#endif
	return cuda_unique_ptr<void>(ptr);
}

template <class T>
#if __CUDA_ARCH__ >= 350
	__device__ __host__ 
#endif //__CUDA_ARCH__
cuda_unique_ptr<T> make_cuda_unique_ptr(std::size_t c)
{
	return cuda_unique_ptr<T>(static_cast<T*>(make_cuda_unique_ptr<void>(c * sizeof(T)).release()));
}

template <class T>
struct impl_cuda_device_unique_ptr_base
{
	typedef T *pointer;
protected:
	__device__ impl_cuda_device_unique_ptr_base() = default;
	__device__ explicit impl_cuda_device_unique_ptr_base(T* ptr):m_ptr(ptr) {}
	__device__ impl_cuda_device_unique_ptr_base(const impl_cuda_device_unique_ptr_base&) = delete;
	__device__ impl_cuda_device_unique_ptr_base operator=(const impl_cuda_device_unique_ptr_base&) = delete;
	__device__ impl_cuda_device_unique_ptr_base(impl_cuda_device_unique_ptr_base&& right):m_ptr(right.m_ptr)
	{
		right.m_ptr = nullptr;
	}
	template <class U, class = typename std::enable_if<std::is_convertible<U*, pointer>::value>::type>
	__device__ impl_cuda_device_unique_ptr_base(impl_cuda_device_unique_ptr_base<U>&& right) : m_ptr(right.m_ptr)
	{
		right.m_ptr = nullptr;
	}
	__device__ impl_cuda_device_unique_ptr_base& operator=(impl_cuda_device_unique_ptr_base&& right)
	{
		m_ptr = right.m_ptr;
		right.m_ptr = nullptr;
		return *this;
	}
	__device__ constexpr const pointer& ptr() const
	{
		return m_ptr;
	}
	__device__ pointer& ptr()
	{
		return m_ptr;
	}
private:
	pointer m_ptr = pointer();
};

template <class T>
struct impl_cuda_device_unique_ptr_spec:impl_cuda_device_unique_ptr_base<T>
{
	typedef T element_type;
	typedef typename impl_cuda_device_unique_ptr_base::pointer pointer;
protected:
	struct resource_mgr
	{
		template <class ... Params>
		__device__ static pointer acquire(Params&& ... args)
		{
			return new element_type(std::forward<Params>(params)...);
		}
		__device__ static void release(pointer ptr)
		{
			delete ptr;
		}
	};
public:
	using impl_cuda_device_unique_ptr_base::impl_cuda_device_unique_ptr_base;
	__device__ pointer operator->() const
	{
		return this->ptr();
	}
	__device__ element_type& operator*() const
	{
		return *this->ptr();
	}
};

template <class T>
struct impl_cuda_device_unique_ptr_spec<T[]>:impl_cuda_device_unique_ptr_base<T>
{
	typedef T element_type;
	typedef typename impl_cuda_device_unique_ptr_base::pointer pointer;
protected:
	struct resource_mgr
	{
		__device__ static pointer acquire(std::size_t cHowMany)
		{
			return new element_type[cHowMany];
		}
		__device__ static void release(pointer ptr)
		{
			delete [] ptr;
		}
	};
public:
	using impl_cuda_device_unique_ptr_base::impl_cuda_device_unique_ptr_base;
	__device__ pointer operator->() const
	{
		return this->ptr();
	}
	__device__ element_type& operator*() const
	{
		return *this->ptr();
	}
	__device__ element_type& operator[](std::size_t i) const
	{
		return this->ptr()[i];
	}
};

template<>
struct impl_cuda_device_unique_ptr_spec<void>:impl_cuda_device_unique_ptr_base<void> 
{
	typedef std::uint8_t element_type;
	typedef typename impl_cuda_device_unique_ptr_base::pointer pointer;
protected:
	struct resource_mgr
	{
		__device__ static pointer acquire(std::size_t cHowMany)
		{
			return new element_type[cHowMany];
		}
		__device__ static void release(pointer ptr)
		{
			delete [] static_cast<std::uint8_t*>(ptr);
		}
	};
public:
	using impl_cuda_device_unique_ptr_base::impl_cuda_device_unique_ptr_base;
	__device__ element_type& operator[](std::size_t i) const
	{
		return reinterpret_cast<element_type*>(this->ptr())[i];
	}
};

template <>
struct impl_cuda_device_unique_ptr_spec<const void>:impl_cuda_device_unique_ptr_base<const void> 
{
	typedef const std::uint8_t element_type;
	typedef typename impl_cuda_device_unique_ptr_base::pointer pointer;
protected:
	struct resource_mgr
	{
		__device__ static pointer acquire(std::size_t cHowMany)
		{
			return new element_type[cHowMany];
		}
		__device__ static void release(pointer ptr)
		{
			delete [] static_cast<std::uint8_t*>(const_cast<void*>(ptr));
		}
	};
public:
	using impl_cuda_device_unique_ptr_base::impl_cuda_device_unique_ptr_base;
	__device__ element_type& operator[](std::size_t i) const
	{
		return reinterpret_cast<element_type*>(this->ptr())[i];
	}
};

template <class T>
struct cuda_device_unique_ptr;

template <class T, class ... Args>
__device__ cuda_device_unique_ptr<T> make_cuda_device_unique_ptr(Args&& ... args);

template <class T>
struct cuda_device_unique_ptr:impl_cuda_device_unique_ptr_spec<T>
{
	typedef typename impl_cuda_device_unique_ptr_spec::pointer pointer;
private:
	typedef typename impl_cuda_device_unique_ptr_spec::resource_mgr resource_mgr;
	template <class T, class ... Args>
	friend __device__ cuda_device_unique_ptr<T> make_cuda_device_unique_ptr(Args&& ... args);
public:
	__device__ cuda_device_unique_ptr() = default;
	__device__ cuda_device_unique_ptr(cuda_device_unique_ptr&&) = default;
	template <class U, class = typename std::enable_if<std::is_convertible<U*, pointer>::value>::type>
	__device__ cuda_device_unique_ptr(cuda_device_unique_ptr<U>&& right)
		:impl_cuda_device_unique_ptr_spec<T>(std::move(right)) {}
	__device__ cuda_device_unique_ptr& operator=(cuda_device_unique_ptr&& right)
	{
		if (this != &right)
		{
			if (this->ptr() != nullptr)
				resource_mgr::release(this->ptr());
			static_cast<impl_cuda_device_unique_ptr_spec<T>&>(*this) = std::move(right);
		}
		return *this;
	}
	__device__ explicit cuda_device_unique_ptr(pointer p) :impl_cuda_device_unique_ptr_spec<T>(p) {};
	__device__ ~cuda_device_unique_ptr()
	{
		if (this->ptr() != nullptr)
			resource_mgr::release(this->ptr());
	};
	__device__ pointer release()
	{
		pointer p = this->ptr();
		this->ptr() = nullptr;
		return p;
	}

	__device__ void reset(pointer p = pointer())
	{
		if (this->ptr() == p)
			return;
		if (this->ptr() != nullptr)
			resource_mgr::release(this->ptr());
		this->ptr() = p;
	}
	__device__ pointer get() const
	{
		return this->ptr();
	}
	__device__ explicit operator bool()
	{
		return this->ptr() != nullptr;
	}


private:
	struct deleter
	{
		__device__ void operator()(pointer ptr) const
		{
			resource_mgr::release(ptr);
		}
	};
	deleter m_del;
public:
	typedef deleter deleter_type;

	__device__ deleter_type& get_deleter()
	{
		return del;
	}
	__device__ const deleter_type& get_deleter() const
	{
		return del;
	}
};

template <class T, class ... Args>
__device__ cuda_device_unique_ptr<T> make_cuda_device_unique_ptr(Args&& ... args)
{
	return cuda_device_unique_ptr<T>(cuda_device_unique_ptr<T>::resource_mgr::acquire(std::forward<Args>(args)...));
}

class cuda_stream
{
	cudaStream_t m_str = nullptr;
	struct default_stream_tag {};
	friend cuda_stream default_cuda_stream();
	inline cuda_stream(default_stream_tag) {}
public:
	inline __host__ cuda_stream()
	{
		//cuda_runtime_call(cudaStreamCreate, &m_str);
	}
	inline explicit cuda_stream(cudaStream_t str):m_str(str) {};
	inline cuda_stream(cuda_stream&& right):m_str(right.m_str)
	{
		right.m_str = nullptr;
	}
	inline cuda_stream& operator=(cuda_stream&& right)
	{
		if (this != &right)
		{
			if (m_str)
				cudaStreamDestroy(m_str);
			m_str = right.m_str;
			right.m_str = nullptr;
		}
		return *this;
	}
	inline ~cuda_stream()
	{
		if (m_str != nullptr)
			cudaStreamDestroy(m_str);
	};
	inline cudaStream_t release()
	{
		auto str = m_str;
		m_str = nullptr;
		return str;
	}
	inline void reset(cudaStream_t str = nullptr)
	{
		if (m_str != str)
			return;
		/*if (m_str != nullptr)
			cuda_runtime_call(cudaStreamDestroy, m_str);*/
		m_str = str;
	}
	inline cudaStream_t get() const
	{
		return m_str;
	}
};

inline cuda_stream default_cuda_stream()
{
	return cuda_stream(cuda_stream::default_stream_tag());
}

class shared_cuda_stream
{
	struct shared_cuda_stream_holder
	{
		cudaStream_t stream;
		std::atomic<std::size_t> cRefs;
	};
	shared_cuda_stream_holder* m_pBuf = nullptr;
public:
	inline shared_cuda_stream()
	{
		m_pBuf = new shared_cuda_stream_holder;
	//	cuda_runtime_call(cudaStreamCreate, &m_pBuf->stream);
		m_pBuf->cRefs.store(1, std::memory_order_relaxed);
	}
	inline explicit shared_cuda_stream(cudaStream_t str)
	{
		if (str != 0)
		{
			m_pBuf = new shared_cuda_stream_holder;
			m_pBuf->stream = str;
			m_pBuf->cRefs.store(1, std::memory_order_relaxed);
		}
	}
	inline shared_cuda_stream(cuda_stream&& str):shared_cuda_stream(str.release())
	{
	}
	inline shared_cuda_stream(const shared_cuda_stream& right) noexcept :m_pBuf(right.m_pBuf)
	{
		if (m_pBuf)
			right.m_pBuf->cRefs.fetch_add(1, std::memory_order_acquire);
	}
	inline shared_cuda_stream(shared_cuda_stream&& right) noexcept:m_pBuf(right.m_pBuf)
	{
		right.m_pBuf = nullptr;
	}
	inline ~shared_cuda_stream()
	{
		if (m_pBuf && m_pBuf->cRefs.fetch_sub(1, std::memory_order_acq_rel) == 1)
		{
			cudaStreamDestroy(m_pBuf->stream);
			delete m_pBuf;
		}
	}
	inline shared_cuda_stream& operator=(const shared_cuda_stream& right) noexcept
	{
		if (this != &right)
		{
			m_pBuf = right.m_pBuf;
			if (m_pBuf != nullptr)
				m_pBuf->cRefs.fetch_add(1, std::memory_order_acquire);
		}
		return *this;
	}
	inline shared_cuda_stream& operator=(shared_cuda_stream&& right) noexcept
	{
		if (this != &right)
		{
			m_pBuf = right.m_pBuf;
			right.m_pBuf = nullptr;
		}
		return *this;
	}
	void reset(cudaStream_t str = 0);
	inline cudaStream_t get() const noexcept
	{
		return m_pBuf == nullptr?0:m_pBuf->stream;
	}
	inline std::size_t use_count() const noexcept
	{
		return m_pBuf == nullptr?0:m_pBuf->cRefs.load(std::memory_order_relaxed);
	}
};

template <class T>
class cuda_shared_ptr
{
public:
	__device__ T* get() const;
	T& operator*() const;
	T* operator->() const;

};

template< class T, class... Args >
cuda_shared_ptr<T> make_cuda_shared(Args&&... args);

#endif //CUDA_MEMORY_H
