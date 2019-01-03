#include <string>
#include <utility>
#include <type_traits>
//#include <cuda/except.cuh>
#include "cuda_iterator.cuh"
#include "cuda_memory.cuh"

#ifndef CUDA_STRING_H_
#define CUDA_STRING_H_

__device__ std::size_t strlen(const char* psz);
__device__ std::size_t wcslen(const wchar_t* psz);
__device__ char* strcat(char *str1, const char *str2);
__device__ char* strcpy(char* dest, const char* src);
__device__ int strcmp(const char* str1, const char* str2);
__device__ int memcmp(const void* str1, const void* str2, std::size_t size);

__host__ __device__ constexpr bool isdigit(const char ch) noexcept
{
	return (ch >= '0' && ch <= '9');  
}

__host__ __device__ constexpr bool isalpha(const char ch) noexcept
{
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
}

__host__ __device__ constexpr bool isalnum(const char ch) noexcept
{
	return (isalpha(ch) || isdigit(ch));
}

__host__ __device__ constexpr bool isspace(const char ch) noexcept
{
	return (ch == ' ');
}

__device__ unsigned long long strtoull(const char *str, char **str_end, int base);

__device__ long long strtoll(const char *str, char **str_end, int base);

__device__ double strtod(const char* str, char** str_end);


//#ifdef CUDA_UTILS_CPP

//__device__ __constant__ char chNull = 0;

//#else

//__device__ __constant__ extern char chNull;

//#endif

class cuda_string
{
	cuda_device_unique_ptr<char> pStr;
	std::size_t strSize;
public:
	typedef char* iterator;
	typedef const char* const_iterator;
	typedef cuda_reverse_iterator<iterator> reverse_iterator;
	typedef cuda_reverse_iterator<const_iterator> const_reverse_iterator;
	__device__ inline cuda_string() : strSize(0) {}
	__device__ inline cuda_string(const cuda_string& str)
	{
		*this = str;
	}
	__device__ inline cuda_string(cuda_string&& str)
	{
		*this = std::move(str);
	}
	__device__ inline cuda_string(const char* str) : strSize(strlen(str)) 
	{
		pStr.reset(static_cast<char*>(malloc(strSize+1)));
		memcpy(pStr.get(), str, strSize + 1);
	}
	__device__ cuda_string(std::size_t size, char ch);
	template <class Iterator>
	__device__ cuda_string(Iterator b, Iterator e)
	{
		auto size = e - b;
		std::size_t i = 0;
		pStr.reset(static_cast<char*>(malloc(size + 1))); 
		for(Iterator it = b; it != e; ++it)
			pStr.get()[i++] = *it;

		pStr.get()[size] = 0;
		strSize = size;
	}
	__device__ inline std::size_t size() const  { return strSize; }
	__device__ inline const char* c_str() const 
	{
		if(strSize)
			return pStr.get();
		return 0;
	}
	__device__ inline const char* data() const
	{
		return c_str();
	}
	__device__ cuda_string& operator=(const cuda_string& str);
	__device__ cuda_string& operator=(cuda_string&& str) = default;

	__device__ cuda_string& operator+=(const cuda_string& str);
	__device__ inline cuda_string operator+(const cuda_string& str) const
	{
		cuda_string temp(*this);
		temp+=str;
		return temp;
	}
	__device__ inline iterator begin()
	{
		return pStr.get();
	}
	__device__ inline const_iterator begin() const
	{
		return pStr.get();
	}
	__device__ inline const_iterator cbegin() const
	{
		return pStr.get();
	}
	__device__ inline iterator end()
	{
		return pStr.get() + strSize;
	}
	__device__ inline const_iterator end() const
	{
		return pStr.get() + strSize;
	}
	__device__ inline const_iterator cend() const
	{
		return pStr.get() + strSize;
	}
	__device__ inline reverse_iterator rbegin()
	{
		return reverse_iterator(this->end());
	}
	__device__ inline const_reverse_iterator rbegin() const
	{
		return const_reverse_iterator(this->end());
	}
	__device__ inline const_reverse_iterator crbegin() const
	{
		return const_reverse_iterator(this->end());
	}
	__device__ inline iterator rend()
	{
		return this->begin();
	}
	__device__ inline const_iterator rend() const
	{
		return this->begin();
	}
	__device__ inline const_iterator crend() const
	{
		return this->cbegin();
	}
	__device__ inline bool empty() const
	{
		return(strSize == 0);
	}
	template<class Iterator>
	__device__ Iterator erase(Iterator begin, Iterator end)
	{
		Iterator it;
		cuda_string temp(this->end() - this->begin() - (end - begin), 0);
		int k = 0;
		for (auto i = this->begin(); i != this->end(); ++i)
		{
			if (i > begin && i < end)
				continue;
			temp.pStr.get()[k]= *i;
			++k;
		}
		if (++end == pStr.get() + strSize)
			it = pStr.get() + strSize;

		it = ++end;
		*this = temp;
		return it;
	}
};
__device__ double stod(const cuda_string& str, std::size_t* pos = 0);


__device__ inline bool operator==(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) == 0);
}

__device__ inline bool operator!=(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) != 0);
}
	
__device__ inline bool operator>=(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) >= 0);
}
__device__ inline bool operator<=(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) <= 0);
}
	
__device__ inline bool operator>(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) > 0);
}	
	
__device__ inline bool operator<(const cuda_string& str1, const cuda_string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) < 0);
}

template <class T>
__device__ cuda_string to_cuda_string(T val)
{
	static_assert(std::is_integral<T>::value, "T must be integral");
	static_assert(std::is_unsigned<T>::value, "T must be unsigned");
	constexpr std::size_t stackSize = (8 * sizeof(T) + ((8 * sizeof(T)) % 3) - 1) / 3;

	char buf[stackSize]; 

	std::size_t bufSize;

	auto v = val;
	for(bufSize = 0; v != 0; ++bufSize)
	{
		v /= 10;
		buf[bufSize] = v % 10;
	}

	for(auto i = 0; i < bufSize/2; ++i)
	{
		auto temp = buf[i]; 
		buf[i] = buf[bufSize - i];
		buf[bufSize - i] = temp;
	}

	return cuda_string(&buf[0], &buf[bufSize]);
}

#endif //CUDA_STRING_H_
