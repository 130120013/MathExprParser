#include <string>
#include <utility>
#include <type_traits>
//#include <cuda/except.cuh>
#include "cuda_iterator.cuh"
#include "cuda_memory.cuh"
#include "cuda_config.cuh"
#include "cuda_return_wrapper.cuh"

#ifndef CUDA_STRING_H_
#define CUDA_STRING_H_

CU_BEGIN

namespace _Implementation
{
	template <class T>
	constexpr T numeric_limits_max = std::numeric_limits<T>::max();
	template <class T>
	constexpr T numeric_limits_min = std::numeric_limits<T>::min();
}

template <class T>
struct cuda_numeric_limits
{
	__device__ static constexpr T max()
	{
		return _Implementation::numeric_limits_max<T>;
	}
	__device__ static constexpr T min()
	{
		return _Implementation::numeric_limits_min<T>;
	}
};

__device__ std::size_t strlen(const char* psz)
{
	std::size_t i;
	for (i = 0; psz[i]; ++i);
	return i;
}

__device__ std::size_t wcslen(const wchar_t* psz)
{
	std::size_t i;
	for (i = 0; psz[i]; ++i);
	return i;
}

__device__ char* strcpy(char* dest, const char* src)
{
	for (char *p = dest; (*p++ = *src++););
	return dest;
}

__device__ char* strncpy(char* dest, const char* src, std::size_t cb)
{
	auto i = cb;
	for (char *p = dest; (*p++ = *src++) != '\0' && i > 0; --i);
	return dest;
}

__device__ char* strcat(char *str1, const char *str2)
{
	char* begin = str1;
	while (*str1)
		str1++;

	while (*str1++ = *str2++)
		;

	*str1 = '\0';
	return begin;
}

__device__ int strcmp(const char* str1, const char* str2)
{
	std::size_t i = 0;
	for (; true; ++i)
	{
		if (str1[i] != str2[i])
			break;
		if (str1[i] == 0)
			return 0;
	}
	return ((unsigned char) str1[i] > (unsigned char) str2[i]) ? 1 : -1;
	//return ((unsigned char)str1[i] - (unsigned char)str2[i]);
}

__device__ int memcmp(const void* str1, const void* str2, std::size_t size)
{
	unsigned char* s1 = (unsigned char*)str1;
	unsigned char* s2 = (unsigned char*)str2;
	std::size_t i = 0;
	for (; i < size; ++i)
	{
		if (s1[i] != s2[i])
			return (s1[i] > s2[i]) ? 1 : -1;
	}
	return 0;
}

__device__ int strncmpnz(const char* pStr1, std::size_t cbStr1, const char* pStr2, std::size_t cbStr2)
{
	//if (!pStr1 && !pStr2)
	//{
		auto cbMin = cbStr1 < cbStr2 ? cbStr1 : cbStr2;
		auto cmp = memcmp(pStr1, pStr2, cbMin);
		if (cmp != 0)
			return cmp;
		return cbStr1 == cbStr2 ? 0 : (cbStr1 < cbStr2 ? -1 : 1);
	//}
	//else
		//return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidArgument);
}

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
	return (ch == ' ' || ch == '\t');
}

__device__ unsigned long long strtoull_n(const char *str, std::size_t cbMax, char **str_end, int base)
{
	unsigned long long result = 0;
	std::size_t i;
	if (base != 10)
		return cuda_numeric_limits<unsigned long long>::max(); //cuda_abort_with_error(CHSVERROR_INVALID_PARAMETER);
	if (cbMax == 0)
		*str_end = (char*)str;
	else
	{
		for (i = (str[0] == '-' || str[0] == '+') ? 1 : 0; i < cbMax && isdigit(str[i]); ++i)
		{
			auto result_new = result * 10 + (str[i++] - 48);
			if (result > result_new)
			{
				result = (str[0] == '-') ? (unsigned long long)(cuda_numeric_limits<long long>::min()) : cuda_numeric_limits<unsigned long long>::max();
				break;
			}
			result = result_new;
		}
		*str_end = (char*)&str[i];
		result = (str[0] == '-') ? (cuda_numeric_limits<unsigned long long>::max() - result) + 1 : result;
	}
	return result;
}

__device__ unsigned long long strtoull(const char *str, char **str_end, int base)
{
	return strtoull_n(str, (std::size_t) - 1, str_end, base);
}

__device__ long long strtoll_n(const char *str, std::size_t cbMax, char **str_end, int base)
{
	long long result = 0;
	std::size_t i;
	if (base != 10)
		return cuda_numeric_limits<long long>::max(); //cuda_abort_with_error(CHSVERROR_INVALID_PARAMETER);

	if (cbMax == 0)
		*str_end = (char*)str;
	else
	{
		for (i = (str[0] == '-' || str[0] == '+') ? 1 : 0; i < cbMax && isdigit(str[i]); ++i)
		{
			auto result_new = result * 10 + (str[i] - 48);
			if (result > result_new)
			{
				result = (str[0] == '-') ? cuda_numeric_limits<long long>::min() : cuda_numeric_limits<long long>::max();
				break;
			}
			result = result_new;
		}
		*str_end = (char*)&str[i];// (str + i)
		result = (str[0] == '-') ? -result : result;
	}
	return result;
}

__device__ long long strtoll(const char *str, char **str_end, int base)
{
	return strtoll_n(str, (std::size_t) - 1, str_end, base);
}

__device__ double strtod_n(const char* str, std::size_t cbMax, char** str_end)
{
#ifndef __CUDA_ARCH__
	using std::pow;
#endif
	double result = 0;

	if (cbMax == 0)
		*str_end = (char*)str;
	else
	{
		const char* p = &str[0];
		const char *p1 = nullptr, *p2 = nullptr, *p3 = nullptr;

		std::size_t cbRest = cbMax;
		auto intPart = strtoll_n(p, cbRest, (char**)&p1, 10);
		result = (double) intPart;
		cbRest -= p1 - p;

		if (cbRest > 0 && (*p1 == '.' /*|| *p1 == ','*/))
		{
			std::size_t cbFractialPart;
			auto realPart = strtoll_n(p1 + 1, cbRest, (char**)&p2, 10);
			cbFractialPart = (std::size_t) (p2 - p1 - 1);
			result += double(realPart) * pow((double)10, -(double)cbFractialPart);
			cbRest -= cbFractialPart;
			p1 = p2;
		}
		if (*p1 == 'E')
		{
			auto expPart = strtoll_n(p2 + 1, cbRest, const_cast<char**>(&p3), 10);
			result *= pow((double)10, (double)expPart);
			p1 = p3;
		}
		*str_end = const_cast<char*>(p1);
	}
	return result;
}

__device__ double strtod(const char* str, char** str_end)
{
	return strtod_n(str, (std::size_t) - 1, str_end);
}


//#ifdef CUDA_UTILS_CPP

//__device__ __constant__ char chNull = 0;

//#else

//__device__ __constant__ extern char chNull;

//#endif

class string
{
	cuda_device_unique_ptr<char[]> pStr;
	std::size_t strSize = 0;
public:
	typedef char* iterator;
	typedef const char* const_iterator;
	typedef cu::reverse_iterator<iterator> reverse_iterator;
	typedef cu::reverse_iterator<const_iterator> const_reverse_iterator;
	__device__ string() = default;
	__device__ inline string(const string& str)
	{
		*this = str;
	}
	__device__ inline string(string&&) = default;
	__device__ string& operator=(const string& str)
	{
		if (&str != this)
		{
			auto tmp = make_cuda_device_unique_ptr<char[]>(str.size() + 1);
			if (bool(tmp))
			{
				if(str.c_str() != nullptr)
					memcpy(tmp.get(), str.c_str(), str.size() + 1);
				this->pStr = std::move(tmp);
				this->strSize = str.size();
			}
		}
		return *this;
	}
	__device__ string& operator=(string&& str) = default;
	__device__ inline string(const char* str) : strSize(strlen(str))
	{
		pStr = make_cuda_device_unique_ptr<char[]>(strSize + 1);
		if (pStr.get() != nullptr && str != nullptr)
		memcpy(pStr.get(), str, strSize + 1);
	}
	//__device__ string(std::size_t size, char ch);
	template <class Iterator>
	__device__ string(Iterator b, Iterator e)
	{
		auto size = e - b;
		std::size_t i = 0;
		pStr = make_cuda_device_unique_ptr<char[]>(size + 1);
		for (Iterator it = b; it != e; ++it)
			pStr.get()[i++] = *it;

		pStr.get()[size] = 0;
		strSize = size;
	}
	__device__ inline std::size_t size() const noexcept { return strSize; }
	__device__ inline const char* c_str() const noexcept
	{
		if (strSize)
			return pStr.get();
		return 0;
	}
	__device__ inline const char* data() const noexcept
	{
		return c_str();
	}

	__device__ string& operator+=(const string& str)
	{
		auto tmp = make_cuda_device_unique_ptr<char[]>(this->size() + str.size() + 1);
		if (bool(tmp))
		{
			if (tmp.get() != nullptr && this != nullptr)
			{
				memcpy(tmp.get(), this->c_str(), this->size());
				memcpy(tmp.get() + this->size(), str.c_str(), str.size() + 1);
			}
			this->pStr = std::move(tmp);
			this->strSize = this->size() + str.size();
		}
		return *this;
	}
	__device__ inline string operator+(const string& str) const
	{
		string temp(*this);
		temp += str;
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
	/*template<class Iterator>
	__device__ Iterator erase(Iterator begin, Iterator end)
	{
		Iterator it;
		string temp(this->end() - this->begin() - (end - begin), 0);
		int k = 0;
		for (auto i = this->begin(); i != this->end(); ++i)
		{
			if (i > begin && i < end)
				continue;
			temp.pStr.get()[k] = *i;
			++k;
		}
		if (++end == pStr.get() + strSize)
			it = pStr.get() + strSize;

		it = ++end;
		*this = temp;
		return it;
	}*/
};
__device__ double stod(const string& str, std::size_t* pos = 0)
{
	char *p;
	double result = strtod(str.c_str(), const_cast<char**>(&p));

	if (pos != 0)
		*pos = p - str.c_str();
	return result;
}


__device__ inline bool operator==(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) == 0);
}

__device__ inline bool operator!=(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) != 0);
}

__device__ inline bool operator>=(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) >= 0);
}
__device__ inline bool operator<=(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) <= 0);
}

__device__ inline bool operator>(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) > 0);
}

__device__ inline bool operator<(const string& str1, const string& str2)
{
	return (strcmp(str1.c_str(), str2.c_str()) < 0);
}

template <class T>
__device__ string to_string(T val)
{
	static_assert(std::is_integral<T>::value, "T must be integral");
	static_assert(std::is_unsigned<T>::value, "T must be unsigned");
	constexpr std::size_t stackSize = (8 * sizeof(T) + ((8 * sizeof(T)) % 3) - 1) / 3;

	char buf[stackSize];

	std::size_t bufSize;

	auto v = val;
	for (bufSize = 0; v != 0; ++bufSize)
	{
		v /= 10;
		buf[bufSize] = v % 10;
	}

	for (auto i = 0; i < bufSize / 2; ++i)
	{
		auto temp = buf[i];
		buf[i] = buf[bufSize - i];
		buf[bufSize - i] = temp;
	}

	return string(&buf[0], &buf[bufSize]);
}

CU_END

#endif //CUDA_STRING_H_
