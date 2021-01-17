﻿#include "cuda_config.cuh"
#include "cuda_return_wrapper.cuh"
#include "cuda_memory.cuh"
#include "cuda_vector.cuh"
#include "cuda_pair.cuh"
#include <cmath>
#include "thrust/complex.h"
#include "cuda_string.cuh"

#ifndef CUDA_TOKENS_H
#define CUDA_TOKENS_H

CU_BEGIN

enum class TokenType
{
	UnaryPlus,
	UnaryMinus,
	BinaryPlus,
	BinaryMinus,
	operatorMul,
	operatorDiv,
	operatorPow,
	sinFunction,
	cosFunction,
	tgFunction,
	log10Function,
	lnFunction,
	logFunction,
	j0Function,
	j1Function,
	jnFunction,
	y0Function,
	y1Function,
	ynFunction,
	gammaFunction,
	argFunction,
	absFunction,
	polarFunction,
	minFunction,
	maxFunction,
	bracket,
	number,
	variable
};

__device__ constexpr bool IsOperatorTokenTypeId(TokenType id)
{
	return id == TokenType::BinaryPlus || id == TokenType::BinaryMinus
		|| id == TokenType::operatorMul || id == TokenType::operatorDiv
		|| id == TokenType::operatorPow;
}


////////////////////////
template <class T>
__device__ T sin(T arg)
{
	return sin(arg);
}
template <class T>
__device__ thrust::complex<T> sin(thrust::complex<T> arg)
{
	//sin a cosh b + i cos a sinh b
	//return thrust::complex<T>(sin(arg.real()) * cosh(arg.imag()), + cos(arg.real()) * sinh(arg.imag()));
	return thrust::sin(arg);
}

template <class T>
__device__ T cos(T arg)
{
	return cos(arg);
}
template <class T>
__device__ thrust::complex<T> cos(thrust::complex<T> arg)
{
	//cos a cosh b − i sin a sinh b
	//return thrust::complex<T>(cos(arg.real()) * cosh(arg.imag()), - (sin(arg.real()) * sinh(arg.imag())));
	return thrust::cos(arg);
}

template <class T>
__device__ T tan(T arg)
{
	return tan(arg);
}
template <class T>
__device__ thrust::complex<T> tan(thrust::complex<T> arg)
{
	//cos a cosh b − i sin a sinh b
	//return thrust::complex<T>(sin(2 * arg.real()), sinh(2 * arg.imag())) / thrust::complex<T>(cos(2 * arg.real(), cosh(2 * y)));
	return thrust::tan(arg);
}

template <class T>
__device__ T gamma(T arg)
{
	return tgamma(arg);
}
template <class T>
__device__ thrust::complex<T> gamma(thrust::complex<T> arg)
{
	const int a = 12;
	static double c_space[12];
	static double *c = NULL;
	thrust::complex<T> accm;
	double pi = acosf(-1);

	if (c == NULL) 
	{
		double k1_factrl = 1.0; 
		c = c_space;
		c[0] = sqrt(2.0 * pi);
		for (auto k = 1; k < a; k++) 
		{
			c[k] = expf(a - k) * pow(T(a - k), T(k - 0.5)) / k1_factrl;
			k1_factrl *= -k;
		}
	}
	accm = c[0];
	for (auto k = 1; k < a; k++) 
	{
		accm = accm + c[k] / (arg + T(k));
	}
	accm *= exp(-(arg + T(a))) * thrust::pow(arg + T(a), arg + T(0.5)); 
	return accm / arg;
}

template <class T>
__device__ T factorial(T n)
{
	auto fact = 1, i = 1;
	while(i <= n)
	{
		fact = fact * i;
		i+= 1;
	}

	return fact;
}

template <class T>
__device__ T j0(T arg)
{
	return j0(arg);
}
template <class T>
__device__ thrust::complex<T> j0(thrust::complex<T> arg)
{
	//alpha = 0
	return jn(T(0), arg);
}

template <class T>
__device__ T j1(T arg)
{
	return j1(arg);
}
template <class T>
__device__ thrust::complex<T> j1(thrust::complex<T> arg)
{
	//alpha = 1
	return jn(T(1), arg);
}

template <class T>
__device__ T jn(T n, T arg)
{
	return jn(n, arg);
}
template <class T>
__device__ thrust::complex<T> jn(T n, thrust::complex<T> arg)
{
	//alpha = n
	thrust::complex<T> result;
	for (auto m = 0; m < 6; ++m)
	{
		result = result + (T(pow(-1, m)) / (factorial(m) * gamma(T(m + n + 1))) * thrust::pow(arg / T(2), T(2 * m + n)));
	}
	return result;
}

template <class T>
__device__ T y0(T arg)
{
	return y0(arg);
}
template <class T>
__device__ thrust::complex<T> y0(thrust::complex<T> arg)
{
	//alpha = 0
	return yn(T(0), arg);
}

template <class T>
__device__ T y1(T arg)
{
	return y1(arg);
}
template <class T>
__device__ thrust::complex<T> y1(thrust::complex<T> arg)
{
	//alpha = 1
	return yn(T(1), arg);
}

template <class T>
__device__ T yn(T n, T arg)
{
	return yn(n, arg);
}
template <class T>
__device__ thrust::complex<T> yn(T n, thrust::complex<T> arg)
{
	auto pi = acosf(-1);
	return (jn(n, arg) * T(cos(n)) * T(pi) - jn(-n, arg)) / T((sin(n) * pi));
}

template <class T>
__device__ T abs(T arg)
{
	return std::abs(arg);
}
template <class T>
__device__ thrust::complex<T> abs(thrust::complex<T> arg)
{
	//alpha = 0
	return thrust::abs(arg);
}

template <class T>
__device__ T arg(T z)
{
	return arg(z);
}
template <class T>
__device__ thrust::complex<T> arg(thrust::complex<T> z)
{
	return thrust::arg(z);
}

template <class T>
__device__ T pow(T base, T arg)
{
	return pow(base, arg);
}
template <class T>
__device__ thrust::complex<T> pow(thrust::complex<T> base, thrust::complex<T> arg)
{
	//alpha = 0
	return thrust::pow(base, arg);
}

template <class T>
__device__ T ln(T arg)
{
	return std::log(arg);
}
template <class T>
__device__ thrust::complex<T> ln(thrust::complex<T> arg)
{
	//alpha = 0
	return thrust::log(arg);
}

template <class T>
__device__ T log10(T arg)
{
	return std::log10(arg);
}
template <class T>
__device__ thrust::complex<T> log10(thrust::complex<T> arg)
{
	//alpha = 0
	return thrust::log10(arg);
}

template <class T>
__device__ T log(T arg1, T arg2)
{
	return std::log(arg1) / std::log(arg2);
}
template <class T>
__device__ thrust::complex<T> log(thrust::complex<T> arg1, thrust::complex<T> arg2)
{
	//alpha = 0
	return thrust::log(arg1) / thrust::log(arg2);
}
////////////////////////

template <class T, std::size_t N>
class static_parameter_storage
{
	struct { T params[N]; } strg;
	T* top = strg.params;
public:
	__device__ static_parameter_storage() = default;
	//__device__ static_parameter_storage(const static_parameter_storage& right)
	//{
	//	*this = right;
	//}
	__device__ static_parameter_storage(static_parameter_storage&& right)
	{
		*this = std::move(right);
	}
	__device__ static_parameter_storage& operator=(static_parameter_storage&& right)
	{
		strg = std::move(right.strg);
		return *this;
	}
	//__device__ static_parameter_storage& operator=(static_parameter_storage&& right)
	//{
	//	strg = std::move(right.strg);
	//	return *this;
	//}
	__device__ return_wrapper_t<const T&> operator[](std::size_t index) const
	{
		if (index < N)
			return return_wrapper_t<const T&>(strg.params[index]); //TODO не должно быть move
		return return_wrapper_t<const T&>(CudaParserErrorCodes::InvalidArgument);
	}
	__device__ return_wrapper_t<T&> operator[](std::size_t index)
	{
		if (index < N)
			return return_wrapper_t<T&>(strg.params[index]); //TODO не должно быть move
		return return_wrapper_t<T&>(CudaParserErrorCodes::InvalidArgument);
	}
	template <class U>
	__device__ auto push_argument(U&& arg) -> std::enable_if_t<std::is_convertible<std::decay_t<U>, T>::value, return_wrapper_t<void>>
	{
		if (top - strg.params >= N)
			return return_wrapper_t<void>(CudaParserErrorCodes::InvalidArgument);
		*(top++) = std::move(arg);
		return return_wrapper_t<void>();
	}
	__device__ bool is_ready() const
	{
		return top == &strg.params[N] && this->is_ready_from<0>();
	}
private:
	template <std::size_t I, class = void>
	__device__ auto is_ready_from() const -> std::enable_if_t<(I >= N), bool>
	{
		return true;
	}
	template <std::size_t I, class = void>
	__device__ auto is_ready_from() const->std::enable_if_t<(I < N), bool>
	{
		return strg.params[I]->is_ready() && this->is_ready_from<I + 1>();
	}
};

template <class T>
class expr_param_init_block
{
	cu::vector<cu::pair<const cu::string*, T>> m_sorted_arguments;
public:
	__device__ expr_param_init_block() = default;
	__device__ inline expr_param_init_block(cu::vector<cu::pair<const cu::string*, T>>&& sorted_arg_frame):m_sorted_arguments(std::move(sorted_arg_frame)) {}
	__device__ cu::return_wrapper_t<T> get_parameter(const char* pName, std::size_t cbName) const
	{
		auto pFrameBegin = &m_sorted_arguments[0];
		auto count = m_sorted_arguments.size();
		while (count != 0)
		{
			auto mid = count / 2;
			auto cmp = cu::strncmpnz(pFrameBegin[mid].first->c_str(), pFrameBegin[mid].first->size(), pName, cbName);
			if (cmp == 0)
				return cu::return_wrapper_t<T>(pFrameBegin[mid].second);
			if (cmp < 0)
			{
				pFrameBegin = &pFrameBegin[mid];
				count -= mid;
			}
			else
				count = mid;
		}
		return cu::make_return_wrapper_error<T>(CudaParserErrorCodes::ParameterIsNotFound);
	}
};

template <class T>
class IToken
{
public:
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const = 0;
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value) = 0;
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const = 0;
	__device__ virtual std::size_t get_required_parameter_count() const = 0;
	__device__ virtual bool is_ready() const = 0; //all parameters are specified
	__device__ virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
	__device__ virtual TokenType type() = 0;
	__device__ virtual short getPriority() = 0;
};

template <class T>
class Number : public IToken<T>
{
public:
	__device__ Number(T val) : m_val(val) {};

	__device__ inline const T& value() const
	{
		return m_val;
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>&) const
	{
		return return_wrapper_t<T>(m_val);
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(*this)));
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ T operator+(const Number<T>& num) const
	{
		return this->m_val + num();
	}
	__device__ T operator-(const Number<T>& num) const
	{
		return this->m_val - num();
	}
	__device__ T operator+() const
	{
		return this->m_val;
	}
	__device__ T operator-() const
	{
		return -this->m_val;
	}
	__device__ T operator*(const Number<T>& num) const
	{
		return this->m_val * num();
	}
	__device__ T operator/(const Number<T>& num) const
	{
		return this->m_val / num();
	}
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& m_val)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual TokenType type()
	{
		return TokenType::number;
	}
	__device__ virtual short getPriority()
	{
		return -2;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 0;
	}
private:
	T m_val;
};

template <class T>
class PI : public IToken<T>
{
public:
	__device__ inline const T& value() const
	{
		return m_val;
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>&) const
	{
		return return_wrapper_t<T>(m_val);
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(this->m_val)));
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& m_val)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual TokenType type()
	{
		return TokenType::number;
	}
	__device__ virtual short getPriority()
	{
		return -2;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 0;
	}
private:
	T m_val = acosf(-1);
};

template <class T>
class Euler : public IToken<T>
{
public:
	__device__ inline const T& value() const
	{
		return m_val;
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>&) const
	{
		return return_wrapper_t<T>(m_val);
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(this->m_val)));
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& m_val)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual TokenType type()
	{
		return TokenType::number;
	}
	__device__ virtual short getPriority()
	{
		return -2;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 0;
	}
private:
	T m_val = std::exp(1.0);
};
//template <class T>
//class Header;

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{
	cu::string m_name;
public:
	__device__ Variable(const char* varname, std::size_t len):m_name(varname, varname + len) {}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Variable<T>>(*this));
	}
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 0;
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		return args.get_parameter(m_name.data(), m_name.size());
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::variable;
	}
	__device__ virtual short getPriority()
	{
		return -2;
	}
};

template <class T>
class Operator : public IToken<T>
{
public:
	//__device__ virtual short getPriority()
	//{
	//	return 0; //default priority, less code but more error prone
	//}
	//__device__ virtual TokenType type()
	//{
	//	return TokenType::Operator;
	//}
	__device__ return_wrapper_t<void> set_required_parameter_count(short value)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
};

template <class T>
class UnaryPlus : public Operator<T> //+-*/
{
	/*This replacement is unnecessary, but the code would be more maintainable, if the storage of parameters
	for functions (with fixed numbers of the parameters) will be managed in one place (static_parameter_storage). */
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		return rw.value()->compute(args);
		//return return_wrapper_t<T>((ops[0].get())->get()->operator()());
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		return rw.value()->simplify(); //unary + does no do anything

		//return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>((ops[0].get())->simplify()); //unary + does no do anything
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::UnaryPlus;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
};

template <class T>
class BinaryPlus : public Operator<T> //+-*/
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error<T>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0c = rw0.value()->compute(args);
		if (!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if (!rw1c)
			return rw1c;

		return rw0c.value() + rw1c.value();
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(static_cast<Number<T>&>(*op0).value() + static_cast<Number<T>&>(*op1).value()));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<cu::IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<BinaryPlus<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::BinaryPlus;
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
};

template <class T>
class UnaryMinus : public Operator<T> //+-*/
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		return -rw.value()->compute(args).value();
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(-static_cast<Number<T>&>(*rws.value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<cu::IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<UnaryMinus<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual TokenType type()
	{
		return TokenType::UnaryMinus;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
};
template <class T>
class BinaryMinus : public Operator<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return rw1c.value() - rw0c.value(); 
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::BinaryMinus;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(static_cast<Number<T>&>(*op1).value() - static_cast<Number<T>&>(*op0).value()));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));				
		}
		auto op_new = make_cuda_device_unique_ptr<BinaryMinus<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
};

template <class T>
class OperatorMul : public Operator<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error<T>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0c = rw0.value()->compute(args);
		if (!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if (!rw1c)
			return rw1c;

		return rw0c.value() * rw1c.value();
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorMul;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(static_cast<Number<T>&>(*op0).value() * static_cast<Number<T>&>(*op1).value()));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<OperatorMul<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return rw1c.value() / rw0c.value(); 
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorDiv;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(static_cast<Number<T>&>(*op1).value() / static_cast<Number<T>&>(*op0).value()));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));				
		}
		auto op_new = make_cuda_device_unique_ptr<OperatorDiv<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class OperatorPow : public Operator<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return cu::pow(rw1c.value(), rw0c.value()); 
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorPow;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::pow(static_cast<Number<T>&>(*op1).value(), static_cast<Number<T>&>(*op0).value())));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<OperatorPow<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
public:
	__device__ virtual return_wrapper_t<void> set_required_parameter_count(std::size_t value)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
};

template <class T>
class GammaFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::gamma(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "gamma";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::gammaFunction;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if (!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::gamma(static_cast<const Number<T>&>((*rws.value())).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<GammaFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};

template <class T>
class AbsFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::abs(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "abs";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::absFunction;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if (!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::abs(static_cast<const Number<T>&>((*rws.value())).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<AbsFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};

template <class T>
class ArgFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::arg(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "arg";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::argFunction;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if (!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if (!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::arg(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<ArgFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};

template <class T>
class SinFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::sin(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "sin";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::sinFunction;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::sin(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<SinFunction<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class CosFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::cos(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "cos";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::cosFunction;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::cos(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<CosFunction<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class TgFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::tan(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__  const char* get_function_name() const
	{
		return "tg";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::tgFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::tan(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<TgFunction<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};

template <class T>
class Log10Function : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return cu::log10(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "log10";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::log10Function;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
	if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::log10(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<Log10Function<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class LnFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);
		if (!rwc)
			return rwc;

		return cu::ln(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "ln";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::lnFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::ln(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<LnFunction<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class LogFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return cu::log(rw1c.value(), rw0c.value()); 
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "log";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::logFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(cu::log(static_cast<Number<T>&>(*op1).value(), static_cast<Number<T>&>(*op0).value())));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<LogFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class JnFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return jn(T(rw0c.value()), rw1c.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "jn";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::jnFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(jn(static_cast<Number<T>&>(*op0).value(), static_cast<Number<T>&>(*op1).value()))); //maybe need std::sph_bessel(unsigned n, double x) or std::cyl_bessel(double v, double x)
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<JnFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class J0Function : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return j0(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "j0";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::j0Function;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(j0(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<J0Function<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class J1Function : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return j1(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "j1";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::j1Function;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(j1(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<J1Function<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class YnFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if(!rw0)
			return rw0;
		auto rw1 = ops[1];
		if(!rw1)
			return rw1;
		
		auto rw0c = rw0.value()->compute(args);
		if(!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if(!rw1c)
			return rw1c;

		return yn(rw0c.value(), rw1c.value()); 
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "yn";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::ynFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(yn(static_cast<Number<T>&>(*op0).value(), static_cast<Number<T>&>(*op1).value())));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<YnFunction<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class Y0Function : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return y0(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "y0";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::y0Function;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(y0(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<Y0Function<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class Y1Function : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 1> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rwc = rw.value()->compute(args);

		return y1(rwc.value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "y1";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::y1Function;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotReady);
		auto rw = ops[0];
		if(!rw)
			return rw;
		auto rws = rw.value()->simplify();
		if(!rws)
			return rws;

		if (rws.value()->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(y1(static_cast<const Number<T>&>(*rws.value()).value())));
			if (!ptr)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<T>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<Y1Function<T>>();
		if(!op_new)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rwp = op_new->push_argument(std::move(rws.value()));
		if (!rwp)
			return rwp;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
/////////////TODO
template <class T>
class PolarFunction {};

//PolarFunction<thrust::complex<double>>
template <class T>
class PolarFunction<thrust::complex<T>>: public Function<thrust::complex<T>>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<thrust::complex<T>>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<thrust::complex<T>>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<thrust::complex<T>> compute(const expr_param_init_block<thrust::complex<T>>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;

		auto rw0c = rw0.value()->compute(args);
		if (!rw0c)
			return rw0c;
		auto rw1c = rw1.value()->compute(args);
		if (!rw1c)
			return rw1c;

		return thrust::polar(rw1c.value().real(), rw0c.value().real());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "polar";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::polarFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<thrust::complex<T>>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<thrust::complex<T>>>>(CudaParserErrorCodes::NotReady);
		auto rw0 = ops[0];
		if (!rw0)
			return rw0;
		auto rw1 = ops[1];
		if (!rw1)
			return rw1;
		auto rw0s = rw0.value()->simplify();
		if (!rw0s)
			return rw0s;
		auto rw1s = rw1.value()->simplify();
		if (!rw1s)
			return rw1s;
		auto& op0 = rw0s.value();
		auto& op1 = rw1s.value();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<thrust::complex<T>>>(Number<thrust::complex<T>>(thrust::polar(static_cast<Number<thrust::complex<T>>&>(*op1).value().real(),
				static_cast<Number<thrust::complex<T>>&>(*op0).value().real())));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<thrust::complex<T>>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return cuda_device_unique_ptr<IToken<thrust::complex<T>>>(std::move(ptr));
		}
		auto op_new = make_cuda_device_unique_ptr<PolarFunction<thrust::complex<T>>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<thrust::complex<T>>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rw = op_new->push_argument(std::move(op1));
		if (!rw)
			return rw;
		rw = op_new->push_argument(std::move(op0));
		if (!rw)
			return rw;
		return return_wrapper_t<cuda_device_unique_ptr<IToken<thrust::complex<T>>>>(std::move(op_new));
	}
};

template <>
class PolarFunction <double> : public Function<double>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<double>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<double>>&& value)
	{
		return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidArgument);
	}
	__device__ virtual return_wrapper_t<double> compute(const expr_param_init_block<double>& args) const
	{
		return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidArgument);
	}
	__device__ virtual bool is_ready() const
	{
		return false;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "polar";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::polarFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<double>>> simplify() const
	{
		return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidArgument);
	}
};

template <class T, class Implementation, class TokenBinPredicate>
class ExtremumFunction : public Function<T>
{
	cu::vector<cuda_device_unique_ptr<IToken<T>>> ops;
	std::size_t nRequiredParamsCount = 0;
	TokenBinPredicate m_pred;

	//ExtremumFunction(cu::vector<cuda_device_unique_ptr<IToken<T>>>&& operands, const TokenBinPredicate& pred)
public:
	__device__ ExtremumFunction() = default;
	__device__ ExtremumFunction(std::size_t paramsNumber) : nRequiredParamsCount(paramsNumber) {}

	//not used, but in case a state is needed by the definition of the predicate:
	template <class Predicate, class = std::enable_if_t<std::is_constructible<TokenBinPredicate, Predicate&&>::value>>
	__device__ ExtremumFunction(std::size_t paramsNumber, Predicate&& pred) : nRequiredParamsCount(paramsNumber), m_pred(std::forward<Predicate>(pred)) {}

	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_back(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error(CudaParserErrorCodes::NotReady);
		auto rwExtr = ops[0]->compute(args);
		if (!rwExtr)
			return rwExtr;
		auto extrVal = rwExtr.value();
		for (typename cu::vector<cuda_device_unique_ptr<IToken<T>>>::size_type iElement = 1; iElement < ops.size(); ++iElement)
		{
			auto rw_i = ops[iElement]->compute(args);
			if (!rw_i)
				return rw_i;
			if (m_pred(rw_i.value(), extrVal))
				extrVal = rw_i.value();
		}
		return extrVal;
	}
	__device__ virtual bool is_ready() const
	{
		if (ops.empty() || ops.size() != nRequiredParamsCount)
			return false;
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			if (!op->get()->is_ready())
				return false;
		}
		return true;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return nRequiredParamsCount;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const///////////////////////////////TODO:continue
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		cu::vector<cuda_device_unique_ptr<IToken<T>>> new_ops;
		T extrem;
		bool fExtrem = false;
		new_ops.reserve(ops.size());

		for (const auto& op : ops)
		{
			auto newarg = op->simplify();
			if (!newarg)
				return newarg;
			auto& new_op = newarg.value();
			if (new_op->type() == TokenType::number)
			{
				auto& num = static_cast<Number<T>&>(*new_op);
				if (!fExtrem || m_pred(num.value(), extrem))
				{
					extrem = num.value();
					fExtrem = true;
				}
			}else
				new_ops.emplace_back(std::move(new_op));
		}
		if (fExtrem)
		{
			auto tkn = make_cuda_device_unique_ptr<Number<T>>(std::move(extrem));
			if (!tkn)
				return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
			if (new_ops.empty())
				return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(tkn));
			auto rw = new_ops.emplace_back(std::move(tkn));
			if (!rw)
				return rw;
		}
		auto pNewTkn = make_cuda_device_unique_ptr<Implementation>();
		if (!pNewTkn)
			return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
		static_cast<ExtremumFunction&>(*pNewTkn).ops = std::move(new_ops);
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
	__device__ return_wrapper_t<void> set_required_parameter_count(std::size_t value)
	{
		nRequiredParamsCount = value;
		return return_wrapper_t<void>();
	}
};

template <class T, class Implementation, class TokenBinPredicate>
class ExtremumFunction<thrust::complex<T>, Implementation, TokenBinPredicate> : public Function<thrust::complex<T>>
{
	cu::vector<cuda_device_unique_ptr<IToken<thrust::complex<T>>>> ops;
	std::size_t nRequiredParamsCount = 0;
	TokenBinPredicate m_pred;

	//ExtremumFunction(cu::vector<cuda_device_unique_ptr<IToken<T>>>&& operands, const TokenBinPredicate& pred)
public:
	__device__ ExtremumFunction() = default;
	__device__ ExtremumFunction(std::size_t paramsNumber) : nRequiredParamsCount(paramsNumber) {}

	//not used, but in case a state is needed by the definition of the predicate:
	template <class Predicate, class = std::enable_if_t<std::is_constructible<TokenBinPredicate, Predicate&&>::value>>
	__device__ ExtremumFunction(std::size_t paramsNumber, Predicate&& pred) : nRequiredParamsCount(paramsNumber), m_pred(std::forward<Predicate>(pred)) {}

	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<thrust::complex<T>>>&& value)
	{
		return cu::make_return_wrapper_error(CudaParserErrorCodes::InvalidArgument);
	}
	__device__ virtual return_wrapper_t<thrust::complex<T>> compute(const expr_param_init_block<thrust::complex<T>>& args) const
	{
		return cu::make_return_wrapper_error(CudaParserErrorCodes::InvalidArgument);
	}
	__device__ virtual bool is_ready() const
	{
		return false;
	}
	__device__ virtual short getPriority()
	{
		return 4;
	}
	__device__ virtual std::size_t get_required_parameter_count() const
	{
		return nRequiredParamsCount;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<thrust::complex<T>>>> simplify() const///////////////////////////////TODO:continue
	{
		return make_return_wrapper_error(CudaParserErrorCodes::InvalidArgument);
	}
	__device__ return_wrapper_t<void> set_required_parameter_count(std::size_t value)
	{
		return return_wrapper_t<void>();
	}
};

template <class T>
struct TokenLess
{
	__device__ constexpr bool operator()(const T& left, const T& right) const
	{
		return left < right;
	};
};

template <class T>
struct TokenGreater
{
	__device__ constexpr bool operator()(const T& left, const T& right) const
	{
		return left > right;
	};
};

template <class T>
class MaxFunction : public ExtremumFunction<T, MaxFunction<T>, TokenGreater<T>>
{
	typedef ExtremumFunction<T, MaxFunction<T>, TokenGreater<T>> MyBase;
public:
	using MyBase::ExtremumFunction; //c-tor inheritance
	__device__ virtual const char* get_function_name() const
	{
		return "max";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::maxFunction;
	}
};

template <class T>
class MinFunction : public ExtremumFunction<T, MinFunction<T>, TokenLess<T>>
{
	typedef ExtremumFunction<T, MinFunction<T>, TokenLess<T>> MyBase;
public:
	using MyBase::ExtremumFunction; //c-tor inheritance
	__device__ virtual const char* get_function_name() const
	{
		return "min";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::minFunction;
	}
};

template <class T>
class Bracket : public Operator<T> //,' '()
{
public:
	__device__ Bracket() = default;

	__device__ virtual return_wrapper_t<T> compute(const expr_param_init_block<T>& args) const
	{
		return return_wrapper_t<T>(true);
	}

	__device__ virtual bool is_ready() const
	{
		return true;
	}

	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return return_wrapper_t<void>(); //openingBracket = value; //true is for opening bracket, false is for closing.
	}

	__device__ virtual short getPriority()
	{
		return -1;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::bracket;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		//return make_cuda_device_unique_ptr<Bracket<T>>(nullptr);
		//throw std::exception("Unexpected call");
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ std::size_t get_required_parameter_count() const
	{
		return 0;
	}
	//__device__ void set_required_parameter_count(std::size_t value)
	//{
	//	throw std::exception("Invalid operation");
	//}
};

CU_END

#endif // !CUDA_TOKENS_H
