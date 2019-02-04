#include "cuda_config.cuh"
#include "cuda_return_wrapper.cuh"
#include "cuda_memory.cuh"
#include "cuda_vector.cuh"

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
	cuda_vector<cu::pair<cu::cuda_string*, T>> m_sorted_arguments;
public:
	expr_param_init_block() = default;
	expr_param_init_block(cuda_vector<cu::pair<cu::cuda_string*, T>>&& sorted_arg_frame):m_sorted_arguments(std::move(sorted_arg_frame)) {}
	cu::return_wrapper_t<T> get_parameter(const char* pName, std::size_t cbName) const
	{
		auto pFrameBegin = &m_sorted_arguments[0];
		auto cb = cbName;
		while (cb != 0)
		{
			auto mid = cb / 2;
			auto cbMin = pFrameBegin[mid].first->size();
			if (cbName < cbMin)
				cbMin = cbName;
			auto cmp = cu::strncmpnz(pFrameBegin[mid].first->c_str(), pName, cbMin);
			if (cmp == 0)
				return cu::return_wrapper_t<T>(pFrameBegin[min].second);
			if (cmp < 0)
			{
				pFrameBegin = &pFrameBegin[mid];
				cb -= min;
			}else
				cb = min;
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
class Header;

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{
	cuda_string m_name;
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
		auto rv = ops[0];
		if (!rv)
			return rv;
		return rv.value()->compute(args);
		//return return_wrapper_t<T>((ops[0].get())->get()->operator()());
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rv = ops[0];
		if (!rv)
			return rv;
		return rv.value()->simplify(); //unary + does no do anything

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
		auto rv0 = ops[0];
		if (!rv0)
			return rv0;
		auto rv1 = ops[1];
		if (!rv1)
			return rv1;
		auto rv0c = rv0.value()->compute(args);
		if (!rv0c)
			return rv0c;
		auto rv1c = rv1.value()->compute(args);
		if (!rv1c)
			return rv1c;

		return rv0c.value() + rv1c.value();
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto rv0 = ops[0];
		if (!rv0)
			return rv0;
		auto rv1 = ops[1];
		if (!rv1)
			return rv1;
		auto rv0s = rv0.value()->simplify();
		if (!rv0s)
			return rv0s;
		auto rv1s = rv1.value()->simplify();
		if (!rv1s)
			return rv1s;
		auto& op0 = *rv0s.value();
		auto& op1 = *rv1s.value();

		if (op0.type() == TokenType::number && op1.type() == TokenType::number)
		{
			auto ptr = make_cuda_device_unique_ptr<Number<T>>(Number<T>(static_cast<Number<T>&>(op0).value() + static_cast<Number<T>&>(op1).value()));
			if (!ptr)
				return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
			return ptr;
		}
		auto op_new = make_cuda_device_unique_ptr<BinaryPlus<T>>();
		if (!op_new)
			return cu::make_return_wrapper_error<cuda_device_unique_ptr<IToken<T>>>(cu::CudaParserErrorCodes::NotEnoughMemory);
		auto rv = op_new->push_argument(std::move(op0));
		if (!rv)
			return rv;
		rv = op_new->push_argument(std::move(op1));
		if (!rv)
			return rv;
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
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(-*((ops[0].get())->get()->operator()().get()));
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!this->is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto op0 = std::move(*((ops[0].get())->get()->simplify().get()));

		if (op0->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(-*static_cast<Number<T>*>(op0.get()))); ///////////NOT READY
		auto op_new = make_cuda_device_unique_ptr<UnaryMinus<T>>();
		op_new->push_argument(std::move(op0));
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
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return  return_wrapper_t<T>(*((ops[1].get())->get()->operator()().get()) - *((ops[0].get())->get()->operator()().get()));
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
		auto op0 = (ops[0].get())->get()->simplify();
		auto op1 = (ops[1].get())->get()->simplify();

		if (op0->get()->type() == TokenType::number && op1->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(*((ops[1].get())->get()->operator()().get()) - *((ops[0].get())->get()->operator()().get()))));
		auto op_new = make_cuda_device_unique_ptr<BinaryMinus<T>>();
		op_new->push_argument(std::move(*op0.get()));
		op_new->push_argument(std::move(*op1.get()));
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
};

//template <class T>
//class OperatorMinus : public Operator<T>
//{
//	std::shared_ptr<IToken<T>> ops[2], *top = ops;
//
//public:
//	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
//	{
//		*top++ = value;
//	}
//	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
//	{
//		if (!ops[0]->is_ready() || !ops[1]->is_ready())
//		//	throw std::exception("Insufficient number are given for the plus operator.");
//			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);
//
//		return return_wrapper_t<T>(*(*ops[0])().get() - *(*ops[1])().get());
//	}
//	__device__ virtual bool is_ready() const
//	{
//		return true;
//	}
//	__device__ virtual std::size_t get_params_count() const
//	{
//		return 2;
//	}
//	__device__ virtual TokenType type()
//	{
//		return TokenType::operatorMinus;
//	}
//	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
//	{
//		if (!is_ready())
//			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
//		auto op0 = ops[0]->simplify();
//		auto op1 = ops[1]->simplify();
//
//		if ((op0.get())->get()->type()  == TokenType::number && (op0.get())->get()->type() == TokenType::number)
//			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() - *((op0.get())->get())()));
//		auto op_new = std::make_shared<OperatorMinus<T>>();
//		op_new->push_argument(std::move(*(op0.get())));
//		op_new->push_argument(std::move(*(op1.get())));
//		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
//	}
//};
template <class T>
class OperatorMul : public Operator<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		return ops.push_argument(std::move(value));
	}
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops.is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return return_wrapper_t<T>((*ops[0].value())().value() * (*ops[1].value())().value());
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual short getPriority()
	{
		return 1;
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
		typedef cuda_device_unique_ptr<IToken<T>> my_token_sptr;
		if (!is_ready())
			return return_wrapper_t<my_token_sptr>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0].value()->simplify();
		if (op0.return_code() != CudaParserErrorCodes::Success)
			return return_wrapper_t<my_token_sptr>(op0.return_code());
		auto op1 = ops[1].value()->simplify();
		if (op1.return_code() != CudaParserErrorCodes::Success)
			return return_wrapper_t<my_token_sptr>(op1.return_code());

		if (op0->get()->type() == TokenType::number && op1->get()->type() == TokenType::number)
		{
			auto result = (*op0.value())().value() * (*op1.value())().value();
			return return_wrapper_t<my_token_sptr>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::move(result))));
		}
		auto op_new = make_cuda_device_unique_ptr<OperatorMul<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<my_token_sptr>(std::move(op_new));
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
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops.is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return return_wrapper_t<T>((*ops[1].value())().value() / (*ops[0].value())().value());
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 1;
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
		typedef cuda_device_unique_ptr<IToken<T>> my_token_sptr;
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<OperatorDiv<T>>(), CudaParserErrorCodes::NotReady);
		auto op0 = ops[0].value()->simplify();
		if (op0.return_code() != CudaParserErrorCodes::Success)
			return return_wrapper_t<my_token_sptr>(op0.return_code());
		auto op1 = ops[1].value()->simplify();
		if (op1.return_code() != CudaParserErrorCodes::Success)
			return return_wrapper_t<my_token_sptr>(op1.return_code());

		if ((op0.get())->get()->type() == TokenType::number && (op0.get())->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>((*op1.value())().value() / (*op0.value())().value())));
		auto op_new = make_cuda_device_unique_ptr<OperatorDiv<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
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
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops.is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return return_wrapper_t<T>(std::pow((*ops[1].value())().value(), (*ops[0].value())().value()));
	}
	__device__ virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 2;
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<OperatorPow<T>>(), CudaParserErrorCodes::NotReady);
		auto op0 = ops[0].value()->simplify();
		auto op1 = ops[1].value()->simplify();

		if ((op0.get())->get()->type() == TokenType::number && (op1.get())->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::pow((*op1.value())().value(), (*op0.value())().value()))));
		//auto op_new = std::make_shared<OperatorPlus<T>>();
		auto op_new = make_cuda_device_unique_ptr<OperatorPow<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
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
class SinFunction : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return return_wrapper_t<T>(std::sin(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<SinFunction<T>>(), CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::sin((*newarg.value())().value()))));
		//return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = make_cuda_device_unique_ptr<SinFunction<T>>();
		pNewTkn->op = std::move(*(newarg.get()));
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class CosFunction : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return return_wrapper_t<T>(std::cos(*((op.get())->operator()()).get()));
		//std::cos((*op)());
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::cos((*newarg.value())().value()))));
		//return make_cuda_device_unique_ptr<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = make_cuda_device_unique_ptr<CosFunction<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class TgFunction : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidNumberOfArguments);

		return  return_wrapper_t<T>(std::tan(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::tan((*newarg.value())().value()))));
		//return make_cuda_device_unique_ptr<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = make_cuda_device_unique_ptr<TgFunction<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};

template <class T>
class Log10Function : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(std::log10(std::move(*((op.get())->operator()()).get())));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if (newarg->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::log10((*newarg.value())().value()))));
		auto pNewTkn = make_cuda_device_unique_ptr<Log10Function<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class LnFunction : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(std::log(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if (newarg->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(std::log((*newarg.value())().value()))));
		auto pNewTkn = make_cuda_device_unique_ptr<LnFunction<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class LogFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		ops.push_argument(std::move(value));
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(std::log((*ops[1].value())().value()) / std::log((*ops[0].value())().value()));
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto op0 = (ops[0].get())->get()->simplify();
		auto op1 = (ops[1].get())->get()->simplify();

		if (op0->get()->type() == TokenType::number && op1->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(
				std::log((*op1.value())().value()) / std::log((*op0.value())().value()))));
		auto op_new = make_cuda_device_unique_ptr<LogFunction<T>>();
		op_new->push_argument(std::move(*op0.get()));
		op_new->push_argument(std::move(*op1.get()));
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
		ops.push_argument(std::move(value));
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(0, CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(jn(int((*ops[0].value())().value()), double((*ops[1].value())().value())));
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
		return TokenType::logFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<JnFunction<T>>(), CudaParserErrorCodes::NotReady);
		auto op0 = (ops[0].get())->get()->simplify();
		auto op1 = (ops[1].get())->get()->simplify();

		if (op0->get()->type() == TokenType::number && op1->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(jn(int((*ops[0].value())().value()), double((*ops[1].value())().value())))));
		auto op_new = make_cuda_device_unique_ptr<JnFunction<T>>();
		op_new->push_argument(std::move(*op0.get()));
		op_new->push_argument(std::move(*op1.get()));
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class J0Function : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(0, CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(j0(*((op.get())->operator()()).get()), CudaParserErrorCodes::Success);
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		return TokenType::lnFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<J0Function<T>>(), CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if (newarg->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(j0((*newarg.value())().value()))));
		auto pNewTkn = make_cuda_device_unique_ptr<J0Function<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class J1Function : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(0, CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(j1(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		return TokenType::lnFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if (newarg->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(j1(*((newarg.get()->get()->operator()()).get())))));
		auto pNewTkn = make_cuda_device_unique_ptr<J1Function<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class YnFunction : public Function<T>
{
	static_parameter_storage<cuda_device_unique_ptr<IToken<T>>, 2> ops;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		ops.push_argument(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(yn(int(*((ops[0].get()->get()->operator()()).get())), *((ops[1].get()->get()->operator()()).get())));
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
		return TokenType::logFunction;
	}
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<YnFunction<T>>(), CudaParserErrorCodes::NotReady);
		auto op0 = (ops[0].get())->get()->simplify();
		auto op1 = (ops[1].get())->get()->simplify();

		if (op0->get()->type() == TokenType::number && op1->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(yn(int(*((ops[0].get()->get()->operator()()).get())), *((ops[1].get()->get()->operator()()).get())))));
		auto op_new = make_cuda_device_unique_ptr<YnFunction<T>>();
		op_new->push_argument(std::move(*op0.get()));
		op_new->push_argument(std::move(*op1.get()));
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(op_new));
	}
};
template <class T>
class Y0Function : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(0, CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(y0(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Y0Function<T>>(), CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if (newarg->get()->type() == TokenType::number)
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(y0(*(newarg.get()->get()->operator()().get())))));
		auto pNewTkn = make_cuda_device_unique_ptr<Y0Function<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};
template <class T>
class Y1Function : public Function<T>
{
	cuda_device_unique_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		op = std::move(value);
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(T(), CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>(y1(*((op.get())->operator()()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
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
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Y1Function<T>>(), CudaParserErrorCodes::NotReady);
		auto newarg = op.get()->simplify(); // (ops[0].get())->get()->simplify()
		if ((newarg.get())->get()->type() == TokenType::number)

			//*(newarg.get()->get()->operator().get())
			//(*op0.value())().value()
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(make_cuda_device_unique_ptr<Number<T>>(Number<T>(y1((*newarg.value())().value()))));
		auto pNewTkn = make_cuda_device_unique_ptr<Y1Function<T>>();
		pNewTkn->op = std::move(newarg.value());
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
};

template <class T, class Implementation, class TokenBinPredicate>
class ExtremumFunction : public Function<T>
{
	cuda_vector<cuda_device_unique_ptr<IToken<T>>> ops;
	std::size_t nRequiredParamsCount = 0;
	TokenBinPredicate m_pred;
public:
	__device__ ExtremumFunction() = default;
	__device__ ExtremumFunction(std::size_t paramsNumber) : nRequiredParamsCount(paramsNumber) {}

	//not used, but in case a state is needed by the definition of the predicate:
	template <class Predicate, class = std::enable_if_t<std::is_constructible<TokenBinPredicate, Predicate&&>::value>>
	__device__ ExtremumFunction(std::size_t paramsNumber, Predicate&& pred) : nRequiredParamsCount(paramsNumber), m_pred(std::forward<Predicate>(pred)) {}

	__device__ virtual return_wrapper_t<void> push_argument(cuda_device_unique_ptr<IToken<T>>&& value)
	{
		ops.push_back(std::move(value));
		return return_wrapper_t<void>();
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);

		return return_wrapper_t<T>((*cu::min_element(ops.begin(), ops.end(), m_pred)).get()->operator()());
	}
	__device__ virtual bool is_ready() const
	{
		if (ops.size() != nRequiredParamsCount)
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
	__device__ virtual return_wrapper_t<cuda_device_unique_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(CudaParserErrorCodes::NotReady);
		cuda_vector<cuda_device_unique_ptr<IToken<T>>> newargs;
		newargs.reserve(ops.size());
		cuda_vector<cuda_device_unique_ptr<IToken<T>>> newargsVar;
		newargsVar.reserve(ops.size());

		for (const auto& op : ops)
		{
			auto newarg = op->simplify();
			if (newarg->get()->type() == TokenType::number)
				newargs.push_back(make_cuda_device_unique_ptr<Number<T>>(Number<T>((*newarg.value())().value())));
			else
				newargsVar.push_back(std::move(newarg.value()));
		}
		if (newargsVar.empty())
			return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(*cu::min_element(newargs.begin(), newargs.end(), m_pred)));

		auto pNewTkn = make_cuda_device_unique_ptr<Implementation>();
		if (newargs.empty())
			pNewTkn = make_cuda_device_unique_ptr<Implementation>(Implementation(newargsVar.size()));
		else
		{
			pNewTkn = make_cuda_device_unique_ptr<Implementation>(Implementation(newargsVar.size() + 1));
			pNewTkn->push_argument(std::move(*cu::min_element(newargs.begin(), newargs.end(), m_pred)));
		}
		for (auto& op : newargsVar)
			pNewTkn->push_argument(std::move(op));
		return return_wrapper_t<cuda_device_unique_ptr<IToken<T>>>(std::move(pNewTkn));
	}
	__device__ return_wrapper_t<void> set_required_parameter_count(std::size_t value)
	{
		nRequiredParamsCount = value;
		return return_wrapper_t<void>();
	}
};

template <class T>
struct TokenLess
{
	__device__ bool operator()(const cuda_device_unique_ptr<IToken<T>>& left, const cuda_device_unique_ptr<IToken<T>>& right) const
	{
		return (*left)().value() < (*right)().value();
	};
};

template <class T>
struct TokenGreater
{
	__device__ bool operator()(const cuda_device_unique_ptr<IToken<T>>& left, const cuda_device_unique_ptr<IToken<T>>& right) const
	{
		return (*left)().value() > (*right)().value();
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

	__device__ virtual return_wrapper_t<T> operator()() const
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
