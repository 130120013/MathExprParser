#include <stack>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <list>
#include <algorithm>
#include <memory>
#include <map>
#include <list>

#ifndef PARSER_H
#define PARSER_H

enum class TokenType
{
	operatorPlus,
	operatorMinus,
	operatorMul,
	operatorDiv,
	operatorPow,
	sinFunction,
	cosFunction,
	tgFunction,
	function,
	bracket,
	Operator,
	number,
	variable,
};

enum class CudaParserErrorCodes
{
	Success,
	NotReady,
	UnexpectedCall,
	InsufficientNumberParams,
	UnexpectedToken,
	ParameterIsNotUnique,
	InvalidArgument,
	ParameterIsNotFound,
	InvalidExpression
};

template <class T>
struct return_wrapper_t
{
	 template <class U, class = std::enable_if_t<std::is_constructible<T, U&&>::value>>
	__device__ return_wrapper_t(U&& value, CudaParserErrorCodes exit_code = CudaParserErrorCodes::Success):m_code(exit_code)
	{
		m_pVal = this->get_buf_ptr();
		new (m_pVal) T(std::forward<U>(value));
	}
	__device__ return_wrapper_t(const return_wrapper_t& rw): m_code(rw.m_code)
	{
		m_pVal = rw.get_buf_ptr();
		memcpy(val_buf, rw.val_buf, sizeof(T));
	}
	__device__ const return_wrapper_t& operator= (const return_wrapper_t& rw)
	{
		this->m_code = rw.m_code;
		this->m_pVal = rw.get_buf_ptr();
		memcpy(val_buf, rw.val_buf, sizeof(T));
		return *this;
	}
	__device__ explicit return_wrapper_t(CudaParserErrorCodes exit_code):m_pVal(nullptr), m_code(exit_code) {}
	__device__ ~return_wrapper_t()
	{
		if (m_pVal)
			destroy_val(m_pVal);
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
	__device__ CudaParserErrorCodes return_code() const
	{
		return m_code;
	}
private:
	CudaParserErrorCodes m_code = CudaParserErrorCodes::Success;
	T* m_pVal = nullptr;
	alignas(T) char val_buf[sizeof(T)];
	__device__ T* get_buf_ptr()
	{
		return reinterpret_cast<T*>(val_buf);
	}
	__device__ const T* get_buf_ptr() const
	{
		return reinterpret_cast<const T*>(val_buf);
	}
	template <class U = T, class = std::enable_if_t<std::is_destructible<U>::value>>
	__device__ static void destroy_val(U* pVal)
	{
		pVal->~T();
	}
	//template <class U = T, class = std::enable_if_t<!std::is_destructible<U>::value>>
	//static void destroy_val(U*) {}
};

template <>
struct return_wrapper_t<void>
{
	__device__ return_wrapper_t() = default;
	__device__ explicit return_wrapper_t(CudaParserErrorCodes exit_code):m_code(exit_code) {}
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
private:
	CudaParserErrorCodes m_code = CudaParserErrorCodes::Success;
};

template <class T>
class IToken
{
public:
	__device__ virtual return_wrapper_t<T> operator()() const = 0; //All derived classes must implement the same method with the same return type
	/*Push a required (as defined at runtime, perhaps erroneously which must be detected by implementations) number of arguments.
	Push arguments from first to last. Then call the operator() above.
	*/
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value) = 0;
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const = 0;
	__device__ virtual std::size_t get_params_count() const = 0;
	__device__ virtual bool is_ready() const = 0; //all parameters are specified
	__device__ virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
	__device__ virtual TokenType type() = 0;
	__device__ virtual short getPriority() = 0;
};

template <class T>
class Number : public IToken<T>
{
public:
	__device__ Number(T val) : value(val) {};
	__device__ Number(const Number<T>& num) = default;

	__device__ virtual return_wrapper_t<T> operator()() const
	{
		return return_wrapper_t<T>(value);
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*this));
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ T operator+(const Number<T>& num) const
	{
		return this->value + num();
	}
	__device__ T operator-(const Number<T>& num) const
	{
		return this->value - num();
	}
	__device__ T operator*(const Number<T>& num) const
	{
		return this->value * num();
	}
	__device__ T operator/(const Number<T>& num) const
	{
		return this->value / num();
	}
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 0;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::number;
	}
	__device__ virtual short getPriority()
	{
		return -2;
	}
protected:
private:
	T value;
};

template <class T>
class Header;

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{
	const T* m_pValue = nullptr;
	std::unique_ptr<char[]> name;
	std::size_t name_length = 0;
	//bool isReady;
public:
	__device__ Variable(const Header<T>& header, const char* varname, std::size_t len)
		:m_pValue(&header.get_argument(varname, len)), name_length(len)
	{
		this->name = std::make_unique<char[]>(len + 1);
		std::strncpy(this->name.get(), varname, len);
		this->name[len] = 0;
	}
	__device__ Variable(Variable<T>&& val) = default;
	__device__ Variable(const Variable<T>& val)
	{
		*this = val;
	}
	__device__ Variable& operator=(Variable<T>&& val) = default;
	__device__ Variable& operator=(const Variable<T>& val)
	{
		m_pValue = val.m_pValue;
		name_length = val.name_length;
		this->name = std::make_unique<char[]>(val.name_length + 1);
		std::strncpy(this->name.get(), val.name.get(), val.name_length);
		this->name[val.name_length] = 0;
		return *this;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Variable<T>>(*this));
	}

	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		return return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 0;
	}
	__device__ virtual T operator()() const
	{
		return *m_pValue;
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	/*char* get_name() const
	{
		return name.get();
	}*/
	/*size_t get_name_length()
	{
		return name_length;
	}*/
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
	__device__ virtual short getPriority()
	{
		return 0; //default priority, less code but more error prone
	}
	__device__ virtual TokenType type()
	{
		return TokenType::Operator;
	}
};

template <class T>
class OperatorPlus : public Operator<T> //+-*/
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			//throw std::exception("Insufficient number are given for the plus operator.");
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		//return (*ops[0])() + (*ops[1])();
		return return_wrapper_t<T>(*(*ops[0])().get() + *(*ops[1])().get());
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if ((op0.get())->get()->type() == TokenType::number && (op0.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() + *((op0.get())->get())()));
		auto op_new = std::make_shared<OperatorPlus<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
	}
	__device__ virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorPlus;
	}
};
template <class T>
class OperatorMinus : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
		//	throw std::exception("Insufficient number are given for the plus operator.");
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return return_wrapper_t<T>(*(*ops[0])().get() - *(*ops[1])().get());
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorMinus;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if ((op0.get())->get()->type()  == TokenType::number && (op0.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() - *((op0.get())->get())()));
		auto op_new = std::make_shared<OperatorMinus<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
	}
};
template <class T>
class OperatorMul : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return return_wrapper_t<T>(*(*ops[0])().get() * *(*ops[1])().get());
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual short getPriority()
	{
		return 1;
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorMul;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if ((op0.get())->get()->type() == TokenType::number && (op1.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() * *((op0.get())->get())()));
		auto op_new = std::make_shared<OperatorMul<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return return_wrapper_t<T>(*(*ops[0])().get() / *(*ops[1])().get());
	}
	__device__ virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 1;
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorDiv;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if ((op0.get())->get()->type() == TokenType::number && (op0.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() / *((op0.get())->get())()));
		auto op_new = std::make_shared<OperatorDiv<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
	}
};
template <class T>
class OperatorPow : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return return_wrapper_t<T>(std::pow(*(*ops[0])().get(), *(*ops[1])().get()));
	}
	__device__ virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 2;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::operatorPow;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if ((op0.get())->get()->type() == TokenType::number && (op0.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(*((op0.get())->get())() / *((op0.get())->get())()));
		auto op_new = std::make_shared<OperatorPlus<T>>();
		op_new->push_argument(std::move(*(op0.get())));
		op_new->push_argument(std::move(*(op1.get())));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(op_new);
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
	std::list<std::shared_ptr<IToken<T>>> m_parameters;
	char* function_name;
public:
	__device__ virtual return_wrapper_t<T> operator()()
	{
		return return_wrapper_t<T>(*(m_parameters.front().get()->operator()().get()));
	}
	__device__ virtual bool is_ready() const
	{
		return true;
	}
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		m_parameters.push_back(value);
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	__device__ virtual const char* get_function_name() const
	{
		return function_name;
	}
	__device__ virtual TokenType type()
	{
		return TokenType::function;
	}
	__device__ virtual short getPriority()
	{
		return 1;
	}
protected:
	__device__ std::list<T>& parameter_queue()
	{
		return m_parameters;
	}
	__device__ const std::list<std::shared_ptr<IToken<T>>>& parameter_queue() const
	{
		return m_parameters;
	}
};

template <class T>
class SinFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return return_wrapper_t<T>(std::sin(*(op.get())().get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "sin";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::sinFunction;
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
		auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(std::sin((newarg.get()).get()->operator().get())));
			//return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = std::make_shared<SinFunction<T>>();
		pNewTkn->op = std::move(*(newarg.get()));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(pNewTkn);
	}
};
template <class T>
class CosFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	__device__ virtual  return_wrapper_t<T>  operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return  return_wrapper_t<T>(std::cos((*(op.get())()).get()));
			//std::cos((*op)());
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "cos";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::cosFunction;
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
				auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(std::cos((newarg.get()).get()->operator().get())));
			//return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = std::make_shared<SinFunction<T>>();
		pNewTkn->op = std::move(*(newarg.get()));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(pNewTkn);
	}
};
template <class T>
class TgFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	__device__ virtual return_wrapper_t<T> operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::InsufficientNumberParams);

		return  return_wrapper_t<T>(std::tan((*(op.get())()).get()));
	}
	__device__ virtual bool is_ready() const
	{
		return op->is_ready();
	}
	__device__ virtual short getPriority()
	{
		return 2;
	}
	__device__ virtual std::size_t get_params_count() const
	{
		return 1;
	}
	__device__ virtual const char* get_function_name() const
	{
		return "tg";
	}
	__device__ virtual TokenType type()
	{
		return TokenType::tgFunction;
	}
	__device__ virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			return return_wrapper_t<T>(CudaParserErrorCodes::NotReady);
				auto newarg = op->simplify();
		if ((newarg.get())->get()->type() == TokenType::number)
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(std::make_shared<Number<T>>(std::tan((newarg.get()).get()->operator().get())));
			//return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = std::make_shared<SinFunction<T>>();
		pNewTkn->op = std::move(*(newarg.get()));
		return return_wrapper_t<std::shared_ptr<IToken<T>>>(pNewTkn);
	}
};

template <class T>
class Bracket : public Operator<T> //,' '()
{
public:
	__device__ Bracket() = default;

	__device__ virtual return_wrapper_t<T> operator()() const
	{
		return true;
	}

	__device__ virtual bool is_ready() const
	{
		return true;
	}

	__device__ virtual return_wrapper_t<void> push_argument(const std::shared_ptr<IToken<T>>& value)
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
	__device__ virtual return_wrapper_t<std::shared_ptr<IToken<T>>> simplify() const
	{
		//return std::make_shared<Bracket<T>>(nullptr);
			return return_wrapper_t<std::shared_ptr<IToken<T>>>(CudaParserErrorCodes::UnexpectedCall);
	}
	__device__ std::size_t get_params_count() const
	{
		return 0;
	}
};

template <class T>
class Header
{
	std::map<std::string, T> m_arguments;
	std::vector<std::string> m_parameters;
	//std::unique_ptr<char[]> function_name;
	//std::size_t function_name_length = 0;
	std::string function_name;
	return_wrapper_t<void> construction_success_code;
public:
	__device__ Header() = default;
	__device__ Header(const char* expression, std::size_t expression_len, char** endPtr)
	{
		char* begPtr = (char*)(expression);
		std::list<std::string> params;

		bool isOpeningBracket = false;
		bool isClosingBracket = false;
		unsigned short commaCount = 0;

		auto isalpha = [](char ch) -> bool {return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');};
		auto isalnum = [&isalpha](char ch) -> bool {return ch >= '0' && ch <= '9' || isalpha(ch);};

		while (*begPtr != '=' && begPtr < expression + expression_len)
		{
			if (isalpha(*begPtr))
			{
				auto l_endptr = begPtr + 1;
				for (; isalnum(*l_endptr); ++l_endptr);
				if (this->function_name.empty())
					this->function_name = std::string(begPtr, l_endptr);
				else
				{
					if (!isOpeningBracket)
						//throw std::invalid_argument("Unexpected token"); //parameter outside of parenthesis
					{
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidArgument);
						return;
					}
					auto param_name = std::string(begPtr, l_endptr);
					if (!m_arguments.emplace(param_name, T()).second)
					{
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotFound);
						return;
					}
						//throw std::invalid_argument("Parameter " + param_name + " is not unique!"); //duplicated '('
					params.emplace_back(std::move(param_name));
				}
				begPtr = l_endptr;
			}

			if (*begPtr == ' ')
			{
				begPtr += 1;
				continue;
			}

			if (*begPtr == '(')
			{
				if (isOpeningBracket)
					//throw std::invalid_argument("ERROR!"); //duplicated '('
				{
					construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidArgument);
					return;
				}
				isOpeningBracket = true;
				begPtr += 1;
			}

			if (*begPtr == ',') //a-zA_Z0-9
			{
				commaCount += 1;
				begPtr += 1;
			}

			if (*begPtr == ')')
			{
				if (!isOpeningBracket)
				{
					//throw std::invalid_argument("ERROR!"); //missing ')'
					construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidArgument);
					return;
				}
				if (isClosingBracket)
				{
					//throw std::invalid_argument("ERROR!"); //dublicated ')'
					construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidArgument);
					return;
				}
				isClosingBracket = true;
				begPtr += 1;
			}
		}
		m_parameters.reserve(params.size());
		for (auto& param : params)
			m_parameters.emplace_back(std::move(param));
		*endPtr = begPtr;
	}
	__device__ Header(const Header<T>& val) /*: function_name_length(val.get_name_length())*/
	{
		std::size_t size = val.get_name_length();
		/*this->function_name = std::make_unique<char[]>(size + 1);
		std::strncpy(this->function_name.get(), val.get_function_name(), size);
		this->function_name[size] = 0;*/
		this->function_name = val.function_name;
		this->m_arguments = val.m_arguments;
		this->m_parameters = val.m_parameters;
		construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::Success);
	}
	__device__ virtual bool is_ready() const
	{
		return construction_success_code.return_code() == CudaParserErrorCodes::Success;
	}
	__device__ return_wrapper_t<void> push_argument(const char* name, std::size_t parameter_name_size, const T& value)
	{
		auto it = m_arguments.find(std::string(name, name + parameter_name_size));
		if (it == m_arguments.end())
		{
			construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::ParameterIsNotFound);
			return construction_success_code;
		}
		it->second = value;
	}
	__device__ const return_wrapper_t<T&> get_argument(const char* parameter_name, std::size_t parameter_name_size) const //call this from Variable::operator()().
	{
		auto it = m_arguments.find(std::string(parameter_name, parameter_name + parameter_name_size));
		if (it == m_arguments.end())
			return return_wrapper_t<T&>(CudaParserErrorCodes::ParameterIsNotFound);
		return it->second;
	}
	__device__ T& get_argument(const char* parameter_name, std::size_t parameter_name_size) //call this from Variable::operator()().
	{
		return const_cast<T&>(const_cast<const Header<T>*>(this)->get_argument(parameter_name, parameter_name_size));
	}
	__device__ const T& get_argument_by_index(std::size_t index) const //call this from Variable::operator()().
	{
		return this->get_argument(m_parameters[index].c_str(), m_parameters[index].size());
	}
	__device__ T& get_argument_by_index(std::size_t index) //call this from Variable::operator()().
	{
		return this->get_argument(m_parameters[index].c_str(), m_parameters[index].size());
	}
	__device__ std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	__device__ const char* get_function_name() const
	{
		return function_name.c_str();
	}
	__device__ size_t get_name_length() const
	{
		return function_name.size();
	}
	/*const std::vector<std::string>& get_params_vector() const
	{
		return m_parameters;
	}*/
	__device__ return_wrapper_t<std::size_t> get_param_index(const std::string& param_name)
	{
		for (std::size_t i = 0; i < this->m_parameters.size(); ++i)
		{
			if (this->m_parameters[i] == param_name)
				return return_wrapper_t<std::size_t>(i);
		}
		return return_wrapper_t<std::size_t>(CudaParserErrorCodes::ParameterIsNotFound);
	}
	__device__ Header(Header&&) = default;
	__device__ Header& operator=(Header&&) = default;
};

template <class T>
class Mathexpr
{
public:
	__device__ Mathexpr(const char* sMathExpr, std::size_t cbMathExpr);
	__device__ Mathexpr(const char* szMathExpr):Mathexpr(szMathExpr, std::strlen(szMathExpr)) {}
	template <class Traits, class Alloc>
	__device__ Mathexpr(const std::basic_string<char, Traits, Alloc>& strMathExpr):Mathexpr(strMathExpr.c_str(), strMathExpr.size()) {}
	__device__ return_wrapper_t<T> compute() const
	{
		auto result = body;
		simplify_body(result);
		if (result.size() != 1)
			return return_wrapper_t<T>(CudaParserErrorCodes::InvalidExpression);
		return return_wrapper_t<T>((*result.front())().get());
	}
	__device__ return_wrapper_t<void> init_variables(const std::vector<T>& parameters)
	{
		if (parameters.size() < header.get_params_count())
			return return_wrapper_t<void>(CudaParserErrorCodes::InsufficientNumberParams);

		for (std::size_t iArg = 0; iArg < header.get_params_count(); ++iArg)
			header.get_argument_by_index(iArg) = parameters[iArg];
	}
	//void clear_variables(); With the map referencing approach this method is not necessary anymore because if we need to reuse the expression
	//with different arguments, we just reassign them with init_variables
private:
	Header<T> header;
	std::list<std::shared_ptr<IToken<T>>> body;
};

__device__ inline void skipSpaces(char* input_string, std::size_t length)
{
	char* endTokPtr = (char*)(input_string + length);
	while ((*input_string == '=' || *input_string == '\t' || *input_string == ' ' || *input_string == '\0') && input_string < endTokPtr)
	{
		input_string += 1;
	}
}

template <class T>
__device__ std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr)
{
	if (*input_string >= '0' && *input_string <= '9')
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
	if (*input_string == '+')
	{
		char* tok = (char*)input_string;
		while (*tok != NULL && *tok == ' ')
			tok += 1;

		if (*tok == '-' || *tok == '+')
			return std::make_shared<OperatorPlus<T>>();
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
	}

	if (*input_string == '-')
	{
		char* tok = (char*)input_string;
		while (*tok != NULL && *tok == ' ')
			tok += 1;

		if (*tok == '-' || *tok == '+')
			return std::make_shared<OperatorMinus<T>>();
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
	}

	if (*input_string == '*')
		return std::make_shared<OperatorMul<T>>();
	if (*input_string == '/')
		return std::make_shared<OperatorDiv<T>>();
	if (*input_string == '^')
		return std::make_shared<OperatorPow<T>>();

	auto iswhitespace = [](char ch) -> bool {return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\0';}; //see also std::isspace
	if (std::strncmp(input_string, "sin", 3) == 0 && !iswhitespace(input_string[3]))
	{
		*endptr = (char*) input_string + 3;
		return std::make_shared<SinFunction<T>>();
	}
	else if (std::strncmp(input_string, "cos", 3) == 0 && !iswhitespace(input_string[3]))
	{
		*endptr = (char*) input_string + 3;
		return std::make_shared<CosFunction<T>>();
	}
	else if (std::strncmp(input_string, "tg", 2) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*) input_string + 2;
		return std::make_shared<TgFunction<T>>();
	}
	
	return nullptr;
}

template <class T>
__device__ std::shared_ptr<Variable<T>> parse_text_token(const Header<T>& header, const char* input_string, char** endptr)
{
	std::size_t name_size;
	char* endTokPtr = (char*)input_string;
	while ((*endTokPtr >= 'A' && *endTokPtr <= 'Z') || (*endTokPtr >= 'a' && *endTokPtr <= 'z') || (*endTokPtr >= '0' && *endTokPtr <= '9'))
	{
		endTokPtr += 1;
	}

	name_size = endTokPtr - input_string;
	std::string name(input_string, input_string + name_size);
	char* tok_name = new char[name_size + 1];
	tok_name = (char*)(name.c_str());
	auto token = std::make_shared<Variable<T>>(header, tok_name, name_size);
	*(endptr) = endTokPtr;
	return token;
}

template <class T>
__device__ std::list<std::shared_ptr<IToken<T>>> lexBody(const Header<T>& header, const char* expr, std::size_t length)
{
	std::list<std::shared_ptr<IToken<T>>> output;
	int prior;

	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto delta = (char*)expr;
	//bool isBinary = true;
	//short hasPunct = 0;
	std::stack<std::shared_ptr<IToken<T>>> operationQueue;

	while (*begPtr != '\0' && begPtr != expr + length)
	{
		delta = begPtr;
		try
		{
			if (*begPtr >= '0' && *begPtr <= '9')
			{
				output.push_back(parse_token<T>(begPtr, &endPtr));

				if (begPtr == endPtr)
					throw std::invalid_argument("Cannot parse: " + *begPtr);
				begPtr = endPtr;
			}
			else
			{
				if (*begPtr == ' ')
				{
					begPtr += 1;
					continue;
				}
				if ((*begPtr >= 'A' && *begPtr <= 'Z') || (*begPtr >= 'a' && *begPtr <= 'z')) //if not sin, cos, tg, it is variable
				{
					/* In your former approach you checked the first letter, and if it was 's' or 'c' or 't' you assumed it was a sine, a cosine, 
					or a tangent function. But, if the token wasn't actually a function, but, e.g. a parameter 's', the parse_token function would
					return nullptr, and the lexBody would throw an exception in this case, which is not correct.
					*/
					auto func = parse_token<T>(begPtr, &endPtr);
					if (func != nullptr)
						operationQueue.push(func);
					else
						//If the parameter is not found, parse_text_token will throw via Variable c-tor, therefore there's no need for pre-check
						//nor to create the token (var) twice.
						/*auto var = parse_text_token<T>(begPtr, &endPtr);
						std::string var_name(var->get_name(), var->get_name_length());
						if (find(m_parameters.begin(), m_parameters.end(), var_name) == m_parameters.end())
							throw std::invalid_argument("Parameter is not found: " + var_name);*/
						output.push_back(parse_text_token<T>(header, begPtr, &endPtr));
					begPtr = endPtr;
					continue;
				}
				if (*begPtr == '+')
				{
					skipSpaces(begPtr + 1, expr + length - begPtr - 1);
					char tok = *(begPtr + 1);
					if (*begPtr == '+' && (tok == '+' || tok == '-')) //unary +
					{
						output.push_back(parse_token<T>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary +
					{
						while (operationQueue.size() != 0)
						{
							//auto plus = dynamic_cast<Operator<T>*>(operationQueue.top().get());
							auto plus = operationQueue.top().get();
							if (plus->type() > TokenType::function)
							{
							//	auto plus1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
							//	if (plus1 == nullptr)
									throw std::invalid_argument("Unexpected error at " + *begPtr);
							//	else
							//		prior = plus1->getPriority();
							}
							else
							{
								prior = plus->getPriority();
							}
							if (OperatorPlus<T>().getPriority() <= prior)
							{
								output.push_back(operationQueue.top());
								operationQueue.pop();
							}
							else
								break;
						}
						operationQueue.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}
				}else if (*begPtr == '-')
				{
					skipSpaces(begPtr + 1, expr + length - begPtr - 1);
					char tok = *(begPtr + 1);
					if (*begPtr == '-' && (tok == '+' || tok == '-')) //unary -
					{
						output.push_back(parse_token<T>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary -
					{
						while (operationQueue.size() != 0)
						{
							auto minus = operationQueue.top().get();
							if (minus->type() > TokenType::function)
							{
								throw std::invalid_argument("Unexpected error at " + *begPtr);
							}
							else
							{
								prior = minus->getPriority();
							}
							if (OperatorMinus<T>().getPriority() <= prior)
							{
								output.push_back(operationQueue.top());
								operationQueue.pop();
							}
							else
								break;
						}
						operationQueue.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}
				}else if (*begPtr == '*')
				{
					while (operationQueue.size() != 0)
					{
						auto mul = operationQueue.top().get();
						if (mul->type() > TokenType::function)
						{
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						}
						else
						{
							prior = mul->getPriority();
						}
						if (OperatorMul<T>().getPriority() <= prior)
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						else
							break;
					}
					operationQueue.push(parse_token<T>(begPtr, &endPtr));
					begPtr += 1;
				}else if (*begPtr == '/')
				{
					while (operationQueue.size() != 0)
					{
						auto div = operationQueue.top().get();
						if (div->type() > TokenType::function)
						{
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						}
						else
						{
							prior = div->getPriority();
						}
						if (OperatorDiv<T>().getPriority() <= prior)
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						else
							break;
					}
					operationQueue.push(parse_token<T>(begPtr, &endPtr));
					begPtr += 1;
				}else if (*begPtr == '^')
				{
					while (operationQueue.size() != 0)
					{
						auto pow = operationQueue.top().get();
						if (pow->type() > TokenType::function)
						{
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						}
						else
						{
							prior = pow->getPriority();
						}
						if (OperatorDiv<T>().getPriority() <= prior)
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						else
							break;
					}
					operationQueue.push(parse_token<T>(begPtr, &endPtr));
					begPtr += 1;
				}else if (*begPtr == ',')
				{
					bool isOpeningBracket = false;
					while (!isOpeningBracket || operationQueue.size() != 0) //while an opening bracket is not found or an operation stack is not empty
					{
						if (operationQueue.top().get()->type() != TokenType::bracket) //if the cast to Bracket is not successfull, return NULL => it is not '('  
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						else
						{
							isOpeningBracket = true;
						}
					}
					if (!isOpeningBracket) //missing '('
						throw std::invalid_argument("There is no opening bracket!");
					begPtr += 1;
				}
				if (*begPtr == '(')
				{
					operationQueue.push(std::make_shared<Bracket<T>>());
					begPtr += 1;
				}
				if (*begPtr == ')')
				{
					bool isOpeningBracket = false;
					while (operationQueue.size() != 0)
					{
						if (operationQueue.top().get()->type() != TokenType::bracket)
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						else
						{
							isOpeningBracket = true;
							break;
						}
					}
					if (!isOpeningBracket)
						throw std::invalid_argument("There is no opening bracket!");
					else
						operationQueue.pop();
					begPtr += 1;
				}
			}
		if (begPtr - delta == 0)
			throw std::invalid_argument("Invalid symbol at " + *begPtr);
		}
		catch (std::exception e)
		{
			throw std::invalid_argument("ERROR!");
		}
	}
	while (operationQueue.size() != 0)
	{
		if (operationQueue.top().get()->type() == TokenType::bracket) //checking enclosing brackets
			throw std::invalid_argument("Enclosing bracket!");
		else
		{
			output.push_back(operationQueue.top());
			operationQueue.pop();
		}
	}
	return output;
}

template <class K>
__device__ typename std::list<std::shared_ptr<IToken<K>>>::iterator simplify(std::list<std::shared_ptr<IToken<K>>>& body, typename std::list<std::shared_ptr<IToken<K>>>::iterator elem)
{
	try
	{
		//bool isComputable = false;
		auto paramsCount = elem->get()->get_params_count();
		auto param_it = elem;
		for (auto i = paramsCount; i > 0; --i)
		{
			--param_it;
			((*elem).get())->push_argument(*param_it); //here std::move must be
			param_it = body.erase(param_it);
		}
		if (elem->get()->is_ready())
			*elem = elem->get()->simplify();
		return ++elem;
	}
	catch (std::exception e)
	{
		throw std::invalid_argument("ERROR!");
	}
}

template <class T>
__device__ void simplify_body(std::list<std::shared_ptr<IToken<T>>>& body)
{
	auto it = body.begin();
	while (body.size() > 1 )
		it = simplify(body, it);
	//When everything goes right, you are left with only one element within the list - the root of the tree.
}

template <class T>
__device__ T compute(const std::list<std::shared_ptr<IToken<T>>>& body)
{
	assert(body.size() == 1);
	return body.front()();
}

template <class T>
__device__ Mathexpr<T>::Mathexpr(const char* sMathExpr, std::size_t cbMathExpr)
{
	const char* endptr;
	header = Header<T>(sMathExpr, cbMathExpr, (char**) &endptr);
	++endptr;
	body = lexBody<T>(header, endptr, cbMathExpr - (endptr - sMathExpr));
	simplify_body(body);
}

#endif // !PARSER_H