#include <stack>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <list>
#include <algorithm>
#include <queue> //to store arguments
#include <stdexcept> //for exceptions
#include <memory>
#include <map>
#include <list>
#include <limits>

#include <iterator>

#include <cassert>

#ifndef PARSER_H
#define PARSER_H

constexpr bool iswhitespace(char ch) noexcept
{
	return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\0'; //see also std::isspace
}

constexpr bool isdigit(char ch) noexcept
{
	return ch >= '0' && ch <= '9';
};

constexpr bool isalpha(char ch) noexcept
{
	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
};

constexpr bool isalnum(char ch) noexcept
{
	return isdigit(ch) || isalpha(ch);
}

template <class Iterator>
Iterator skipSpaces(Iterator pBegin, Iterator pEnd) noexcept
{
	auto pCurrent = pBegin;
	while (iswhitespace(*pCurrent) && ++pCurrent < pEnd);
	return pCurrent;
}

const char* skipSpaces(const char* input_string, std::size_t length) noexcept
{
	return skipSpaces(input_string, input_string + length);
}

const char* skipSpaces(const char* input_string) noexcept
{
	auto pCurrent = input_string;
	while (iswhitespace(*pCurrent) && *pCurrent++ != 0);
	return pCurrent;
}

inline char* skipSpaces(char* input_string, std::size_t length) noexcept
{
	return const_cast<char*>(skipSpaces(const_cast<const char*>(input_string), length));
}

inline char* skipSpaces(char* input_string) noexcept
{
	return const_cast<char*>(skipSpaces(const_cast<const char*>(input_string)));
}

struct token_string_entity
{
	token_string_entity() = default;
	token_string_entity(const char* start_ptr, const char* end_ptr):m_pStart(start_ptr), m_pEnd(end_ptr) {}

	//return a negative if the token < pString, 0 - if token == pString, a positive if token > pString
	int compare(const char* pString, std::size_t cbString) const noexcept
	{
		auto my_size = this->size();
		auto min_length = my_size < cbString?my_size:cbString;
		for (std::size_t i = 0; i < min_length; ++i)
		{
			if (m_pStart[i] < pString[i])
				return -1;
			if (m_pStart[i] > pString[i])
				return -1;
		}
		return my_size < cbString?-1:(my_size > cbString?1:0);
	}
	inline int compare(const token_string_entity& right) const noexcept
	{
		this->compare(right.begin(), right.size());
	}
	inline int compare(const char* pszString) const noexcept
	{
		this->compare(pszString, std::strlen(pszString));
	}
	template <std::size_t N>
	inline auto compare(const char(&strArray)[N]) const noexcept -> std::enable_if_t<N == 0, int>
	{
		return this->size() == 0;
	}
	template <std::size_t N>
	inline auto compare(const char(&strArray)[N]) const noexcept -> std::enable_if_t<(N > 0), int>
	{
		return strArray[N-1] == '\0'?
			compare(strArray, N - 1)://null terminated string, like "sin"
			compare(strArray, N); //generic array, like const char Array[] = {'s', 'i', 'n'}
	}
	inline const char* begin() const noexcept
	{
		return m_pStart;
	}
	inline const char* end() const noexcept
	{
		return m_pEnd;
	}
	inline std::size_t size() const noexcept
	{
		return std::size_t(this->end() - this->begin());
	}
private:
	const char* m_pStart = nullptr, *m_pEnd = nullptr;
};

bool operator==(const token_string_entity& left, const token_string_entity& right) noexcept
{
	return left.compare(right) == 0;
}

bool operator==(const token_string_entity& left, const char* pszRight) noexcept
{
	return left.compare(pszRight) == 0;
}

template <std::size_t N>
bool operator==(const token_string_entity& left, const char (&strArray)[N]) noexcept
{
	return left.compare(strArray) == 0;
}

bool operator!=(const token_string_entity& left, const token_string_entity& right) noexcept
{
	return left.compare(right) != 0;
}

bool operator!=(const token_string_entity& left, const char* pszRight) noexcept
{
	return left.compare(pszRight) != 0;
}

template <std::size_t N>
bool operator!=(const token_string_entity& left, const char (&strArray)[N]) noexcept
{
	return left.compare(strArray) != 0;
}


bool operator==(const char* pszLeft, const token_string_entity& right) noexcept
{
	return right == pszLeft;
}

template <std::size_t N>
bool operator==(const char (&strArray)[N], const token_string_entity& right) noexcept
{
	return right == strArray;
}

bool operator!=(const char* pszRight, const token_string_entity& right) noexcept
{
	return right != pszRight;
}

template <std::size_t N>
bool operator!=(const char (&strArray)[N], const token_string_entity& right) noexcept
{
	return right != strArray;
}

//let's define token as a word.
//One operator: = + - * / ^ ( ) ,
//Or: something that only consists of digits and one comma
//Or: something that starts with a letter and proceeds with letters or digits

token_string_entity parse_string_token(const char* pExpression, std::size_t cbExpression)
{
	auto pStart = skipSpaces(pExpression, cbExpression);
	if (pStart == pExpression + cbExpression)
		return token_string_entity();
	if (isdigit(*pStart) || *pStart == '.')
	{
		bool fFlPointFound = *pStart == '.';
		const char* pEnd = pStart;
		while (isdigit(*++pEnd) || (!fFlPointFound && *pEnd == '.')) continue;
		return token_string_entity(pStart, pEnd);
	}
	if (isalpha(*pStart))
	{
		const char* pEnd = pStart;
		while (isalnum(*++pEnd)) continue;
		return token_string_entity(pStart, pEnd);
	}
	return token_string_entity(pStart, pStart + 1);
	/* check if necessary:
	switch (*pStart)
	{
	case '=':
	case '+':
	case '-':
	case '*':
	case '/':
	case '^':
	case '(':
	case ')':
	case ',':
		return token_string_entity(pStart, pStart + 1);
	default:
		
	};*/
}

struct invalid_exception :std::exception
{
	invalid_exception() = default;
	invalid_exception(const char* decription);
};

struct bad_expession_parameters :std::exception
{
	bad_expession_parameters() = default;
	bad_expession_parameters(const char* decription);
};

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
	function,
	bracket,
	Operator,
	number,
	variable
};

template <class T>
class IToken
{
public:
	virtual T operator()() const = 0; //All derived classes must implement the same method with the same return type
	/*Push a required (as defined at runtime, perhaps erroneously which must be detected by implementations) number of arguments.
	Push arguments from first to last. Then call the operator() above.
	*/
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value) = 0;
	virtual std::shared_ptr<IToken<T>> simplify() const = 0;
	virtual std::size_t get_params_count() const = 0;
	virtual bool is_ready() const = 0; //all parameters are specified
	virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
	virtual TokenType type() = 0;
	virtual short getPriority() = 0;
};

template <class T>
class Number : public IToken<T>
{
public:
	Number(T val) : value(val) {};
	Number(const Number<T>& num) = default;

	virtual T operator()() const
	{
		return value;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		return std::make_shared<Number<T>>(*this);
	}
	virtual bool is_ready() const
	{
		return true;
	}
	T operator+(const Number<T>& num) const
	{
		return this->value + num();
	}
	T operator-(const Number<T>& num) const
	{
		return this->value - num();
	}
	T operator*(const Number<T>& num) const
	{
		return this->value * num();
	}
	T operator/(const Number<T>& num) const
	{
		return this->value / num();
	}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		//do nothing for literals - they do not accept parameters. But the implementation (even empty) must be provided for polymorphic methods.
		//or throw an exception
#ifndef __CUDACC__
		throw std::invalid_argument("Unexpected call");
#endif //__CUDACC__
	}
	virtual std::size_t get_params_count() const
	{
		return 0;
	}
	virtual TokenType type()
	{
		return TokenType::number;
	}
	virtual short getPriority()
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
	Variable(const Header<T>& header, const char* varname, std::size_t len)
		:m_pValue(&header.get_argument(varname, len)), name_length(len)
	{
		this->name = std::make_unique<char[]>(len + 1);
		std::strncpy(this->name.get(), varname, len);
		this->name[len] = 0;
	}
	Variable(Variable<T>&& val) = default;
	Variable(const Variable<T>& val)
	{
		*this = val;
	}
	Variable& operator=(Variable<T>&& val) = default;
	Variable& operator=(const Variable<T>& val)
	{
		m_pValue = val.m_pValue;
		name_length = val.name_length;
		this->name = std::make_unique<char[]>(val.name_length + 1);
		std::strncpy(this->name.get(), val.name.get(), val.name_length);
		this->name[val.name_length] = 0;
		return *this;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		return std::make_shared<Variable<T>>(*this);
	}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
#ifndef __CUDACC__
		throw std::invalid_argument("Unexpected call");
#endif //__CUDACC__
	}
	virtual std::size_t get_params_count() const
	{
		return 0;
	}
	virtual T operator()() const
	{
		return *m_pValue;
	}
	virtual bool is_ready() const
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
	virtual TokenType type()
	{
		return TokenType::variable;
	}

	virtual short getPriority()
	{
		return -2;
	}
};

template <class T>
class Operator : public IToken<T>
{
//public:
//	virtual short getPriority()
//	{
//		return 0; //default priority, less code but more error prone
//	}
	/*virtual TokenType type()
	{
		return TokenType::Operator;
	}*/
};

template <class T>
class UnaryPlus : public Operator<T> //+-*/
{
	std::shared_ptr<IToken<T>> op;
	bool fOpSet = false;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		if (fOpSet)
			throw std::exception("Invalid arguments of an unary plus operator.");
		op = value;
		fOpSet = true;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of an unary plus operator.");

		return (*op)();
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!this->is_ready())
			throw std::exception("Not ready to simplify an operator");
		return op->simplify(); //unary + does no do anything
	}
	virtual bool is_ready() const
	{
		return fOpSet && op->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual TokenType type()
	{
		return TokenType::UnaryPlus;
	}
	virtual short getPriority()
	{
		return 2;
	}
};

template <class T>
class UnaryMinus : public Operator<T> //+-*/
{
	std::shared_ptr<IToken<T>> op;
	bool fOpSet = false;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		if (fOpSet)
			throw std::exception("Invalid arguments of an unary plus operator.");
		op = value;
		fOpSet = true;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of an unary plus operator.");

		return (*ops)();
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!this->is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = op->simplify();

		if (op0->type() == TokenType::number)
			return std::make_shared<Number<T>>(-(*dynamic_cast<Number<T>*>(op0.get()))());
		auto op_new = std::make_shared<UnaryMinus<T>>(*this);
		op_new->push_argument(std::move(op0));
		return op_new;
	}
	virtual bool is_ready() const
	{
		return fOpSet && op->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual TokenType type()
	{
		return TokenType::UnaryMinus;
	}
	virtual short getPriority()
	{
		return 2;
	}
};

template <class T>
class BinaryPlus : public Operator<T> //+-*/
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Invalid arguments of a binary plus operator.");

		return (*ops[0])() + (*ops[1])();
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op0)() + (*op1)());
		auto op_new = std::make_shared<BinaryPlus<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual TokenType type()
	{
		return TokenType::BinaryPlus;
	}
};
template <class T>
class BinaryMinus : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops + 1;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top-- = value;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Invalid arguments of a binary minus operator.");

		return (*ops[0])() - (*ops[1])();
	}
	virtual bool is_ready() const
	{
		return true;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual TokenType type()
	{
		return TokenType::BinaryMinus;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op0)() - (*op1)());
		auto op_new = std::make_shared<BinaryMinus<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class OperatorMul : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Invalid arguments of a multiplication operator.");

		return (*ops[0])() * (*ops[1])();
	}
	virtual bool is_ready() const
	{
		return true;
	}
	virtual short getPriority()
	{
		return 1;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual TokenType type()
	{
		return TokenType::operatorMul;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op0)() * (*op1)());
		auto op_new = std::make_shared<OperatorMul<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops + 1;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top-- = value;
	}
	virtual T operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Invalid arguments of a division operator.");

		return (*ops[0])() / (*ops[1])();
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual short getPriority()
	{
		return 1;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual TokenType type()
	{
		return TokenType::operatorDiv;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op0)() / (*op1)());
		auto op_new = std::make_shared<OperatorDiv<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class OperatorPow : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Invalid arguments of a power operator.");

		return std::pow((*ops[0])(), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual TokenType type()
	{
		return TokenType::operatorPow;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::pow((*op0)(), (*op1)()));
		auto op_new = std::make_shared<OperatorPow<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
	/*std::list<std::shared_ptr<IToken<T>>> m_parameters;
	char* function_name;
public:
	virtual T operator()()
	{
		return m_parameters.front().get()->operator()();
	}
	virtual bool is_ready() const
	{
		return true;
	}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		m_parameters.push_back(value);
	}
	virtual std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	virtual const char* get_function_name() const
	{
		return function_name;
	}
	virtual TokenType type()
	{
		return TokenType::function;
	}
	virtual short getPriority()
	{
		return 1;
	}
protected:
	std::list<T>& parameter_queue()
	{
		return m_parameters;
	}
	const std::list<std::shared_ptr<IToken<T>>>& parameter_queue() const
	{
		return m_parameters;
	}*/
};

template <class T>
class SinFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::sin((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "sin";
	}
	virtual TokenType type()
	{
		return TokenType::sinFunction;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = std::make_shared<SinFunction<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class CosFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::cos((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "cos";
	}
	virtual TokenType type()
	{
		return TokenType::cosFunction;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::cos((*newarg)()));
		auto pNewTkn = std::make_shared<CosFunction<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class TgFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::tan((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "tg";
	}
	virtual TokenType type()
	{
		return TokenType::tgFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::tan((*newarg)()));
		auto pNewTkn = std::make_shared<TgFunction<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};

////////// not ready

template <class T>
class Log10Function : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::log10((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "log10";
	}
	virtual TokenType type()
	{
		return TokenType::log10Function;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::log10((*newarg)()));
		auto pNewTkn = std::make_shared<Log10Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class LnFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::log((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "ln";
	}
	virtual TokenType type()
	{
		return TokenType::lnFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::log((*newarg)()));
		auto pNewTkn = std::make_shared<LnFunction<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class LogFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return std::log((*ops[1])()) / std::log((*ops[0])());
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "log";
	}
	virtual TokenType type()
	{
		return TokenType::logFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::pow((*op0)(), (*op1)()));
		auto op_new = std::make_shared<OperatorPow<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class JnFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _jn(int((*ops[0])()), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "jn";
	}
	virtual TokenType type()
	{
		return TokenType::logFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(_jn(int((*ops[0])()), (*ops[1])()));
		auto op_new = std::make_shared<JnFunction<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class J0Function : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _j0((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "j0";
	}
	virtual TokenType type()
	{
		return TokenType::lnFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(_j0((*newarg)()));
		auto pNewTkn = std::make_shared<J0Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class J1Function : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _j1((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "j1";
	}
	virtual TokenType type()
	{
		return TokenType::lnFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(_j1((*newarg)()));
		auto pNewTkn = std::make_shared<J1Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class YnFunction : public Function<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!ops[0]->is_ready() || !ops[1]->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _yn(int((*ops[0])()), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "yn";
	}
	virtual TokenType type()
	{
		return TokenType::logFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(_yn(int((*ops[0])()), (*ops[1])()));
		auto op_new = std::make_shared<YnFunction<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
};
template <class T>
class Y0Function : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _y0((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "y0";
	}
	virtual TokenType type()
	{
		return TokenType::y0Function;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(_y0((*newarg)()));
		auto pNewTkn = std::make_shared<Y0Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};
template <class T>
class Y1Function : public Function<T>
{
	std::shared_ptr<IToken<T>> op;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		op = value;
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!op->is_ready())
			throw std::exception("Insufficient number are given for the plus operator.");

		return _y1((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "y1";
	}
	virtual TokenType type()
	{
		return TokenType::y1Function;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(_y1((*newarg)()));
		auto pNewTkn = std::make_shared<Y1Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};

//////////// not ready
template <class T>
class MaxFunction : public Function<T>
{
	std::vector<std::shared_ptr<IToken<T>>> ops;
public:
	short paramsCount;
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_back(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			if (!op->is_ready())
				throw std::exception("Insufficient number are given for the plus operator.");
		}

		return std::max_element(ops.begin(), ops.end()); //нужен компаратор
	}
	virtual bool is_ready() const
	{
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			if (!op->is_ready())
				return false;
		}
		return true;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return paramsCount;
	}
	virtual const char* get_function_name() const
	{
		return "max";
	}
	virtual TokenType type()
	{
		return TokenType::maxFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			auto newarg = op->simplify();
			if (newarg->type() == TokenType::number)
				return std::make_shared<Number<T>>(_y1((*newarg)()));
			auto pNewTkn = std::make_shared<MaxFunction<T>>();
			pNewTkn->ops = std::move(newarg);
			return pNewTkn;
		}
	}
};
template <class T>
class MinFunction : public Function<T>
{
	std::vector<std::shared_ptr<IToken<T>>> ops;
public:
	short paramsCount;
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_back(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			if (!op->is_ready())
				throw std::exception("Insufficient number are given for the plus operator.");
		}

		return std::min_element(ops.begin(), ops.end()); //нужен компаратор
	}
	virtual bool is_ready() const
	{
		for (auto op = ops.begin(); op != ops.end(); ++op)
		{
			if (!op->is_ready())
				return false;
		}
		return true;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::size_t get_params_count() const
	{
		return paramsCount;
	}
	virtual const char* get_function_name() const
	{
		return "min";
	}
	virtual TokenType type()
	{
		return TokenType::minFunction;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = op->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(_y1((*newarg)()));
		auto pNewTkn = std::make_shared<Y1Function<T>>();
		pNewTkn->op = std::move(newarg);
		return pNewTkn;
	}
};

template <class T>
class Bracket : public Operator<T> //,' '()
{
public:
	Bracket() = default;

	virtual T operator()() const
	{
		return true;
	}

	virtual bool is_ready() const
	{
		return true;
	}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		return; //openingBracket = value; //true is for opening bracket, false is for closing.
	}
	virtual short getPriority()
	{
		return -1;
	}
	virtual TokenType type()
	{
		return TokenType::bracket; 
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		//return std::make_shared<Bracket<T>>(nullptr);
		throw std::exception("Unexpected call");
	}
	std::size_t get_params_count() const
	{
		return 0;
	}
};

template <class T>
class TokenStorage
{
	std::stack<std::shared_ptr<IToken<T>>> operationStack;
	std::list<std::shared_ptr<IToken<T>>> outputList;

public:
	void push_operator(const std::shared_ptr<IToken<T>>& op)
	{
		//not ready
		auto my_priority = op->getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(op);
		pLastToken = operationStack.top().get();
	}

	void push_operator(std::shared_ptr<IToken<T>>&& op)
	{
		//not ready
		auto my_priority = op->getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(std::move(op));
		pLastToken = operationStack.top().get();
	}

		void push_operator(std::shared_ptr<IToken<T>>&& op)
	{
		//not ready
		auto my_priority = op->getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(std::move(op));
		pLastToken = operationStack.top().get();
	}
	void push_operator(std::shared_ptr<IToken<T>>&& op)
	{
		//not ready
		auto my_priority = op->getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(std::move(op));
		pLastToken = operationStack.top().get();
	}
	void push_operator(std::shared_ptr<IToken<T>>&& op)
	{
		//not ready
		auto my_priority = op->getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(std::move(op));
		pLastToken = operationStack.top().get();
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
	bool isReady = false;
public:
	Header() = default;
	Header(const char* expression, std::size_t expression_len, char** endPtr)
	{
		char* begPtr = (char*)(expression);
		std::list<std::string> params;

		bool isOpeningBracket = false;
		bool isClosingBracket = false;
		unsigned short commaCount = 0;

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
						throw std::invalid_argument("Unexpected token"); //parameter outside of parenthesis
					auto param_name = std::string(begPtr, l_endptr);
					if (!m_arguments.emplace(param_name, T()).second)
						throw std::invalid_argument("Parameter " + param_name + " is not unique!"); //duplicated '('
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
					throw std::invalid_argument("ERROR!"); //duplicated '('
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
					throw std::invalid_argument("ERROR!"); //missing ')'
				if (isClosingBracket)
					throw std::invalid_argument("ERROR!"); //dublicated ')'
				isClosingBracket = true;
				begPtr += 1;
			}
		}
		m_parameters.reserve(params.size());
		for (auto& param : params)
			m_parameters.emplace_back(std::move(param));
		*endPtr = begPtr;
	}
	Header(const Header<T>& val) /*: function_name_length(val.get_name_length())*/
	{
		std::size_t size = val.get_name_length();
		/*this->function_name = std::make_unique<char[]>(size + 1);
		std::strncpy(this->function_name.get(), val.get_function_name(), size);
		this->function_name[size] = 0;*/
		this->function_name = val.function_name;
		this->m_arguments = val.m_arguments;
		this->m_parameters = val.m_parameters;
		isReady = true;
	}
	virtual bool is_ready() const
	{
		return true;
	}
	void push_argument(const char* name, std::size_t parameter_name_size, const T& value)
	{
		auto it = m_arguments.find(std::string(name, name + parameter_name_size));
		if (it == m_arguments.end())
			throw std::invalid_argument("Parameter is not found");
		it->second = value;
	}
	const T& get_argument(const char* parameter_name, std::size_t parameter_name_size) const //call this from Variable::operator()().
	{
		auto it = m_arguments.find(std::string(parameter_name, parameter_name + parameter_name_size));
		if (it == m_arguments.end())
			throw std::invalid_argument("Parameter is not found");
		return it->second;
	}
	T& get_argument(const char* parameter_name, std::size_t parameter_name_size) //call this from Variable::operator()().
	{
		return const_cast<T&>(const_cast<const Header<T>*>(this)->get_argument(parameter_name, parameter_name_size));
	}
	const T& get_argument_by_index(std::size_t index) const //call this from Variable::operator()().
	{
		return this->get_argument(m_parameters[index].c_str(), m_parameters[index].size());
	}
	T& get_argument_by_index(std::size_t index) //call this from Variable::operator()().
	{
		return this->get_argument(m_parameters[index].c_str(), m_parameters[index].size());
	}
	std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	const char* get_function_name() const
	{
		return function_name.c_str();
	}
	size_t get_name_length() const
	{
		return function_name.size();
	}
	/*const std::vector<std::string>& get_params_vector() const
	{
		return m_parameters;
	}*/
	std::size_t get_param_index(const std::string& param_name)
	{
		for (std::size_t i = 0; i < this->m_parameters.size(); ++i)
		{
			if (this->m_parameters[i] == param_name)
				return i;
		}
		throw std::invalid_argument("Parameter not found");
	}
	Header(Header&&) = default;
	Header& operator=(Header&&) = default;
};

template <class T>
class Mathexpr
{
public:
	Mathexpr(const char* sMathExpr, std::size_t cbMathExpr);
	Mathexpr(const char* szMathExpr):Mathexpr(szMathExpr, std::strlen(szMathExpr)) {}
	template <class Traits, class Alloc>
	Mathexpr(const std::basic_string<char, Traits, Alloc>& strMathExpr):Mathexpr(strMathExpr.c_str(), strMathExpr.size()) {}
	T compute() const
	{
		auto result = body;
		simplify_body(result);
		if (result.size() != 1)
			throw std::exception("Invalid expression");
		return (*result.front())();
	}
	void init_variables(const std::vector<T>& parameters)
	{
		if (parameters.size() < header.get_params_count())
			throw std::invalid_argument("Count of arguments < " + header.get_params_count());
		for (std::size_t iArg = 0; iArg < header.get_params_count(); ++iArg)
			header.get_argument_by_index(iArg) = parameters[iArg];
	}
	//void clear_variables(); With the map referencing approach this method is not necessary anymore because if we need to reuse the expression
	//with different arguments, we just reassign them with init_variables
private:
	Header<T> header;
	std::list<std::shared_ptr<IToken<T>>> body;

	template <class T>
	void lexBody(const char* expr, std::size_t length)
	{
		int prior;
		char* begPtr = expr;
		std::size_t cbRest = length;
		auto delta;
		short paramCount = 1;
		std::stack<std::shared_ptr<IToken<T>>> operationStack;
		IToken<T> *pLastToken = nullptr;

		while (cbRest > 0)
		{
			delta = begPtr;
			auto tkn = parse_string_token(begPtr, cbRest);
			if (tkn == "+")
			{
				if (pLastToken == nullptr 
					|| pLastToken->type() == TokenType::BinaryPlus || pLastToken->type() == TokenType::BinaryMinus
					|| pLastToken->type() == TokenType::operatorMul || pLastToken->type() == TokenType::operatorDiv
					|| pLastToken->type() == TokenType::operatorPow) //unary form
				{
					auto pMyTkn = std::make_shared<UnaryPlus<T>>();
					auto my_priority = pMyTkn->getPriority();
					while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
					{
						body.push_back(operationStack.top());
						operationStack.pop();
					}
					operationStack.push(std::move(pMyTkn));
					pLastToken = operationStack.top().get();
				}else //binary form
				{
					auto pMyTkn = std::make_shared<BinaryPlus<T>>();
					auto my_priority = pMyTkn->getPriority();
					while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
					{
						body.push_back(operationStack.top());
						operationStack.pop();
					}
					operationStack.push(std::move(pMyTkn));
					pLastToken = operationStack.top().get();
				}
			}else if (tkn == "-")
			{
				if (pLastToken == nullptr 
					|| pLastToken->type() == TokenType::BinaryPlus || pLastToken->type() == TokenType::BinaryMinus
					|| pLastToken->type() == TokenType::operatorMul || pLastToken->type() == TokenType::operatorDiv
					|| pLastToken->type() == TokenType::operatorPow) //unary form
				{
					operationStack.push(std::make_shared<UnaryMinus<T>>());
					pLastToken = operationStack.top().get();
				}else //binary form
				{
					auto pMyTkn = std::make_shared<BinaryMinus<T>>();
					auto my_priority = pMyTkn->getPriority();
					while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority())
					{
						body.push_back(operationStack.top());
						operationStack.pop();
					}
					operationStack.push(std::move(pMyTkn));
					pLastToken = operationStack.top().get();
				}
			}else if (tkn == "*")
			{
			}else if (tkn == "/")
			{
			}else if (tkn == "^")
			{
			}else if (tkn == ",")
			{
			}

			/////////////////////////

			try
			{
				auto func = parse_token<T>(begPtr, &endPtr);
				if (func == nullptr)
					body.push_back(parse_text_token<T>(header, begPtr, &begPtr));
				else
				{
					switch (func->type())
					{
					case operatorPlus:
					default:
						throw std::exception("Unexpected token type");
					}
					operationStack.push(func);
				}
				if (*begPtr >= '0' && *begPtr <= '9')
				{
					body.push_back(parse_token<T>(begPtr, &endPtr));

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
							operationStack.push(func);
						else
							//If the parameter is not found, parse_text_token will throw via Variable c-tor, therefore there's no need for pre-check
							//nor to create the token (var) twice.
							/*auto var = parse_text_token<T>(begPtr, &endPtr);
							std::string var_name(var->get_name(), var->get_name_length());
							if (find(m_parameters.begin(), m_parameters.end(), var_name) == m_parameters.end())
								throw std::invalid_argument("Parameter is not found: " + var_name);*/
							body.push_back(parse_text_token<T>(header, begPtr, &endPtr));
						begPtr = endPtr;
						continue;
					}
					if (*begPtr == '+')
					{
						skipSpaces(begPtr + 1, expr + length - begPtr - 1);
						char tok = *(begPtr + 1);
						if (*begPtr == '+' && (tok == '+' || tok == '-')) //unary +
						{
							body.push_back(parse_token<T>(begPtr, &endPtr));
							begPtr = endPtr;
						}
						else //binary +
						{
							while (operationStack.size() != 0)
							{
								//auto plus = dynamic_cast<Operator<T>*>(operationStack.top().get());
								auto plus = operationStack.top().get();
								if (plus->type() > TokenType::function)
								{
								//	auto plus1 = dynamic_cast<Function<T>*>(operationStack.top().get());
								//	if (plus1 == nullptr)
										throw std::invalid_argument("Unexpected error at " + *begPtr);
								//	else
								//		prior = plus1->getPriority();
								}
								else
								{
									prior = plus->getPriority();
								}
								if (BinaryPlus<T>().getPriority() <= prior)
								{
									body.push_back(operationStack.top());
									operationStack.pop();
								}
								else
									break;
							}
							operationStack.push(parse_token<T>(begPtr, &endPtr));
							begPtr += 1;
						}
					}else if (*begPtr == '-')
					{
						skipSpaces(begPtr + 1, expr + length - begPtr - 1);
						char tok = *(begPtr + 1);
						if (*begPtr == '-' && (tok == '+' || tok == '-')) //unary -
						{
							body.push_back(parse_token<T>(begPtr, &endPtr));
							begPtr = endPtr;
						}
						else //binary -
						{
							while (operationStack.size() != 0)
							{
								auto minus = operationStack.top().get();
								if (minus->type() > TokenType::function)
								{
									throw std::invalid_argument("Unexpected error at " + *begPtr);
								}
								else
								{
									prior = minus->getPriority();
								}
								if (BinaryMinus<T>().getPriority() <= prior)
								{
									body.push_back(operationStack.top());
									operationStack.pop();
								}
								else
									break;
							}
							operationStack.push(parse_token<T>(begPtr, &endPtr));
							begPtr += 1;
						}
					}else if (*begPtr == '*')
					{
						while (operationStack.size() != 0)
						{
							auto mul = operationStack.top().get();
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
								body.push_back(operationStack.top());
								operationStack.pop();
							}
							else
								break;
						}
						operationStack.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}else if (*begPtr == '/')
					{
						while (operationStack.size() != 0)
						{
							auto div = operationStack.top().get();
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
								body.push_back(operationStack.top());
								operationStack.pop();
							}
							else
								break;
						}
						operationStack.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}else if (*begPtr == '^')
					{
						while (operationStack.size() != 0)
						{
							auto pow = operationStack.top().get();
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
								body.push_back(operationStack.top());
								operationStack.pop();
							}
							else
								break;
						}
						operationStack.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}else if (*begPtr == ',')
					{
						bool isOpeningBracket = false;
						while (!isOpeningBracket || operationStack.size() != 0) //while an opening bracket is not found or an operation stack is not empty
						{
							if (operationStack.top().get()->type() != TokenType::bracket) //if the cast to Bracket is not successfull, return NULL => it is not '('  
							{
								if (operationStack.top().get()->type() == TokenType::maxFunction || operationStack.top().get()->type() == TokenType::minFunction)
									++paramCount;

								body.push_back(operationStack.top());
								operationStack.pop();
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
						if (operationStack.top().get()->type() == TokenType::maxFunction || operationStack.top().get()->type() == TokenType::minFunction)
							paramCount = 1;

						operationStack.push(std::make_shared<Bracket<T>>());
						begPtr += 1;
					}
					if (*begPtr == ')')
					{
						bool isOpeningBracket = false;
						while (operationStack.size() != 0)
						{
							if (operationStack.top().get()->type() != TokenType::bracket)
							{
								if (operationStack.top().get()->type() == TokenType::maxFunction || operationStack.top().get()->type() == TokenType::minFunction)
								{
									auto min = dynamic_cast<MinFunction<T>*>(operationStack.top().get());
									auto max = dynamic_cast<MaxFunction<T>*>(operationStack.top().get());
									if (min == nullptr)
									{
										if (max == nullptr)
											throw std::invalid_argument("ERROR!");
										else
											(*max).paramsCount = paramCount;
									}
									else
									{
										(*min).paramsCount = paramCount;
									}
									paramCount = 1;
								}//проверить, является ли фукнция мин макс, записать туда количество аргументов в поле paramsCount

								body.push_back(operationStack.top());
								operationStack.pop();
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
							operationStack.pop();
						begPtr += 1;
					}
				}
			if (begPtr - delta == 0)
				throw std::invalid_argument("Invalid symbol at " + *begPtr);
			}catch (std::exception e)
			{
				throw std::invalid_argument("ERROR!");
			}
		}
		while (operationStack.size() != 0)
		{
			if (operationStack.top().get()->type() == TokenType::bracket) //checking enclosing brackets
				throw std::invalid_argument("Enclosing bracket!");
			else
			{
				body.push_back(operationStack.top());
				operationStack.pop();
			}
		}
		return body;
	}
};

template <class T>
std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr)
{
	auto tok = skipSpaces(input_string);
	if (*input_string == '+')
	{
		auto tok = skipSpaces(input_string);
		if (*tok == '-' || *tok == '+')
		{
			*endptr = (char*) tok;
			return std::make_shared<BinaryPlus<T>>();
		}
		//return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
		return nullptr;
	} //TODO: Adjust other operators accordingly

	if (*input_string == '-')
	{
		auto tok = skipSpaces(input_string);
		if (*tok == '-' || *tok == '+')
			return std::make_shared<BinaryMinus<T>>();
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
	}

	if (*input_string == '*')
		return std::make_shared<OperatorMul<T>>();
	if (*input_string == '/')
		return std::make_shared<OperatorDiv<T>>();
	if (*input_string == '^')
		return std::make_shared<OperatorPow<T>>();

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
	else if (std::strncmp(input_string, "log10", 3) == 0 && !iswhitespace(input_string[5]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<Log10Function<T>>();
	}
	else if (std::strncmp(input_string, "ln", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<LnFunction<T>>();
	}
	else if (std::strncmp(input_string, "log", 3) == 0 && !iswhitespace(input_string[3]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<LogFunction<T>>();
	}
	else if (std::strncmp(input_string, "j0", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<J0Function<T>>();
	}
	else if (std::strncmp(input_string, "j1", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<J1Function<T>>();
	}
	else if (std::strncmp(input_string, "jn", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<JnFunction<T>>();
	}
	else if (std::strncmp(input_string, "y0", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<Y0Function<T>>();
	}
	else if (std::strncmp(input_string, "y1", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<Y1Function<T>>();
	}
	else if (std::strncmp(input_string, "yn", 3) == 0 && !iswhitespace(input_string[2]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<YnFunction<T>>();
	}
	else if (std::strncmp(input_string, "max", 3) == 0 && !iswhitespace(input_string[3]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<CosFunction<T>>();
	}
	else if (std::strncmp(input_string, "min", 3) == 0 && !iswhitespace(input_string[3]))
	{
		*endptr = (char*)input_string + 3;
		return std::make_shared<CosFunction<T>>();
	}
	return nullptr;
}

template <class T>
std::shared_ptr<IToken<T>> parse_text_token(const Header<T>& header, const char* input_string, char** endptr)
{
	std::size_t name_size;
	char* endTokPtr = (char*)input_string;
	if (*input_string >= '0' && *input_string <= '9')
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
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

template <class K>
typename std::list<std::shared_ptr<IToken<K>>>::iterator simplify(std::list<std::shared_ptr<IToken<K>>>& body, typename std::list<std::shared_ptr<IToken<K>>>::iterator elem)
{
	try
	{
		bool isComputable = false;
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
void simplify_body(std::list<std::shared_ptr<IToken<T>>>& body)
{
	auto it = body.begin();
	while (body.size() > 1 )
		it = simplify(body, it);
	//When everything goes right, you are left with only one element within the list - the root of the tree.
}

template <class T>
T compute(const std::list<std::shared_ptr<IToken<T>>>& body)
{
	assert(body.size() == 1);
	return body.front()();
}

template <class T>
Mathexpr<T>::Mathexpr(const char* sMathExpr, std::size_t cbMathExpr)
{
	const char* endptr;
	header = Header<T>(sMathExpr, cbMathExpr, (char**) &endptr);
	++endptr;
	body = lexBody<T>(header, endptr, cbMathExpr - (endptr - sMathExpr));
	simplify_body(body);
}

#endif // !PARSER_H
