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
		return this->compare(right.begin(), right.size());
	}
	inline int compare(const char* pszString) const noexcept
	{
		return this->compare(pszString, std::strlen(pszString));
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

enum class TokenType : unsigned int
{
	UnaryPlus,
	UnaryMinus,
	BinaryPlus,
	BinaryMinus,
	operatorMul,
	operatorDiv,
	operatorPow,
	sqrtFunction,
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

constexpr bool IsOperatorTokenTypeId(TokenType id)
{
	return id == TokenType::BinaryPlus || id == TokenType::BinaryMinus
			|| id == TokenType::operatorMul || id == TokenType::operatorDiv
			|| id == TokenType::operatorPow;
}

template <class T, std::size_t N>
class static_parameter_storage
{
	struct {T params[N];} strg;
	T* top = strg.params;
public:
	static_parameter_storage() = default;
	static_parameter_storage(const static_parameter_storage& right)
	{
		*this = right;
	}
	static_parameter_storage(static_parameter_storage&& right)
	{
		*this = std::move(right);
	}
	static_parameter_storage& operator=(const static_parameter_storage& right)
	{
		strg = right.strg;
		return *this;
	}
	static_parameter_storage& operator=(static_parameter_storage&& right)
	{
		strg = std::move(right.strg);
		return *this;
	}
	const T& operator[](std::size_t index) const
	{
		if (index < N)
			return strg.params[index];
		throw std::range_error("static_parameter_storage: invalid parameter index");
	}
	T& operator[](std::size_t index)
	{
		return const_cast<T&>(const_cast<const static_parameter_storage&>(*this)[index]);
	}
	template <class U>
	auto push_argument(U&& arg) -> std::enable_if_t<std::is_convertible<std::decay_t<U>, T>::value>
	{
		if (top - strg.params >= N)
			throw std::range_error("static_parameter_storage: buffer overflow");
		*(top++) = std::forward<U>(arg);
	}	
	bool is_ready() const
	{
		return top == &strg.params[N] && this->is_ready_from<0>();
	}
private:
	template <std::size_t I, class = void>
	auto is_ready_from() const -> std::enable_if_t<(I >= N), bool>
	{
		return true;
	}
	template <std::size_t I, class = void>
	auto is_ready_from() const -> std::enable_if_t<(I < N), bool>
	{
		return strg.params[I]->is_ready() && this->is_ready_from<I + 1>();
	}
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
	virtual std::size_t get_required_parameter_count() const = 0;
	virtual bool is_ready() const = 0; //all parameters are specified
	virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
	virtual constexpr TokenType type() = 0;
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const = 0;
	virtual short getPriority() = 0;
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx) = 0;
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const = 0;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for number");
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
	virtual std::size_t get_required_parameter_count() const
	{
		return 0;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::number;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a number");
	}
	virtual short getPriority()
	{
		return -2;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const 
	{
		return nullptr;
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
	//TODO: temp
	Variable(const char* varname, std::size_t len)
		:m_pValue(nullptr), name_length(len)
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

	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for variable");
	}

	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a number");
	}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
#ifndef __CUDACC__
		throw std::invalid_argument("Unexpected call");
#endif //__CUDACC__
	}
	virtual std::size_t get_required_parameter_count() const
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
	virtual constexpr TokenType type()
	{
		return TokenType::variable;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		if (!strcmp((this->name).get(), target_variable))
			return std::make_shared<Variable<T>>(*this);
		return nullptr;
	}

	virtual short getPriority()
	{
		return -2;
	}
};

template <class T>
class Operator : public IToken<T>
{
	virtual void set_required_parameter_count(std::size_t value)
	{
		throw std::exception("Invalid operation");
	}

	virtual std::shared_ptr<IToken<T>> get_operand(std::size_t index) const
	{
		throw std::exception("Cannot get operand with index" + index);
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}	
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a number");
	}

	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for operator");
	}
};

template <class T>
class NoncommutativeOperator : public Operator<T>
{
	virtual void set_required_parameter_count(std::size_t value)
	{
		throw std::exception("Invalid operation");
	}

	virtual std::shared_ptr<IToken<T>> get_operand(std::size_t index) const
	{
		throw std::exception("Cannot get operand with index" + index);
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}	
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a number");
	}

	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for operator");
	}
};

template <class T>
class OperatorDiv;
template <class T>
class BinaryMinus;
template <class T>
class UnaryMinus;

template <class T>
class UnaryPlus : public Operator<T> //+-*/
{
	/*This replacement is unnecessary, but the code would be more maintainable, if the storage of parameters
	for functions (with fixed numbers of the parameters) will be managed in one place (static_parameter_storage). */
	static_parameter_storage<std::shared_ptr<IToken<T>>, 1> ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of an unary plus operator.");

		return (*ops[0])();
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!this->is_ready())
			throw std::exception("Not ready to simplify an operator");
		return ops[0]->simplify(); //unary + does no do anything
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::UnaryPlus;
	}	
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<UnaryMinus<T>>();
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		static_parameter_storage<std::shared_ptr<IToken<T>>, 1> operands;
		operands.push_argument((ops[0].get())->transform(root, target_variable, target_operator_idx));

		if (this == root.get())
		{ 
			return operands[0];
		}
		else if (operands[0] != nullptr)
			return std::make_shared<UnaryPlus<T>>(*this);

		return nullptr;
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};

template <class T>
class UnaryMinus : public Operator<T> //+-*/
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 1> ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of an unary minus operator.");

		return -(*ops[0])();
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!this->is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();

		if (op0->type() == TokenType::number)
			return std::make_shared<Number<T>>(-(*dynamic_cast<Number<T>*>(op0.get()))());
		auto op_new = std::make_shared<UnaryMinus<T>>(*this);
		op_new->push_argument(std::move(op0));
		return op_new;
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw TokenType::UnaryPlus;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::UnaryMinus;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};

template <class T>
class BinaryPlus : public Operator<T> //+-*/
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;

public:
	BinaryPlus() {}
	BinaryPlus(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands) : ops(operands) {}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
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
		return ops.is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<BinaryMinus<T>>(operands);
	}
	virtual constexpr TokenType type()
	{
		return TokenType::BinaryPlus;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands;
		operands.push_argument((ops[0].get())->transform(root, target_variable, target_operator_idx));
		operands.push_argument((ops[1].get())->transform(root, target_variable, target_operator_idx));

		for (std::size_t i = 0; i < 2; ++i)
		{
			if (operands[i].get() != nullptr) {
				*target_operator_idx = i;
				if (this == root.get())
					return operands[i];
				else
					return std::make_shared<BinaryPlus<T>>(*this);
			}
		}

		return nullptr;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};
template <class T>
class BinaryMinus : public Operator<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;

public:
	BinaryMinus() {}
	BinaryMinus(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands) : ops(operands) {}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of a binary minus operator.");

		return  (*ops[1])() - (*ops[0])();
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::BinaryMinus;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<BinaryPlus<T>>(operands);
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op1)() - (*op0)());
		auto op_new = std::make_shared<BinaryMinus<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
	virtual short getPriority()
	{
		return 2;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};
template <class T>
class OperatorMul : public Operator<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;

public:
	OperatorMul() {}
	OperatorMul(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands): ops(operands) {}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of a multiplication operator.");

		return (*ops[0])() * (*ops[1])();
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::operatorMul;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		//static_parameter_storage<std::shared_ptr<IToken<T>>, 2> reversed_operands;
		//reversed_operands.push_argument(operands[1]);
		//reversed_operands.push_argument(operands[0]);
		return std::make_shared<OperatorDiv<T>>(operands);
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx) 
	{
		return ops[idx];
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands;
		operands.push_argument((ops[0].get())->transform(root, target_variable, target_operator_idx));
		operands.push_argument((ops[1].get())->transform(root, target_variable, target_operator_idx));

		for (std::size_t i = 0; i < 2; ++i)
		{
			if (operands[i].get() != nullptr) {
				*target_operator_idx = i;
				if (this == root.get())
					return operands[i];
				else
					return std::make_shared<OperatorMul<T>>(*this);
			}
		}

		return nullptr;
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
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;

public:
	OperatorDiv() {}
	OperatorDiv(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands) : ops(operands) {}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of a division operator.");

		return (*ops[1])() / (*ops[0])();
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::operatorDiv;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		if (target_idx == 0)
		{
			static_parameter_storage<std::shared_ptr<IToken<T>>, 2> reversed_operands;
			reversed_operands.push_argument(operands[1]);
			reversed_operands.push_argument(operands[0]);
			return std::make_shared<OperatorDiv<T>>(reversed_operands);
		}
		else if (target_idx == 1)
			return std::make_shared<OperatorMul<T>>(operands);
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>((*op1)() / (*op0)());
		auto op_new = std::make_shared<OperatorDiv<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}

	virtual std::shared_ptr<IToken<T>> get_operand(std::size_t index) const
	{
		return ops[index];
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};
template <class T>
class OperatorPow : public Operator<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;

public:
	OperatorPow() {}
	OperatorPow(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands) : ops(operands) {}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const
	{
		if (!this->is_ready())
			throw std::exception("Invalid arguments of a power operator.");

		return std::pow((*ops[0])(), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 5;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::operatorPow;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("");
		//return TokenType::sqrtFunction; //TODO: operator root or log
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::pow((*op1)(), (*op0)()));
		auto op_new = std::make_shared<OperatorPow<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
public:
	virtual void set_required_parameter_count(std::size_t value)
	{
		throw std::exception("Invalid operation");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		throw std::exception("Cannot transform abstract function");
	}	
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a number");
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for function");
	}
};

template <class T>
class SqrtFunction : public Function<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 1> ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the sqrt function.");

		return std::sqrt((*ops[0])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "sqrt";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::sqrtFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return TokenType::operatorPow;
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = ops[0]->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::sqrt((*newarg)()));
		auto pNewTkn = std::make_shared<SqrtFunction<T>>();
		pNewTkn->push_argument(std::move(newarg));
		return pNewTkn;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};
template <class T>
class SinFunction : public Function<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 1> ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the sin function.");

		return std::sin((*ops[0])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "sin";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::sinFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("");
		//return TokenType::sinFunction; //TODO: arcsin
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto newarg = ops[0]->simplify();
		if (newarg->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::sin((*newarg)()));
		auto pNewTkn = std::make_shared<SinFunction<T>>();
		pNewTkn->push_argument(std::move(newarg));
		return pNewTkn;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
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
			throw std::exception("Insufficient number are given for the cos function.");

		return std::cos((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "cos";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::cosFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("");
		//return TokenType::cosFunction; //TODO: arccos
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
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
			throw std::exception("Insufficient number are given for the tg function.");

		return std::tan((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "tg";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::tgFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("");
		//return TokenType::tgFunction; //TODO: arctg
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
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
			throw std::exception("Insufficient number are given for the log10 function.");

		return std::log10((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "log10";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::log10Function;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<OperatorPow<T>>(operands); //TODO: pow to 10
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
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
			throw std::exception("Insufficient number are given for the ln function.");

		return std::log((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "ln";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::lnFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<OperatorPow<T>>(operands); //TODO: pow e
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
	}
};
template <class T>
class LogFunction : public Function<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the log function.");

		return std::log((*ops[1])()) / std::log((*ops[0])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual const char* get_function_name() const
	{
		return "log";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::logFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		return std::make_shared<OperatorPow<T>>(operands); //TODO: pow arbitrary
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify an operator");
		auto op0 = ops[0]->simplify();
		auto op1 = ops[1]->simplify();

		if (op0->type() == TokenType::number && op1->type() == TokenType::number)
			return std::make_shared<Number<T>>(std::log((*op1)())/std::log((*op0)()));
		auto op_new = std::make_shared<LogFunction<T>>();
		op_new->push_argument(std::move(op0));
		op_new->push_argument(std::move(op1));
		return op_new;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};
template <class T>
class JnFunction : public Function<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the jn function.");

		return _jn(int((*ops[0])()), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual const char* get_function_name() const
	{
		return "jn";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::jnFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for jn");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
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
			throw std::exception("Insufficient number are given for the j0 function.");

		return _j0((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "j0";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::j0Function;
	}	
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for j0");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
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
			throw std::exception("Insufficient number are given for the j1 function.");

		return _j1((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "j1";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::j1Function;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for j1");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
	}
};
template <class T>
class YnFunction : public Function<T>
{
	static_parameter_storage<std::shared_ptr<IToken<T>>, 2> ops;
public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_argument(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the yn function.");

		return _yn(int((*ops[0])()), (*ops[1])());
	}
	virtual bool is_ready() const
	{
		return ops.is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 2;
	}
	virtual const char* get_function_name() const
	{
		return "yn";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::ynFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for yn");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
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
			throw std::exception("Insufficient number are given for the y0 function.");

		return _y0((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "y0";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::y0Function;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for y0");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
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
			throw std::exception("Insufficient number are given for the y1 function.");

		return _y1((*op)());
	}
	virtual bool is_ready() const
	{
		return op->is_ready();
	}
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "y1";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::y1Function;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for y1");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return op;
	}
};

template <class T, class Implementation, class TokenBinPredicate>
class ExtremumFunction : public Function<T>
{
	std::vector<std::shared_ptr<IToken<T>>> ops;
	std::size_t nRequiredParamsCount = 0;
	TokenBinPredicate m_pred;
public:
	ExtremumFunction() = default;
	ExtremumFunction(std::size_t paramsNumber) : nRequiredParamsCount(paramsNumber) {}

	//not used, but in case a state is needed by the definition of the predicate:
	template <class Predicate, class = std::enable_if_t<std::is_constructible<TokenBinPredicate, Predicate&&>::value>>
	ExtremumFunction(std::size_t paramsNumber, Predicate&& pred) : nRequiredParamsCount(paramsNumber), m_pred(std::forward<Predicate>(pred)) {}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		ops.push_back(value);
	}
	virtual T operator()() const /*Implementation of IToken<T>::operator()()*/
	{
		if (!this->is_ready())
			throw std::exception("Insufficient number are given for the extremum function.");

		return (*std::min_element(ops.begin(), ops.end(), m_pred)).get()->operator()();
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for extremum");
	}
	virtual bool is_ready() const
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
	virtual short getPriority()
	{
		return 4;
	}
	virtual std::size_t get_required_parameter_count() const
	{
		return nRequiredParamsCount;
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		if (!is_ready())
			throw std::exception("Not ready to simplify");
		std::vector<std::shared_ptr<IToken<T>>> newargs;
		newargs.reserve(ops.size());
		std::vector<std::shared_ptr<IToken<T>>> newargsVar;
		newargsVar.reserve(ops.size());

		for (const auto& op:ops)
		{
			auto newarg = op->simplify();
			if (newarg->type() == TokenType::number)
				newargs.push_back(newarg);
			else
				newargsVar.push_back(newarg);
		}
		if (newargsVar.empty())
			return *std::min_element(newargs.begin(), newargs.end(), m_pred);

		std::shared_ptr<Implementation> pNewTkn;
		if (newargs.empty())
			pNewTkn = std::make_shared<Implementation>(Implementation(newargsVar.size()));
		else
		{
			pNewTkn = std::make_shared<Implementation>(Implementation(newargsVar.size() + 1));
			pNewTkn->push_argument(*std::min_element(newargs.begin(), newargs.end(), m_pred));
		}
		for(const auto& op:newargsVar)
			pNewTkn->push_argument(op);
		return pNewTkn;
	}
	void set_required_parameter_count(std::size_t value)
	{
		nRequiredParamsCount = value;
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		return ops[idx];
	}
};

template <class T>
struct TokenLess
{
	bool operator()(const std::shared_ptr<IToken<T>>& left, const std::shared_ptr<IToken<T>>& right) const
	{
		return (*left)() < (*right)();
	};
};

template <class T>
struct TokenGreater
{
	bool operator()(const std::shared_ptr<IToken<T>>& left, const std::shared_ptr<IToken<T>>& right) const
	{
		return (*left)() > (*right)();
	};
};

template <class T>
class MaxFunction : public ExtremumFunction<T, MaxFunction<T>, TokenGreater<T>>
{
	typedef ExtremumFunction<T, MaxFunction<T>, TokenGreater<T>> MyBase;
public:
	using MyBase::ExtremumFunction; //c-tor inheritance
	virtual const char* get_function_name() const
	{
		return "max";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::maxFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
};
template <class T>
class MinFunction : public ExtremumFunction<T, MinFunction<T>, TokenLess<T>>
{
	typedef ExtremumFunction<T, MinFunction<T>, TokenLess<T>> MyBase;
public:
	using MyBase::ExtremumFunction; //c-tor inheritance
	virtual const char* get_function_name() const
	{
		return "min";
	}
	virtual constexpr TokenType type()
	{
		return TokenType::minFunction;
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("Error");
	}
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
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
	virtual std::shared_ptr<IToken<T>> transform(std::shared_ptr<IToken<T>> root, const char* target_variable, std::size_t* target_operator_idx) const
	{
		return nullptr;
	}
	virtual constexpr TokenType type()
	{
		return TokenType::bracket; 
	}
	virtual std::shared_ptr<IToken<T>> get_reverse_operator(const static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands, std::size_t target_idx = -1) const
	{
		throw std::exception("There is no reverse operation for a bracket");
	}
	virtual std::shared_ptr<IToken<T>> simplify() const
	{
		//return std::make_shared<Bracket<T>>(nullptr);
		throw std::exception("Unexpected call");
	}
	std::size_t get_required_parameter_count() const
	{
		return 0;
	}
	void set_required_parameter_count(std::size_t value)
	{
		throw std::exception("Invalid operation");
	}
	virtual std::shared_ptr<IToken<T>> get_param(std::size_t idx)
	{
		throw std::exception("not implemented for bracket");
	}
};

template <class T>
class TokenStorage
{
	std::stack<std::shared_ptr<IToken<T>>> operationStack;
	std::list<std::shared_ptr<IToken<T>>> outputList;

public:

	template <class TokenParamType>
	auto push_token(TokenParamType&& op) -> std::enable_if_t<
		std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
		std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value,
		IToken<T>*
	>
	{
		//not ready
		auto my_priority = op.getPriority();
		while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority() && op.type() != TokenType::bracket)
		{
			outputList.push_back(operationStack.top());
			operationStack.pop();
		}
		operationStack.push(std::make_shared<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(op)));
		return operationStack.top().get();
	}

	template <class TokenParamType>
	auto push_token(TokenParamType&& value) -> std::enable_if_t<
		!(std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
		std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value ||
		std::is_base_of<Bracket<T>, std::decay_t<TokenParamType>>::value),
		IToken<T>*
	>
	{
		//not ready
		outputList.push_back(std::make_shared<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(value)));
		return outputList.back().get();
	}

	void pop_bracket()
	{
		bool isOpeningBracket = false;
		while (operationStack.size() != 0)
		{
			if (operationStack.top().get()->type() != TokenType::bracket)
			{
				this->outputList.push_back(operationStack.top());
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

		//return Bracket<T>(); //: operationStack.top().get();
	}
	std::list<std::shared_ptr<IToken<T>>>&& finalize() &&
	{
		while (operationStack.size() != 0)
		{
			if (operationStack.top().get()->type() == TokenType::bracket) //checking enclosing brackets
				throw std::invalid_argument("Enclosing bracket!");
			else
			{
				outputList.push_back(std::move(operationStack.top()));
				operationStack.pop();
			}
		}
		return std::move(outputList);
	}
	std::shared_ptr<IToken<T>>& get_top_operation()
	{
		return operationStack.top();
	}
	void comma_parameter_replacement()
	{
		bool isOpeningBracket = false;

		while (!isOpeningBracket && operationStack.size() != 0) //while an opening bracket is not found or an operation stack is not empty
		{
			if (operationStack.top().get()->type() != TokenType::bracket) //if the cast to Bracket is not successfull, return NULL => it is not '('
			{
				outputList.push_back(operationStack.top());
				operationStack.pop();
			}
			else
			{
				isOpeningBracket = true;
			}
		}

		if (!isOpeningBracket) //missing '('
			throw std::invalid_argument("There is no opening bracket!");
	}
};

template <class T>
class Header
{
	std::map<std::string, T> m_arguments;
	std::vector<std::string> m_parameters;
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
		std::size_t commaCount = 0;

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
	Header(const Header<T>& val) 
	{
		std::size_t size = val.get_name_length();
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
	std::size_t get_required_parameter_count() const
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
	std::shared_ptr<IToken<T>> get_header_expression() const
	{
		return std::make_shared<Variable<T>>(function_name.c_str(), function_name.size());
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
		//auto result = body;
		//simplify_body(result);
		//if (result.size() != 1)
		//	throw std::exception("Invalid expression");
		return (*body)();
	}
	void init_variables(const std::vector<T>& parameters)
	{
		if (parameters.size() < header.get_required_parameter_count())
			throw std::invalid_argument("Count of arguments < " + header.get_required_parameter_count());
		for (std::size_t iArg = 0; iArg < header.get_required_parameter_count(); ++iArg)
			header.get_argument_by_index(iArg) = parameters[iArg];
	}

	std::shared_ptr<IToken<T>> transformation(const char* target_parameter, std::size_t parameter_size) 
	{
		auto pBody = body;
		auto header_expression = header.get_header_expression();
		std::size_t* target_param_idx = new std::size_t();
		try {
			auto param = header.get_argument(target_parameter, parameter_size);

			while ((pBody.get())->type() != TokenType::variable || (pBody.get())->type() != TokenType::variable)
			{
				auto target_variable_operator = (pBody.get())->transform(pBody, target_parameter, target_param_idx);
				auto another_operator_idx = (pBody.get())->get_required_parameter_count() - 1 - *target_param_idx;
				auto res = reverse_operation(pBody, another_operator_idx, header_expression);
				header_expression = res;
				pBody = target_variable_operator;
			}
		}
		catch (std::exception e)
		{
			return nullptr;
		}
		return header_expression;
		//for (auto i = body.get(); i != (body.get())->end(); ++i)
		//{
		//	*i;
		//}
	}

	//void clear_variables(); With the map referencing approach this method is not necessary anymore because if we need to reuse the expression
	//with different arguments, we just reassign them with init_variables
private:
	Header<T> header;
	std::shared_ptr<IToken<T>> body;

	std::shared_ptr<IToken<T>> reverse_operation(std::shared_ptr<IToken<T>> operation,
		std::size_t reverse_param_index, std::shared_ptr<IToken<T>> header)
	{
		//TODO: use header
		//auto head = std::make_shared<Variable<T>>("F", 1);
		std::shared_ptr<IToken<T>> reverse_param = (operation.get())->get_param(reverse_param_index);
		TokenType type = (operation.get())->type();
		static_parameter_storage<std::shared_ptr<IToken<T>>, 2> operands;
		
		operands.push_argument(reverse_param);
		operands.push_argument(header);
		
		auto new_operator = (operation.get())->get_reverse_operator(operands, reverse_param_index);
		return new_operator;
	}

	template <class T>
	std::list<std::shared_ptr<IToken<T>>> lexBody(const char* expr, std::size_t length)
	{
		char* begPtr = (char*)expr;
		std::size_t cbRest = length;
		TokenStorage<T> tokens;
		IToken<T> *pLastToken = nullptr;
		std::stack <std::pair<std::shared_ptr<Function<T>>, std::size_t>> funcStack;
		int last_type_id = -1;
		std::list<std::shared_ptr<IToken<T>>> formula;

		while (cbRest > 0)
		{
			auto tkn = parse_string_token(begPtr, cbRest);
			if (tkn == "+")
			{
				if (last_type_id == -1 || IsOperatorTokenTypeId(TokenType(last_type_id))) //unary form
					last_type_id = int(tokens.push_token(UnaryPlus<T>())->type());
				else //binary form
					last_type_id = int(tokens.push_token(BinaryPlus<T>())->type());
			}
			else if (tkn == "-")
			{
				if (last_type_id == -1 || IsOperatorTokenTypeId(TokenType(last_type_id))) //unary form
					last_type_id = int(tokens.push_token(UnaryMinus<T>())->type());
				else //binary form
					last_type_id = int(tokens.push_token(BinaryMinus<T>())->type());
			}
			else if (tkn == "*")
				last_type_id = int(tokens.push_token(OperatorMul<T>())->type());
			else if (tkn == "/")
				last_type_id = int(tokens.push_token(OperatorDiv<T>())->type());
			else if (tkn == "^")
				last_type_id = int(tokens.push_token(OperatorPow<T>())->type());
			else if (tkn == ",")
			{
				tokens.comma_parameter_replacement();

				if (funcStack.top().first.get()->type() == TokenType::maxFunction ||
					funcStack.top().first.get()->type() == TokenType::minFunction)
					funcStack.top().second += 1;
			}
			else if (isdigit(*tkn.begin()))
			{
				char* conversion_end;
				static_assert(std::is_same<T, double>::value, "The following line is only applicable to double");
				auto value = std::strtod(tkn.begin(), (char**) &conversion_end);
				if (conversion_end != tkn.end())
					std::exception("Invalid syntax");
				last_type_id = int(tokens.push_token(Number<T>(value))->type());
			}
			else if (isalpha(*tkn.begin()))
			{
				if (tkn == "sin")
				{
					last_type_id = int(tokens.push_token(SinFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<SinFunction<T>>(), 1));
				}
				else if (tkn == "cos")
				{
					last_type_id = int(tokens.push_token(CosFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<CosFunction<T>>(), 1));
				}
				else if (tkn == "tg")
				{
					last_type_id = int(tokens.push_token(TgFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<TgFunction<T>>(), 1));
				}
				else if (tkn == "log10")
				{
					last_type_id = int(tokens.push_token(Log10Function<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<Log10Function<T>>(), 1));
				}
				else if (tkn == "ln")
				{
					last_type_id = int(tokens.push_token(LnFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<LnFunction<T>>(), 1));
				}
				else if (tkn == "log")
				{
					last_type_id = int(tokens.push_token(LogFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<LogFunction<T>>(), 2));
				}
				else if (tkn == "j0")
				{
					last_type_id = int(tokens.push_token(J0Function<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<J0Function<T>>(), 1));
				}
				else if (tkn == "j1")
				{
					last_type_id = int(tokens.push_token(J1Function<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<J1Function<T>>(), 1));
				}
				else if (tkn == "jn")
				{
					last_type_id = int(tokens.push_token(JnFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<JnFunction<T>>(), 2));
				}
				else if (tkn == "y0")
				{
					last_type_id = int(tokens.push_token(Y0Function<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<Y0Function<T>>(), 1));
				}
				else if (tkn == "y1")
				{
					last_type_id = int(tokens.push_token(Y1Function<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<Y1Function<T>>(), 1));
				}
				else if (tkn == "yn")
				{
					last_type_id = int(tokens.push_token(YnFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<YnFunction<T>>(), 2));
				}
				else if (tkn == "max")
				{
					last_type_id = int(tokens.push_token(MaxFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<MaxFunction<T>>(), 0));
				}
				else if (tkn == "min")
				{
					last_type_id = int(tokens.push_token(MinFunction<T>())->type());
					funcStack.push(std::make_pair(std::make_shared<MinFunction<T>>(), 0));
				}
				else if (this->header.get_param_index(std::string(tkn.begin(), tkn.end())) >= 0)
					last_type_id = int(tokens.push_token(Variable<T>(this->header, std::string(tkn.begin(), tkn.end()).c_str(), tkn.end() - tkn.begin()))->type());
			}
			else if (tkn == ")")
			{
				tokens.pop_bracket();
				switch (funcStack.top().first.get()->type())
				{
				case TokenType::maxFunction:
					static_cast<MaxFunction<T>&>(*tokens.get_top_operation()) = MaxFunction<T>(funcStack.top().second + 1);
					break;
				case TokenType::minFunction:
					static_cast<MinFunction<T>&>(*tokens.get_top_operation()) = MinFunction<T>(funcStack.top().second + 1);
					break;
				}
				funcStack.pop();
			}
			else if (tkn == "(")
			{
				last_type_id = int(tokens.push_token(Bracket<T>())->type());
			}
			else
				throw std::exception("Unexpected token");
			cbRest -= tkn.end() - begPtr;
			begPtr = (char*)tkn.end();
		}
		
		//body = std::move(tokens).finalize();
		formula = std::move(tokens).finalize();
		return formula;
	}
};

template <class K>
typename std::list<std::shared_ptr<IToken<K>>>::iterator simplify(std::list<std::shared_ptr<IToken<K>>>& body, typename std::list<std::shared_ptr<IToken<K>>>::iterator elem)
{
	try
	{
		bool isComputable = false;
		auto paramsCount = elem->get()->get_required_parameter_count();
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
auto simplify_body(std::list<std::shared_ptr<IToken<T>>>&& body)
{
	auto it = body.begin();
	while (body.size() > 1 )
		it = simplify(body, it);
	//auto val = it;
	return body.front();
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
	//lexBody<T>(endptr, cbMathExpr - (endptr - sMathExpr));
	auto formula = std::move(lexBody<T>(endptr, cbMathExpr - (endptr - sMathExpr)));
	this->body = simplify_body(std::move(formula));
}

#endif // !PARSER_H
