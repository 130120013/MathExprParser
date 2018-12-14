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

#ifndef PARSER_H
#define PARSER_H

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

template <class T>
class IToken
{
public:
	virtual T operator()() const = 0; //All derived classes must implement the same method with the same return type
	/*Push a required (as defined at runtime, perhaps erroneously which must be detected by implementations) number of arguments.
	Push arguments from first to last. Then call the operator() above.
	*/
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value) = 0;
	virtual bool is_ready() const = 0; //all parameters are specified
	virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
	virtual std::string type() = 0;
	virtual void drop_ready_flag() = 0;
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
		return;
		//#ifndef __CUDACC__
		//		throw bad_expession_parameters("An argument is specified for a literal");
		//#endif
	}
	virtual std::string type()
	{
		return "num";
	}
	void drop_ready_flag()
	{
		return;
	}
protected:
private:
	T value;
};

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{
	T op = 0;
	std::unique_ptr<char[]> name;
	std::size_t name_length = 0;
	bool isReady;
public:
	Variable(char* varname, std::size_t len, T value = std::numeric_limits<T>::max()) : op(value), name_length(len)
	{
		this->name = std::make_unique<char[]>(len + 1);
		std::strncpy(this->name.get(), varname, len);
		this->name[len] = 0;
		this->op = value;
		isReady = (value == std::numeric_limits<T>::max()) ? false : true;
	}
	Variable(Variable<T>&& val) : op(val()), name_length(val.get_name_length()), isReady(val.is_ready())
	{
		this->name = std::make_unique<char[]>(val.get_name_length() + 1);
		std::strncpy(this->name.get(), val.get_name(), val.get_name_length());
		this->name[val.get_name_length()] = 0;
	}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		if (this->isReady)
			throw std::invalid_argument("ERROR!");
		op = value.get()->operator()();
		isReady = true;
	}
	virtual T operator()() const
	{
		return op;
	}
	virtual bool is_ready() const
	{
		return isReady;
	}
	char* get_name() const
	{
		return name.get();
	}
	size_t get_name_length()
	{
		return name_length;
	}
	virtual std::string type()
	{
		return "var";
	}
	void drop_ready_flag()
	{
		isReady = false;
	}
};

template <class T>
class Operator : public IToken<T>
{
public:
	virtual short getPriority()
	{
		return 0; //default priority, less code but more error prone
	}
};

template <class T>
class OperatorPlus : public Operator<T> //+-*/
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		auto opN1 = dynamic_cast<Number<T>*>(ops[0].get());
		auto opN2 = dynamic_cast<Number<T>*>(ops[1].get());
		auto opV1 = dynamic_cast<Variable<T>*>(ops[0].get());
		auto opV2 = dynamic_cast<Variable<T>*>(ops[1].get());

		T val1, val2;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}
		if (opN2 == nullptr)
		{
			if (opV2->is_ready())
				val2 = opV2->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val2 = opN2->operator()();
		}
		return val1 + val2;
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual int get_params_count() const
	{
		return 2;
	}
	virtual std::string type()
	{
		return "plus";
	}
	void drop_ready_flag()
	{
		top = ops;
		return;
	}
};
template <class T>
class OperatorMinus : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const/*Implementation of IToken<T>::operator()()*/
	{
		auto opN1 = dynamic_cast<Number<T>*>(ops[0].get());
		auto opN2 = dynamic_cast<Number<T>*>(ops[1].get());
		auto opV1 = dynamic_cast<Variable<T>*>(ops[0].get());
		auto opV2 = dynamic_cast<Variable<T>*>(ops[1].get());

		T val1, val2;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}
		if (opN2 == nullptr)
		{
			if (opV2->is_ready())
				val2 = opV2->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val2 = opN2->operator()();
		}
		return val1 - val2;
	}
	virtual bool is_ready() const
	{
		return true;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	virtual std::string type()
	{
		return "minus";
	}
	void drop_ready_flag()
	{
		top = ops;
		return;
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
		auto opN1 = dynamic_cast<Number<T>*>(ops[0].get());
		auto opN2 = dynamic_cast<Number<T>*>(ops[1].get());
		auto opV1 = dynamic_cast<Variable<T>*>(ops[0].get());
		auto opV2 = dynamic_cast<Variable<T>*>(ops[1].get());

		T val1, val2;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}
		if (opN2 == nullptr)
		{
			if (opV2->is_ready())
				val2 = opV2->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val2 = opN2->operator()();
		}
		return val1 * val2;
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
	virtual std::string type()
	{
		return "mul";
	}
	void drop_ready_flag()
	{
		top = ops;
		return;
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
	std::shared_ptr<IToken<T>> ops[2], *top = ops;

public:
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		*top++ = value;
	}
	virtual T operator()() const
	{
		auto opN1 = dynamic_cast<Number<T>*>(ops[0].get());
		auto opN2 = dynamic_cast<Number<T>*>(ops[1].get());
		auto opV1 = dynamic_cast<Variable<T>*>(ops[0].get());
		auto opV2 = dynamic_cast<Variable<T>*>(ops[1].get());

		T val1, val2;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}
		if (opN2 == nullptr)
		{
			if (opV2->is_ready())
				val2 = opV2->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val2 = opN2->operator()();
		}
		return val1 / val2;
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
	virtual std::string type()
	{
		return "div";
	}
	void drop_ready_flag()
	{
		top = ops;
		return;
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
		auto opN1 = dynamic_cast<Number<T>*>(ops[0].get());
		auto opN2 = dynamic_cast<Number<T>*>(ops[1].get());
		auto opV1 = dynamic_cast<Variable<T>*>(ops[0].get());
		auto opV2 = dynamic_cast<Variable<T>*>(ops[1].get());

		T val1, val2;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}
		if (opN2 == nullptr)
		{
			if (opV2->is_ready())
				val2 = opV2->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val2 = opN2->operator()();
		}
		return std::pow(val1, val2);
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
	virtual std::string type()
	{
		return "pow";
	}
	void drop_ready_flag()
	{
		top = ops;
		return;
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
	std::list<std::shared_ptr<IToken<T>>> m_parameters;
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
	virtual std::string type()
	{
		return "func";
	}
	virtual short getPriority()
	{
		return 1;
	}
	void drop_ready_flag()
	{
		return;
	}
protected:
	std::list<T>& parameter_queue()
	{
		return m_parameters;
	}
	const std::list<std::shared_ptr<IToken<T>>>& parameter_queue() const
	{
		return m_parameters;
	}
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
		auto opN1 = dynamic_cast<Number<T>*>(op.get());
		auto opV1 = dynamic_cast<Variable<T>*>(op.get());

		T val1;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}

		return std::sin(val1);
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
	virtual std::string type()
	{
		return "sin";
	}
	virtual short getPriority()
	{
		return 2;
	}
	void drop_ready_flag()
	{
		return;
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
		auto opN1 = dynamic_cast<Number<T>*>(op.get());
		auto opV1 = dynamic_cast<Variable<T>*>(op.get());

		T val1;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}

		return std::cos(val1);
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
	virtual std::string type()
	{
		return "cos";
	}
	virtual short getPriority()
	{
		return 2;
	}
	void drop_ready_flag()
	{
		return;
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
		auto opN1 = dynamic_cast<Number<T>*>(op.get());
		auto opV1 = dynamic_cast<Variable<T>*>(op.get());

		T val1;

		if (opN1 == nullptr)
		{
			if (opV1->is_ready())
				val1 = opV1->operator()();
			else
				return std::numeric_limits<T>::max();
		}
		else
		{
			val1 = opN1->operator()();
		}

		return std::tan(val1);
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
	virtual std::string type()
	{
		return "tg";
	}
	void drop_ready_flag()
	{
		return;
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
	virtual std::string type()
	{
		return "br"; 
	}
	void drop_ready_flag()
	{
		return;
	}
};

template <class T>
std::shared_ptr<Variable<T>> parse_text_token(const char* input_string, char** endptr);

template <class T>
class Header
{
	std::map<std::string, T> m_arguments;
	std::vector<std::string> m_parameters;
	std::unique_ptr<char[]> function_name;
	std::size_t function_name_length = 0;
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
			if ((*begPtr >= 'A' && *begPtr <= 'Z') || (*begPtr >= 'a' && *begPtr <= 'z'))
			{
				if (this->function_name == nullptr)
				{
					while (begPtr < expression + expression_len + 1)
					{

						auto brPtr = std::strstr(expression, "(");
						if (brPtr == nullptr)
						{
							if (begPtr == expression + expression_len)
								throw std::invalid_argument("ERROR");
							begPtr += 1;
						}
						else
						{
							begPtr = (char*)brPtr;
							auto size = std::size_t(brPtr - expression);
							this->function_name = std::make_unique<char[]>(size + 1);
							std::strncpy(this->function_name.get(), expression, size);
							this->function_name[size] = 0;
							this->function_name_length = size;
							break;
						}
					}
					continue;
				}
				//char param_name;
				auto param = parse_text_token<T>(begPtr, endPtr);//, param_name
				if (!m_arguments.emplace(param.get()->get_name(), T()).second)
					throw std::invalid_argument("Parameter is not unique!"); //duplicated '('
				params.emplace_back(param.get()->get_name());
				begPtr = *endPtr;
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
	Header(const Header<T>& val) : function_name_length(val.get_name_length())
	{
		std::size_t size = val.get_name_length();
		this->function_name = std::make_unique<char[]>(size + 1);
		std::strncpy(this->function_name.get(), val.get_function_name(), size);
		this->function_name[size] = 0;
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
	int get_params_count() const
	{
		return m_parameters.size();
	}
	const char* get_function_name() const
	{
		return function_name.get();
	}
	size_t get_name_length() const
	{
		return function_name_length;
	}
	const std::vector<std::string>& get_params_vector() const
	{
		return m_parameters;
	}
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
		auto pNumber = dynamic_cast<Number<T>*>(result.front().get());
		if (pNumber == nullptr)
			throw std::exception("Invalid expression");
		return (*pNumber)();
	}
	void init_variables(const std::vector<T>& parameters)
	{
		if (parameters.size() < header.get_params_count())
			throw std::invalid_argument("Count of arguments < " + header.get_params_count());
		auto it = this->body.begin();
		while (it != this->body.end())
		{
			auto var = dynamic_cast<Variable<T>*>((*it).get());
			if (var != nullptr)
			{
				int idx = int(header.get_param_index(std::string(var->get_name(), var->get_name_length())));
				(*it).get()->push_argument(std::make_shared<Variable<T>>(var->get_name(), var->get_name_length(), parameters[idx]));
			}
			++it;
		}
	}
	void clear_variables()
	{
		auto it = this->body.begin();
		while (it != this->body.end())
		{
			if (dynamic_cast<Variable<T>*>((*it).get()) != nullptr ||
				dynamic_cast<OperatorPlus<T>*>((*it).get()) != nullptr ||
				dynamic_cast<OperatorMinus<T>*>((*it).get()) != nullptr ||
				dynamic_cast<OperatorMul<T>*>((*it).get()) != nullptr ||
				dynamic_cast<OperatorDiv<T>*>((*it).get()) != nullptr ||
				dynamic_cast<OperatorPow<T>*>((*it).get()) != nullptr)
				(*it).get()->drop_ready_flag();
			++it;
		}
	}
private:
	Header<T> header;
	std::list<std::shared_ptr<IToken<T>>> body;
};

inline void skipSpaces(char* input_string, std::size_t length)
{
	char* endTokPtr = (char*)(input_string + length);
	while ((*input_string == '=' || *input_string == '\t' || *input_string == ' ' || *input_string == '\0') && input_string < endTokPtr)
	{
		input_string += 1;
	}
}

template <class T>
std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr)
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

	if (*input_string == 's')
	{
		if (std::strstr(input_string, "sin") != NULL)
		{
			return std::make_shared<SinFunction<T>>();
		}
	}
	if (*input_string == 'c')
	{
		if (std::strstr(input_string, "cos") != NULL)
		{
			return std::make_shared<CosFunction<T>>();
		}
	}
	if (*input_string == 't')
	{
		if (std::strstr(input_string, "tg") != NULL)
		{
			return std::make_shared<TgFunction<T>>();
		}
	}
	return NULL;
}

template <class T>
std::shared_ptr<Variable<T>> parse_text_token(const char* input_string, char** endptr)
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
	auto token = std::make_shared<Variable<T>>(tok_name, name_size);
	*(endptr) = endTokPtr;
	return token;
}

template <class T>
std::list<std::shared_ptr<IToken<T>>> lexBody(const char* expr, std::size_t length, const std::vector<std::string>& m_parameters)
{
	std::list<std::shared_ptr<IToken<T>>> output;
	int prior;

	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto delta = (char*)expr;
	bool isBinary = true;
	short hasPunct = 0;
	std::stack<std::shared_ptr<IToken<T>>> operationQueue;

	while (*begPtr != NULL && *begPtr != '\0' && begPtr != expr + length)
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
					if (*begPtr == 's') //sin
					{
						auto funcSin = parse_token<T>(begPtr, &endPtr);
						if (funcSin != NULL)
							operationQueue.push(funcSin);
						else
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						begPtr += 3;
						continue;
					}
					if (*begPtr == 'c') //cos
					{
						auto funcCos = parse_token<T>(begPtr, &endPtr);
						if (funcCos != NULL)
							operationQueue.push(funcCos);
						else
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						begPtr += 3;
						continue;
					}
					if (*begPtr == 't') //tg
					{
						auto funcTg = parse_token<T>(begPtr, &endPtr);
						if (funcTg != NULL)
							operationQueue.push(funcTg);
						else
							throw std::invalid_argument("Unexpected error at " + *begPtr);
						begPtr += 2;
						continue;
					}
					auto var = parse_text_token<T>(begPtr, &endPtr);
					std::string var_name(var.get()->get_name(), var.get()->get_name_length());
					if (find(m_parameters.begin(), m_parameters.end(), var_name) != m_parameters.end())
					{
						output.push_back(parse_text_token<T>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else
						throw std::invalid_argument("Parameter is not found: " + var_name);
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
							auto plus = dynamic_cast<Operator<T>*>(operationQueue.top().get());
							if (plus == nullptr)
							{
								auto plus1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
								if (plus1 == nullptr)
									throw std::invalid_argument("Unexpected error at " + *begPtr);
								else
									prior = plus1->getPriority();
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
				}
				if (*begPtr == '-')
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
							auto minus = dynamic_cast<Operator<T>*>(operationQueue.top().get());
							if (minus == nullptr)
							{
								auto minus1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
								if (minus1 == nullptr)
									throw std::invalid_argument("Unexpected error at " + *begPtr);
								else
									prior = minus1->getPriority();
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
				}
				if (*begPtr == '*')
				{
					while (operationQueue.size() != 0)
					{

						auto mul = dynamic_cast<Operator<T>*>(operationQueue.top().get());
						if (mul == nullptr)
						{
							auto mul1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
							if (mul1 == nullptr)
								throw std::invalid_argument("Unexpected error at " + *begPtr);
							else
								prior = mul1->getPriority();
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
				}
				if (*begPtr == '/')
				{
					while (operationQueue.size() != 0)
					{
						auto div = dynamic_cast<Operator<T>*>(operationQueue.top().get());
						if (div == nullptr)
						{
							auto div1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
							if (div1 == nullptr)
								throw std::invalid_argument("Unexpected error at " + *begPtr);
							else
								prior = div1->getPriority();
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
				}
				if (*begPtr == '^')
				{
					while (operationQueue.size() != 0)
					{
						auto pow = dynamic_cast<Operator<T>*>(operationQueue.top().get());
						if (pow == nullptr)
						{
							auto pow1 = dynamic_cast<Function<T>*>(operationQueue.top().get());
							if (pow1 == nullptr)
								throw std::invalid_argument("Unexpected error at " + *begPtr);
							else
								prior = pow1->getPriority();
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
				}
				if (*begPtr == ',')
				{
					bool isOpeningBracket = false;
					while (!isOpeningBracket || operationQueue.size() != 0) //while an opening bracket is not found or an operation stack is not empty
					{
						if (dynamic_cast<Bracket<T>*>(operationQueue.top().get()) == NULL) //if the cast to Bracket is not successfull, return NULL => it is not '('  
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
						if (operationQueue.size() != 0 && dynamic_cast<Bracket<T>*>(operationQueue.top().get()) == NULL)
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
		if (dynamic_cast<Bracket<T>*>(operationQueue.top().get()) != NULL) //checking enclosing brackets
			throw std::invalid_argument("Enclosing bracket!");
		else
		{
			output.push_back(operationQueue.top());
			operationQueue.pop();
		}
	}
	return output;
}

template <class T, class K>
void compute(std::list<std::shared_ptr<IToken<K>>>& body, typename std::list<std::shared_ptr<IToken<K>>>::iterator elem)
{
	try
	{
		auto val = dynamic_cast<T*>((*elem).get());
		bool isComputable = false;
		auto paramsCount = val->get_params_count();

		for (auto i = paramsCount; i > 0; --i)
		{
			auto param_it = elem;
			std::advance(param_it, -1 * i);
			((*elem).get())->push_argument(*param_it); //here std::move must be
			body.remove(*param_it);
		}

		if (val->is_ready())
		{
			K res = val->operator()();
			auto calc = std::make_shared<Number<K>>(res);
			*elem = calc;
		}
		++elem;
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
	{
		std::string type = (*it).get()->type();
		if (type == "plus")
		{
			compute<OperatorPlus<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "minus")
		{
			compute<OperatorMinus<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "mul")
		{
			compute<OperatorMul<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "div")
		{
			compute<OperatorDiv<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "pow")
		{
			compute<OperatorPow<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "sin")
		{
			compute<SinFunction<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "cos")
		{
			compute<CosFunction<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "tg")
		{
			compute<TgFunction<T>>(body, it);
			it = body.begin();
			continue;
		}
		if (type == "num")
		{
			++it;
			continue;
		}
		if (type == "var")
		{
			++it;
			continue;
		}

		throw std::invalid_argument("Missed operator or function");
	}
}

template <class T>
Mathexpr<T>::Mathexpr(const char* sMathExpr, std::size_t cbMathExpr)
{
	const char* endptr;
	header = Header<T>(sMathExpr, cbMathExpr, (char**) &endptr);
	++endptr;
	body = lexBody<T>(endptr, cbMathExpr - (endptr - sMathExpr), header.get_params_vector());
	simplify_body(body);
}

#endif // !PARSER_H
