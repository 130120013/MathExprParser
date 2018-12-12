#ifndef TOKENS_H
#define TOKENS_H

//#include <iostream> - what for?
#include <queue> //to store arguments
#include <stdexcept> //for exceptions
#include <memory>
#include <map>
#include <list>
#include <limits>

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
	virtual std::size_t get_params_count() const
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
	std::size_t get_params_count() const
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
	const std::size_t get_param_index(std::string param_name)
	{
		for (std::size_t i = 0; i < this->m_parameters.size(); ++i)
		{
			if (this->m_parameters[i] == param_name)
				return i;
		}
	}
};

template <class T>
class Mathexpr
{
public:
	Mathexpr(Header<T> funcHeader, std::list<std::shared_ptr<IToken<T>>> funcBody) :header(funcHeader), body(funcBody) {}
	Header<T>& get_header() const
	{
		return header;
	}
	std::list<std::shared_ptr<IToken<T>>>& get_body()
	{
		return body;
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
				int idx = header.get_param_index(std::string(var->get_name(), var->get_name_length()));
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
#endif // !TOKENS_H