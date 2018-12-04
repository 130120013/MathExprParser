//#include <iostream> - what for?
#include <queue> //to store arguments
#include <stdexcept> //for exceptions
#include <memory>

struct invalid_exception:std::exception
{
	invalid_exception() = default;
	invalid_exception(const char* decription);
};

struct bad_expession_parameters:std::exception
{
	bad_expession_parameters() = default;
	bad_expession_parameters(const char* decription);
};

//template <class T>
//class IToken;
//
//template <class T>
//T compute_token(const IToken<T>& tkn)
//{
//	return tkn();
//}
//
//template <class T>
//T compute_token(const T& tkn)
//{
//	return tkn;
//}

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
};

template <class T>
class Number : public IToken<T>
{
public:
	Number(T val) : value(val) {};
	//Number(const Number<T>& num) : value(num()) {}; /*-*/
	//Better:
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
protected:
private:
	T value;
};

template <class T>
class Operator : public IToken<T>
{
	//std::queue<T> m_parameters;
public:
	//virtual T operator()()  = 0;/*Implementation of IToken<T>::operator()()*/

	/*If this form is defined, then it will hide the "virtual T operator()()" overload, unless that form is also explicitly declared even as pure virtual*/
	//T operator()(const Number<T> a, const Number<T> b)
	//{
	//	return //a() - b();
	//		;
	//}
	/*virtual void push_argument(T value)
	{
		m_parameters.push(value);
	}*/
	virtual short getPriority()
	{
		return 0; //default priority, less code but more error prone
	}
	/*virtual std::size_t get_params_count() const
	{
		return m_parameters.size();
	}*/
protected:
	/*std::queue<T>& parameter_queue()
	{
		return m_parameters;
	}
	const std::queue<T>& parameter_queue() const
	{
		return m_parameters;
	}*/
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
		if (dynamic_cast<Number<T>*>(ops[0].get()) != nullptr && dynamic_cast<Number<T>*>(ops[1].get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(ops[0].get());
			const Number<T>* k2 = dynamic_cast<Number<T>*>(ops[1].get());

			return *k1 / *k2;
		}

		return 0;
	}
	virtual bool is_ready() const
	{
		return top == &ops[2] && ops[0]->is_ready() && ops[1]->is_ready();
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
	}
	/*T operator()(const Number<T> a, const Number<T> b)
	{
		return a() + b();
	}*/
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
		if (dynamic_cast<Number<T>*>(ops[0].get()) != nullptr && dynamic_cast<Number<T>*>(ops[1].get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(ops[0].get());
			const Number<T>* k2 = dynamic_cast<Number<T>*>(ops[1].get());

			return *k1 - *k2;
		}

		return 0;
	}
	virtual bool is_ready() const
	{
		return true;//this->parameter_queue().size() == 2;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
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
		if (dynamic_cast<Number<T>*>(ops[0].get()) != nullptr && dynamic_cast<Number<T>*>(ops[1].get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(ops[0].get());
			const Number<T>* k2 = dynamic_cast<Number<T>*>(ops[1].get());

			return *k1 * *k2;
		}

		return 0;
	}
	virtual bool is_ready() const
	{
		return true;//this->parameter_queue().size() == 2;
	}
	virtual short getPriority()
	{
		return 1;
	}
	virtual std::size_t get_params_count() const
	{
		return 2;
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
		if (dynamic_cast<Number<T>*>(ops[0].get()) != nullptr && dynamic_cast<Number<T>*>(ops[1].get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(ops[0].get());
			const Number<T>* k2 = dynamic_cast<Number<T>*>(ops[1].get());

			return *k1 / *k2;
		}

		return 0;
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
};

template <class T>
class Function : public IToken<T> //sin,cos...
{
	std::queue<std::shared_ptr<IToken<T>>> m_parameters;
	char* function_name;
public:
	virtual T operator()()
	{
		return m_parameters.front().get()->operator()();
	}
	virtual bool is_ready() const 
	{
		return true; //is it needed function?
	}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		m_parameters.push(value);
	}
	virtual std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	virtual const char* get_function_name() const
	{
		return function_name;
	}
protected:
	std::queue<T>& parameter_queue()
	{
		return m_parameters;
	}
	const std::queue<std::shared_ptr<IToken<T>>>& parameter_queue() const
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
		if (dynamic_cast<Number<T>*>(op.get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(op.get());

			return std::sin(k1->operator()());
		} 
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 1;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "sin";
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
		if (dynamic_cast<Number<T>*>(op.get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(op.get());

			return std::cos(k1->operator()());
		}
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 1;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "cos";
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
		if (dynamic_cast<Number<T>*>(op.get()) != nullptr)
		{
			const Number<T>* k1 = dynamic_cast<Number<T>*>(op.get());

			return std::tan(k1->operator()());
		}
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 1;
	}
	virtual short getPriority()
	{
		return -1;
	}
	virtual std::size_t get_params_count() const
	{
		return 1;
	}
	virtual const char* get_function_name() const
	{
		return "tg";
	}
};

template <class T>
class Bracket : public Operator<T> //,' '()
{
	//T openingBracket;
public:
	//Bracket(const T isOpeningBracket) : openingBracket(isOpeningBracket) {};
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
};

template <class T>
class Header : public IToken<T> //sin,cos...
{
	std::queue<std::shared_ptr<IToken<T>>> m_parameters;
	std::unique_ptr<char> function_name;
	std::size_t function_name_length = 0;
	bool isReady = false;
public:
	Header() = default;
	Header(char* varname, std::size_t len) : function_name_length(len)
	{
		function_name.reset(varname);
	}
	Header(Header<T>&& val) : function_name_length(val.get_name_length())
	{
		function_name.reset(std::move(varname));
	}
	virtual T operator()() const
	{
		return m_parameters.size();
	}
	virtual bool is_ready() const
	{
		return true; //is it needed function?
	}
	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		m_parameters.push(value);
	}
	virtual std::size_t get_params_count() const
	{
		return m_parameters.size();
	}
	virtual char* get_function_name() const
	{
		return function_name.get();
	}
	virtual void set_function_name(char* name)
	{
		function_name.reset(name);
	}
	size_t get_name_length()
	{
		return function_name_length;
	}
protected:
	std::queue<T>& parameter_queue()
	{
		return m_parameters;
	}
	const std::queue<T>& parameter_queue() const
	{
		return m_parameters;
	}
};

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{
	T op = 0;
	std::unique_ptr<char> name;
	std::size_t name_length = 0;
	bool isReady = false;
public:
	Variable(char* varname, std::size_t len, T value = 0) : op(value), name_length(len) 
	{
		name.reset(varname);
		isReady = true;
	}
	Variable(Variable<T>&& val) : op(val()), name_length(val.get_name_length())
	{
		name.reset(val.get_name());
		isReady = true;
	}

	virtual void push_argument(const std::shared_ptr<IToken<T>>& value)
	{
		if (isReady)
			throw std::invalid_argument("ERROR!");
		//op = value;
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
};