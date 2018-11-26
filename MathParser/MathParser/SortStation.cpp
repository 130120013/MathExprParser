//#include <iostream> - what for?
#include <queue> //to store arguments
#include <stdexcept> //for exceptions

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

template <class T>
class IToken
{
public:
	virtual T operator()() = 0; //All derived classes must implement the same method with the same return type
	/*Push a required (as defined at runtime, perhaps erroneously which must be detected by implementations) number of arguments.
	Push arguments from first to last. Then call the operator() above.
	*/
	virtual void push_argument(T value) = 0;
	virtual bool is_ready() const = 0; //all parameters are specified

//	virtual T action() = 0;
	virtual ~IToken() {} //virtual d-tor is to allow correct destruction of polymorphic objects
};

template <class T>
class Number : public IToken<T>
{
public:
	Number(T val) : value(val) {};
	//Number(const Number<T>& num) : value(num()) {}; /*-*/
	//Better:
	Number(const Number<T>&) = default;
	T operator()()
	{
		return value;
	}
	bool is_ready() const
	{
		return true;
	}

	void push_argument(T)
	{
		//do nothing for literals - they do not accept parameters. But the implementation (even empty) must be provided for polymorphic methods.
		//or throw an exception
#ifndef __CUDACC__
		throw bad_expession_parameters("An argument is specified for a literal");
#endif
	}
protected:
private:
	T value;
};

template <class T>
class Operator : public IToken<T>
{
	std::queue<T> m_parameters;
public:
	//virtual T operator()()  = 0;/*Implementation of IToken<T>::operator()()*/

	/*If this form is defined, then it will hide the "virtual T operator()()" overload, unless that form is also explicitly declared even as pure virtual*/
	//T operator()(const Number<T> a, const Number<T> b)
	//{
	//	return //a() - b();
	//		;
	//}
	virtual void push_argument(T value)
	{
		m_parameters.push(value);
	}
	virtual short getPriority()
	{
		return 0; //default priority, less code but more error prone
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

template <class T = double>
class OperatorPlus : public Operator<T> //+-*/
{
public:
	virtual T operator()()/*Implementation of IToken<T>::operator()()*/
	{
		auto result = T(); //zero-initialization
		for (auto arg&:this->parameter_queue())
			result += arg;
		return arg;
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 2;
	}
	/*T operator()(const Number<T> a, const Number<T> b)
	{
		return a() + b();
	}*/
};
template <class T>
class OperatorMinus : public Operator<T>
{
public:
	virtual T operator()()/*Implementation of IToken<T>::operator()()*/
	{
		auto result = T(); //zero-initialization
		for (auto arg& : this->parameter_queue())
			result += arg;
		return arg;
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 2;
	}
};
template <class T>
class OperatorMul : public Operator<T>
{
public:
	virtual T operator()()/*Implementation of IToken<T>::operator()()*/
	{
		auto result = T(); //zero-initialization
		for (auto arg& : this->parameter_queue())
			result += arg;
		return arg;
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 2;
	}
	virtual short getPriority()
	{
		return 1;
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
public:
	virtual T operator()()/*Implementation of IToken<T>::operator()()*/
	{
		auto result = T(); //zero-initialization
		for (auto& arg : this->parameter_queue())
			result += arg;
		return arg;
	}
	virtual bool is_ready() const
	{
		return this->parameter_queue().size() == 2;
	}
	virtual short getPriority()
	{
		return 1;
	}
};

template <class T>
class Function : public IToken<T> //sin,cos...
{

};

template <class T>
class Delimiter : public IToken<T> //,' '()
{

};

template <class T>
class Variable : public IToken<T> //arguments of Header, e.g. F(x) x - Variable
{

};