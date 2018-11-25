#include <iostream>

template <class T>
class IToken
{
//public:
//	virtual T action() = 0;
};

template <class T = double>
class Number : public IToken<T>
{
public:
	Number(T val) : value(val) {};
	Number(Number<T> num) : value(num()) {};
	T operator()()
	{
		return value;
	}

private:
	T value;
};

template <class T>
class Operator : public IToken<T>
{
public:
	T operator()(const Number<T> a, const Number<T> b)
	{
		return //a() - b();
			;
	}
	short getPriority()
	{
		return priority;
	}
private:
	short priority; 
};

template <class T = double>
class OperatorPlus : public Operator<T> //+-*/
{
public:
	Number<T> operator()(const Number<T> a, const Number<T> b)
	{
		return a() + b();
	}
};
template <class T>
class OperatorMinus : public Operator<T>
{
public:
	T operator()(const Number<T> a, const Number<T> b)
	{
		return a() - b();
	}
};
template <class T>
class OperatorMul : public Operator<T>
{
public:
	Number<T> operator()(const Number<T> a, const Number<T> b)
	{
		return a() * b();
	}
};
template <class T>
class OperatorDiv : public Operator<T>
{
public:
	Number<T> operator()(const Number<T> a, const Number<T> b)
	{
		return a() / b();
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
class Variable : public IToken //arguments of Header, e.g. F(x) x - Variable
{

};