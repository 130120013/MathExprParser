#include <iostream>

template <class T>
class IToken
{
public:
	virtual T action() = 0;
};

template <class T = double>
class Number : public IToken
{
public:
	Number(T val) : value(val) {};
	T action()
	{
		return value;
	}

private:
	T value;
};

template <class T>
class Operator : public IToken //+-*/
{
public:
	T action()
	{

	}
private:


};

template <class T>
class Function : public IToken //sin,cos...
{

};

template <class T>
class Delimiter : public IToken //,' '()
{

};

template <class T>
class Variable : public IToken //arguments of Header, e.g. F(x) x - Variable
{

};