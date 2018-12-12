#ifndef PARSER_H
#define PARSER_H

#include <iostream> 
#include <stack>
#include "Tokens.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <list>
#include <algorithm>

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
std::list<std::shared_ptr<IToken<T>>> lexBody(const char* expr, const int length, const std::vector<std::string>& m_parameters)
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
		int paramsCount = val->get_params_count();

		for (int i = paramsCount; i > 0; --i)
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
std::list<std::shared_ptr<IToken<T>>> simplify(std::list<std::shared_ptr<IToken<T>>> body)
{
	auto it = body.begin();

	while (body.size() > 1 )
	{
		std::string type = (*it).get()->type();
		if (type == "plus")
		{
			compute<OperatorPlus<T>, T>(body, it);
			continue;
		}
		if (type == "minus")
		{
			compute<OperatorMinus<T>, T>(body, it);
			continue;
		}
		if (type == "mul")
		{
			compute<OperatorMul<T>, T>(body, it);
			continue;
		}
		if (type == "div")
		{
			compute<OperatorDiv<T>, T>(body, it);
			continue;
		}
		if (type == "pow")
		{
			compute<OperatorPow<T>, T>(body, it);
			continue;
		}
		if (type == "sin")
		{
			compute<SinFunction<T>, T>(body, it);
			continue;
		}
		if (type == "cos")
		{
			compute<CosFunction<T>, T>(body, it);
			continue;
		}
		if (type == "tg")
		{
			compute<TgFunction<T>, T>(body, it);
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
	return body;
}

#endif // !PARSER_H
