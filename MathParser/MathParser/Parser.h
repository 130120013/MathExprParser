#ifndef PARSER_H
#define PARSER_H

#include <iostream> 
#include <queue>
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
		char tok = *(input_string + 1);
		if (tok >= '0' && tok <= '9')
			return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
		return std::make_shared<OperatorPlus<T>>();
	}

	if (*input_string == '-')
	{
		char tok = *(input_string + 1);
		if (tok >= '0' && tok <= '9')
			return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
		return std::make_shared<OperatorMinus<T>>();
	}

	if (*input_string == '*')
		return std::make_shared<OperatorMul<T>>();
	if (*input_string == '/')

		return std::make_shared<OperatorDiv<T>>();
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
	auto token = std::make_shared<Variable<T>>(tok_name, name_size, 0);
	*(endptr) = endTokPtr;
	return token;
}

template <class T>
std::list<std::shared_ptr<IToken<T>>> lexBody(const char* expr, const int length, const std::vector<std::string>& m_parameters)
{
	std::list<std::shared_ptr<IToken<T>>> output;

	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	std::stack<std::shared_ptr<IToken<T>>> operationQueue;

	while (*begPtr != NULL && *begPtr != '\0' && begPtr != expr + length)
	{
		try
		{
			if (*begPtr >= '0' && *begPtr <= '9')
			{
				output.push_back(parse_token<T>(begPtr, &endPtr));

				if (begPtr == endPtr)
					throw std::invalid_argument("ERROR!");
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
							throw std::invalid_argument("ERROR!");
						begPtr += 3;
						continue;
					}
					if (*begPtr == 'c') //cos
					{
						auto funcCos = parse_token<T>(begPtr, &endPtr);
						if (funcCos != NULL)
							operationQueue.push(funcCos);
						else
							throw std::invalid_argument("ERROR!");
						begPtr += 3;
						continue;
					}
					if (*begPtr == 't') //tg
					{
						auto funcTg = parse_token<T>(begPtr, &endPtr);
						if (funcTg != NULL)
							operationQueue.push(funcTg);
						else
							throw std::invalid_argument("ERROR!");
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
						throw std::invalid_argument("Error!");
				}
				if (*begPtr == '+')
				{

					skipSpaces(begPtr + 1, expr + length - begPtr - 1);
					char tok = *(begPtr + 1);
					if (*begPtr == '+' && tok >= '0' && tok <= '9') //unary +
					{
						output.push_back(parse_token<T>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary +
					{
						if (operationQueue.size() != 0 && OperatorPlus<T>().getPriority() <= dynamic_cast<Operator<T>*>(operationQueue.top().get())->getPriority())
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						operationQueue.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '-')
				{
					skipSpaces(begPtr + 1, expr + length - begPtr - 1);
					char tok = *(begPtr + 1);
					if (*begPtr == '-' && tok >= '0' && tok <= '9') //unary +
					{
						output.push_back(parse_token<T>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary -
					{
						if (operationQueue.size() != 0 && OperatorMinus<T>().getPriority() <= dynamic_cast<Operator<T>*>(operationQueue.top().get())->getPriority())
						{
							output.push_back(operationQueue.top());
							operationQueue.pop();
						}
						operationQueue.push(parse_token<T>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '*')
				{
					if (operationQueue.size() != 0 && OperatorMul<T>().getPriority() <= dynamic_cast<Operator<T>*>(operationQueue.top().get())->getPriority())
					{
						output.push_back(operationQueue.top());
						operationQueue.pop();
					}
					operationQueue.push(parse_token<T>(begPtr, &endPtr));
					begPtr += 1;
				}
				if (*begPtr == '/')
				{
					if (operationQueue.size() != 0 && OperatorDiv<T>().getPriority() <= dynamic_cast<Operator<T>*>(operationQueue.top().get())->getPriority())
					{
						output.push_back(operationQueue.top());
						operationQueue.pop();
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
						throw std::invalid_argument("ERROR!");
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
						throw std::invalid_argument("ERROR!");
					else
						operationQueue.pop();
					begPtr += 1;
				}
			}
		}
		catch (std::exception e)
		{
			throw std::invalid_argument("ERROR!");
		}
	}
	while (operationQueue.size() != 0)
	{
		if (dynamic_cast<Bracket<T>*>(operationQueue.top().get()) != NULL) //checking enclosing brackets
			throw std::invalid_argument("ERROR!");
		else
		{
			output.push_back(operationQueue.top());
			operationQueue.pop();
		}
	}
	return output;
}


#endif // !PARSER_H
