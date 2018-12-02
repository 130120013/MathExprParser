#include <iostream> 
#include <queue>
#include <stack>
#include "SortStation.cpp"
#include <cstdlib>
#include <cstring>
#include <memory>

template <class T>
std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr) //пока что только для чисел и операторов
{
	if(*input_string >= '0' && *input_string <= '9')
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

//template <class Iterator> 
std::queue<std::shared_ptr<IToken<double>>> lex(const char* expr, const int length)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	std::stack<std::shared_ptr<IToken<double>>> operationQueue;
	std::queue<std::shared_ptr<IToken<double>>> outputQueue;

	while (*begPtr != NULL || *begPtr != '\0' || begPtr != expr + length) 
	{
		try
		{
			if (*begPtr >= '0' && *begPtr <= '9')
			{
				outputQueue.push(parse_token<double>(begPtr, &endPtr));

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
				if (*begPtr == 's') //sin
				{
					auto funcSin = parse_token<double>(begPtr, &endPtr);
					if(funcSin != NULL)
						operationQueue.push(funcSin);
					else 
						throw std::invalid_argument("ERROR!");
					begPtr += 3;
				}
				if (*begPtr == 'c') //cos
				{
					auto funcCos = parse_token<double>(begPtr, &endPtr);
					if (funcCos != NULL)
						operationQueue.push(funcCos);
					else
						throw std::invalid_argument("ERROR!");
					begPtr += 3;
				}
				if (*begPtr == 't') //tg
				{
					auto funcTg = parse_token<double>(begPtr, &endPtr);
					if (funcTg != NULL)
						operationQueue.push(funcTg);
					else
						throw std::invalid_argument("ERROR!");
					begPtr += 2;
				}

				if (*begPtr == '+')
				{
					char tok = *(begPtr + 1);
					if (tok >= '0' && tok <= '9') //unary +
					{
						outputQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary +
					{
						if (operationQueue.size() != 0 && OperatorPlus<double>().getPriority() <= dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						operationQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '-')
				{
					char tok = *(begPtr + 1);
					if (tok >= '0' && tok <= '9') //unary -
					{
						outputQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //binary -
					{
						if (operationQueue.size() != 0 && OperatorMinus<double>().getPriority() <= dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						operationQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '*')
				{
					if (operationQueue.size() != 0 && OperatorMul<double>().getPriority() <= dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
					{
						outputQueue.push(operationQueue.top());
						operationQueue.pop();
					}
					operationQueue.push(parse_token<double>(begPtr, &endPtr));
					begPtr += 1;
				}
				if (*begPtr == '/')
				{
					if (operationQueue.size() != 0 && OperatorDiv<double>().getPriority() <= dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
					{
						outputQueue.push(operationQueue.top());
						operationQueue.pop();
					}
					operationQueue.push(parse_token<double>(begPtr, &endPtr));
					begPtr += 1;
				}
				if (*begPtr == ',')
				{
					bool isOpeningBracket = false;
					while (!isOpeningBracket || operationQueue.size() != 0) //while an opening bracket is not found or an operation stack is not empty
					{
						if (dynamic_cast<Bracket<bool>*>(operationQueue.top().get()) == NULL) //if the cast to Bracket is not successfull, return NULL => it is not '('  
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						else
						{
							isOpeningBracket = true;
						}
					}
					if(!isOpeningBracket) //missing '('
						throw std::invalid_argument("ERROR!");
					begPtr += 1;
				}
				if (*begPtr == '(')
				{
					operationQueue.push(std::make_shared<Bracket<double>>());
					begPtr += 1;
				}
				if (*begPtr == ')')
				{
					bool isOpeningBracket = false;
					while (operationQueue.size() != 0)
					{
						if (operationQueue.size() != 0 && dynamic_cast<Bracket<double>*>(operationQueue.top().get()) == NULL)
						{
							outputQueue.push(operationQueue.top());
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
		if (dynamic_cast<Bracket<double>*>(operationQueue.top().get()) != NULL) //checking enclosing brackets
			throw std::invalid_argument("ERROR!");
		else
		{
			outputQueue.push(operationQueue.top());
			operationQueue.pop();
		}
	}
	return outputQueue;
}

int main()
{
	int length = 11;
	lex("-7 + sin(6)", length);

	return 0;
} 