#include <iostream> 
#include <queue>
#include <stack>
#include "SortStation.cpp"
#include <cstdlib>
#include <memory>

template <class T>
std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr) //пока что только для чисел и операторов
{
	if(*input_string >= '0' || *input_string <= '9')
		return std::make_shared<Number<T>>(std::strtod(input_string, endptr));
	if (*input_string == '+')
		return std::make_shared<OperatorPlus<T>>();
	if (*input_string == '-')
		return std::make_shared<OperatorMinus<T>>();
	if (*input_string == '*')
		return std::make_shared<OperatorMul<T>>();
	if (*input_string == '/')
		return std::make_shared<OperatorDiv<T>>();

	return NULL;
}

//template <class Iterator> 
std::queue<std::shared_ptr<IToken<double>>> lex(const char* expr, const int length, double* number)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	*number = 0;
	std::stack<std::shared_ptr<IToken<double>>> operationQueue;
	std::queue<std::shared_ptr<IToken<double>>> outputQueue;

	while (*begPtr != NULL || begPtr != expr + length) //если NULL, значит, что достиг конца строки
	{
		try
		{
			/*	if (*it == '.')
				{
					hasPunct += 1;
					if (hasPunct == 2)
					{
						throw std::invalid_argument("dublicated dot");
					}
				}
			 */
			//не описан случай вхождения токена функции


			if (*begPtr >= '0' && *begPtr <= '9') //если число class Number
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

				if (*begPtr == '+')
				{
					char tok = *(begPtr + 1);
					if (tok >= '0' && tok <= '9') //унарный плюс
					{
						outputQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //бинарный плюс
					{
						if (operationQueue.size() != 0 && OperatorPlus<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						else
							operationQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '-')
				{
					char tok = *(begPtr + 1);
					if (tok >= '0' && tok <= '9') //унарный минус
					{
						outputQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr = endPtr;
					}
					else //бинарный минус
					{
						if (operationQueue.size() != 0 && OperatorMinus<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						else
							operationQueue.push(parse_token<double>(begPtr, &endPtr));
						begPtr += 1;
					}
				}
				if (*begPtr == '*')
				{
					if (operationQueue.size() != 0 && OperatorMul<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
					{
						outputQueue.push(operationQueue.top());
						operationQueue.pop();
					}
					else
						operationQueue.push(parse_token<double>(begPtr, &endPtr));
					begPtr += 1;
				}
				if (*begPtr == '/')
				{
					if (operationQueue.size() != 0 && OperatorDiv<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
					{
						outputQueue.push(operationQueue.top());
						operationQueue.pop();
					}
					else
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
					while (!isOpeningBracket || operationQueue.size() != 0)
					{
						if (dynamic_cast<Bracket<bool>*>(operationQueue.top().get()) == NULL)
						{
							outputQueue.push(operationQueue.top());
							operationQueue.pop();
						}
						else
						{
							isOpeningBracket = true;
						}
					}
					if (!isOpeningBracket)
						throw std::invalid_argument("ERROR!");
					else
						operationQueue.pop();
					//if (operationQueue.top() == "sin") //если функция
					//	outputQueue.push(Operator<double>());
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
		//if (operationQueue.pop() == '(' || operationQueue.pop() == ')') //если скобка
		//	throw std::invalid_argument("ERROR!");
		//else
			outputQueue.push(operationQueue.top());
			operationQueue.pop();
	}
	return outputQueue;
}

int main()
{
	char* endptr;
	double number = std::strtod("123.5c 5", &endptr);
	int length = 9;
	lex("4 - 8 - 9", length, &number);

	return 0;
} 