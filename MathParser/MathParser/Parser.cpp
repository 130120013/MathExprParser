#include <iostream> 
#include <queue>
#include <stack>
#include "SortStation.cpp"


//template <class Iterator> 
void lex(const char* expr, const int length, double* number)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	*number = 0;
	std::stack<IToken<double>> operationQueue;
	std::queue<IToken<double>> outputQueue;

	while (*endPtr != NULL || begPtr != expr + length) //если NULL, значит, что достиг конца строки
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

			if (*begPtr >= '0' || *begPtr <= '9') //если число class Number
			{
				outputQueue.push(Number<double>(std::strtod(begPtr, &endPtr)));

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

				if (*begPtr == '+') //надо различать бинарный и унарный +
				{
					if (OperatorPlus<double>().getPriority() < operationQueue.top().getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(OperatorPlus<double>());

					begPtr += 1;
					//если следующий токен - число, то унарный
					//необходимо ввести счетчик операторов? 
				}
				if (*begPtr == '-')
				{
					if (OperatorMinus<double>().getPriority() < operationQueue.top().getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(OperatorMinus<double>());

					begPtr += 1;
				}
				if (*begPtr == '*')
				{
					if (OperatorMul<double>().getPriority() < operationQueue.top().getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(OperatorMul<double>());

					begPtr += 1;
				}
				if (*begPtr == '/')
				{
					if (OperatorDiv<double>().getPriority() < operationQueue.top().getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(OperatorDiv<double>());

					begPtr += 1;
				}
				if (*begPtr == ',')
				{
					bool isOpeningBracket = false;
					while (!isOpeningBracket) 
					{
						if (operationQueue.top() != '(') //надо спроектировать метод получения символа
						{
							outputQueue.push(operationQueue.pop());
						}
						else
						{
							isOpeningBracket = true;
						}
					}
					if(!isOpeningBracket)
						throw std::invalid_argument("ERROR!");
					begPtr += 1;
				}
				if (*begPtr == '(')
				{
					operationQueue.push(Delimiter<double>());
					begPtr += 1;
				}
				if (*begPtr == ')')
				{
					bool isOpeningBracket = false;
					while (!isOpeningBracket)
					{
						if (operationQueue.top() != '(') 
						{
							outputQueue.push(operationQueue.pop());
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
					if (operationQueue.top() == "sin") //если функция
						outputQueue.push(Operator<double>());
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
		if (operationQueue.pop() == '(' || operationQueue.pop() == ')') //если скобка
			throw std::invalid_argument("ERROR!");
		else
			outputQueue.push(operationQueue.pop());
	}
}

int main()
{
	char* endptr;
	double number = std::strtod("123.5c 5", &endptr);
	int length = 11;
	lex("123.5- 56.6", length, &number);

	return 0;
} 