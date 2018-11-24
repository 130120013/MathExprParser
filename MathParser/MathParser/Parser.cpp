#include <iostream> 
#include <queue>


//template <class Iterator> 
void lex(const char* expr, const int length, double* number)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	*number = 0;
	std::queue<IToken> operationQueue;
	std::queue<IToken> outputQueue;

	while (*endPtr != NULL) //если NULL, значит, что strtod достиг конца строки
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

			if (*begPtr >= '0' || *begPtr <= '9') //если число class Number
			{
				outputQueue.push(std::strtod(begPtr, &endPtr));

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
					//если следующий токен - число, то унарный
					//необходимо ввести счетчик операторов? 
				}
			}
		}
		catch (std::exception e)
		{
			throw std::invalid_argument("ERROR!");
		}
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