#include <utility> 
#include <cctype> 
#include <iostream> 
#include <cstring> 

//template <class Iterator> 
void lex(const char* expr, const int length, double* number)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	*number = 0;

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
			*number = std::strtod(begPtr, &endPtr);
			std::cout << "Num = " << *number << "\n"; 
			if (begPtr == endPtr)
				throw std::invalid_argument("ERROR!");
			begPtr = endPtr;
		}
		catch (std::exception& e)
		{
			throw std::invalid_argument("ERROR!");
		}
	} 
}

int main()
{
	char* endptr;
	double number = std::strtod("123.5c 5", &endptr);
	int length = 7;
	lex("123.5    - 5 6.6", length, &number);

	return 0;
} 