#include <utility>
#include <cctype>
#include <iostream>
#include <cstring>

char* substr(const char* arr, int begin, int len)
{
	char* res = new char[len];
	for (int i = 0; i < len; i++)
		res[i] = *(arr + begin + i);
	res[len] = 0;
	return res;
}
//template <class Iterator>
//std::pair<int, int>
void lex(const char* expr, const int length, double* number)
{
	auto res = std::make_pair(-1, -1);
	char* endPtr = 0;
	auto it = expr;
	short hasPunct = 0;
	*number = 0;

	while (it != expr + length)
	{
		try
		{
			if (*it == '.')
			{
				hasPunct += 1;
				if (hasPunct == 2)
				{
					throw std::invalid_argument("dublicated dot");
				}			
			}
			else
			{

			}

			*number = std::strtod(it, &endPtr);
			//beginPos = i + 1;
			std::cout << "Num = " << *number << "\n";
			hasPunct = 0;
			it = endPtr;
		}
		catch (...)
		{
			std::cout << "Error";
		}
	}
	
	//for (auto i = expr; i != expr + length; ++i)
	//{
	//	if (isdigit(expr[i]) || expr[i] == '.')
	//	{
	//		if (expr[i] == '.')
	//		{
	//			hasPunct += 1;
	//			if (hasPunct == 2)
	//			{
	//				std::cout << "Error";
	//				return;
	//			}

	//		}
	//			

	//		if ((!isdigit(expr[i + 1])) && expr[i + 1] != '.')
	//		{
	//			*number = std::strtod(substr(expr, beginPos, i + 1 - beginPos), &endPtr);
	//				//beginPos = i + 1;
	//				std::cout << "Num = " << *number << "\n";
	//				hasPunct = 0;
	//		}

	//	}
	//	else
	//	{
	//		beginPos = i;
	//	}
	//}
}

int main()
{
	char* endptr;
	double number = std::strtod("123.5c 5", &endptr);
	std::cout << number << "\n";
	int length = 10;
	//auto res = 
		lex(" 123.5 c 5", length, &number);
	//std::cout << "First = " << res.first << ", Second = " << res.second << ", Number = " << *number;
	/*while (res.second != length - 1)
	{

	}*/

	return 0;
}