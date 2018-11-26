#include <iostream> 
#include <queue>
#include <stack>
#include "SortStation.cpp"
#include <cstdlib>
#include <memory>

template <class T>
std::shared_ptr<IToken<T>> parse_token(const char* input_string, char** endptr) //���� ��� ������ ��� ����� � ����������
{
	if(*input_string >= '0' || *input_string <= '9')
		return Number<T>(std::strtod(input_string, endptr));
	if (*begPtr == '+')
		return OperatorPlus<T>();
	if (*begPtr == '-')
		return OperatorMinus<T>();
	if (*begPtr == '*')
		return OperatorMul<T>();
	if (*begPtr == '/')
		return OperatorDiv<T>();

	return NULL;
}

//template <class Iterator> 
void lex(const char* expr, const int length, double* number)
{
	char* endPtr = (char*)(expr + length - 1);
	char* begPtr = (char*)(expr + 0);
	auto it = expr;
	short hasPunct = 0;
	*number = 0;
	std::stack<std::shared_ptr<IToken<double>>> operationQueue;
	std::queue<std::shared_ptr<IToken<double>>> outputQueue;

	while (*endPtr != NULL || begPtr != expr + length) //���� NULL, ������, ��� ������ ����� ������
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
			//�� ������ ������ ��������� ������ �������

			if (*begPtr >= '0' || *begPtr <= '9') //���� ����� class Number
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

				if (*begPtr == '+') //���� ��������� �������� � ������� +
				{
					if (OperatorPlus<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(parse_token<double>(begPtr, &endPtr));

					begPtr += 1;
					//���� ��������� ����� - �����, �� �������
					//���������� ������ ������� ����������? 
				}
				if (*begPtr == '-')
				{
					if (OperatorMinus<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(parse_token<double>(begPtr, &endPtr));

					begPtr += 1;
				}
				if (*begPtr == '*')
				{
					if (OperatorMul<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(parse_token<double>(begPtr, &endPtr));

					begPtr += 1;
				}
				if (*begPtr == '/')
				{
					if (OperatorDiv<double>().getPriority() < dynamic_cast<Operator<double>*>(operationQueue.top().get())->getPriority())
						outputQueue.push(operationQueue.top());
					else
						operationQueue.push(parse_token<double>(begPtr, &endPtr));

					begPtr += 1;
				}
				//if (*begPtr == ',')
				//{
				//	bool isOpeningBracket = false;
				//	while (!isOpeningBracket) 
				//	{
				//		if (operationQueue.top() != '(') //���� �������������� ����� ��������� �������
				//		{
				//			outputQueue.push(operationQueue.pop());
				//		}
				//		else
				//		{
				//			isOpeningBracket = true;
				//		}
				//	}
				//	if(!isOpeningBracket)
				//		throw std::invalid_argument("ERROR!");
				//	begPtr += 1;
				//}
				//if (*begPtr == '(')
				//{
				//	operationQueue.push(Delimiter<double>());
				//	begPtr += 1;
				//}
				//if (*begPtr == ')')
				//{
				//	bool isOpeningBracket = false;
				//	while (!isOpeningBracket)
				//	{
				//		if (operationQueue.top() != '(') 
				//		{
				//			outputQueue.push(operationQueue.pop());
				//		}
				//		else
				//		{
				//			isOpeningBracket = true;
				//		}
				//	}
				//	if (!isOpeningBracket)
				//		throw std::invalid_argument("ERROR!");
				//	else
				//		operationQueue.pop();
				//	if (operationQueue.top() == "sin") //���� �������
				//		outputQueue.push(Operator<double>());
				//	begPtr += 1;
				//}

			}
		}
		catch (std::exception e)
		{
			throw std::invalid_argument("ERROR!");
		}
	}
	while (operationQueue.size() != 0)
	{
		//if (operationQueue.pop() == '(' || operationQueue.pop() == ')') //���� ������
		//	throw std::invalid_argument("ERROR!");
		//else
			outputQueue.push(operationQueue.top());
			operationQueue.pop();
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