#include "Parser.h"

int main()
{
	const char* func = "f(x) = 7 * 7 + 3";
	int length = 11;
	//lex("7 + sin(6)", length);
	const char funcHeader[] = "f(x) = ";
	int length1 = 4;
	char *endptr;
	Header<double> header(funcHeader, sizeof(funcHeader) - 1, &endptr);
	std::cout << header.get_function_name();

	Mathexpr<double> mathexpr(header, lexBody<double>("7*sin(1)/2", 10, header.get_params_vector()));
	simplify<double>(mathexpr.get_body());

	return 0;
}