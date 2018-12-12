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

	Mathexpr<double> mathexpr(header, lexBody<double>("7^2+sin(x)/2.2-1", 16, header.get_params_vector()));
	std::vector<double> v;
	v.push_back(1);
	mathexpr.init_variables(v);
	auto res = simplify<double>(mathexpr.get_body());
	v[0] = 2;
	mathexpr.clear_variables();
	mathexpr.init_variables(v);
	res = simplify<double>(mathexpr.get_body());
	return 0;
}