#include "Parser.h"

int main()
{
	const char funcHeader[] = "f(x) = ";
	int length1 = 4;
	char *endptr;
	Header<double> header(funcHeader, sizeof(funcHeader) - 1, &endptr);

	Mathexpr<double> mathexpr(header, lexBody<double>("sin() + 5", 9, header.get_params_vector()));
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