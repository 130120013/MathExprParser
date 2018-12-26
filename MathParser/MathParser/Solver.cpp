#include "Parser.h"
#include <iostream>

int main()
{
	/*const char funcHeader[] = "f(x) = ";
	int length1 = 4;
	char *endptr;
	Header<double> header(funcHeader, sizeof(funcHeader) - 1, &endptr);

	Mathexpr<double> mathexpr(header, lexBody<double>("sin() + 5", 9, header.get_params_vector()));
	std::vector<double> v;
	v.push_back(1);
	mathexpr.init_variables(v);
	auto res = simplify(mathexpr.get_body());
	v[0] = 2;
	mathexpr.clear_variables();
	mathexpr.init_variables(v);
	res = simplify(mathexpr.get_body());*/
	/*
	FAILS:
	1. "f(x) = -x" - passed
	2. "f(x) = +x" - passed
	3. "f(x) = 5 - -x" - passed
	4. "f(x) = 5 - +x" - passed
	5. "f(x) = 5 - x" - passed
	6. "f(x) = 5 + x" - passed
	8. "f(x) = x + -5" - passed
	10. "f(x) = x + +5" - passed
	*/
	std::string expression = "f(x) = sin(3.14/6) + min(10, 2, 51, 4)";
	Mathexpr<double> mathexpr = {expression};
	std::vector<double> v;
	v.push_back(1);
	mathexpr.init_variables(v);
	std::cout << "Value: " << mathexpr.compute() << "\n";

	return 0;
}