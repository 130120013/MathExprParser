#include "Parser.h"
#include <iostream>

/*
std::string expression = "f(x, y) = x * y"; - PASSED
std::string expression = "f(x <-, y, z) = x * y + z"; - PASSED
std::string expression = "f(x, y <-, z) = x * y + z"; - PASSED
std::string expression = "f(x, y, z <-) = x * y + z"; - PASSED 
std::string expression = "f(x <-, y) = x / y"; - PASSED
std::string expression = "f(x, y <-) = x / y"; - PASSED

*/


void transform_test()
{
	std::string expression = "f(x, y, z) = x / y - z";
	Mathexpr<double> mathexpr = { expression };
	auto result = mathexpr.transformation("z", 1);

	
	/*auto type = (result.get())->type();
	auto variableType = type == TokenType::variable;*/
}

void compute_test() 
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

	std::string expression = "f(x, y) = x * y";
	Mathexpr<double> mathexpr = { expression };

	std::vector<double> v;
	v.push_back(2);
	v.push_back(10);
	mathexpr.init_variables(v);
	std::cout << "Value: " << mathexpr.compute() << "\n";
}

int main()
{
	transform_test();

	return 0;
}