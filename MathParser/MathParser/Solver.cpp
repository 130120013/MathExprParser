#include "Parser.h"

template <typename T>
void simplify(std::list<std::shared_ptr<IToken<T>>>& body)
{
	auto it = body.begin();

	while (body.size() > 1)//(it != body.end())
	{
		//need to implement typeid
		if (dynamic_cast<Operator<T>*>((*it).get())) //+*-/ or sin/cos/tg
		{
			auto val = dynamic_cast<OperatorMul<T>*>((*it).get());
			bool isComputable = false;
			int paramsCount = val->get_params_count();

			for (int i = paramsCount; i > 0; --i)
			{
				auto param_it = it;
				std::advance(param_it, -1 * i);
				((*it).get())->push_argument(*param_it); //here std::move must be
				body.remove(*param_it);
			}

			if (val->is_ready())
			{
				T res = val->operator()();
				auto calc = std::make_shared<Number<T>>(res);
				*it = calc;
			}
			++it;
			continue;
		}
		if (dynamic_cast<Function<T>*>((*it).get()))
		{
			auto val = dynamic_cast<SinFunction<T>*>((*it).get());
			int paramsCount = val->get_params_count();

			for (int i = paramsCount; i > 0; --i)
			{
				auto param_it = it;
				std::advance(param_it, -1 * i);
				((*it).get())->push_argument(*param_it); //here std::move must be
				body.remove(*param_it);
			}

			if (val->is_ready())
			{
				T res = val->operator()();
				auto calc = std::make_shared<Number<T>>(res);
				*it = calc;
			}
			++it;
			continue;
		}
		++it;
	}
}

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

	Mathexpr<double> mathexpr(header, lexBody<double>("7*sin(1)", 8, header.get_params_vector()));
	simplify<double>(mathexpr.get_body());

	return 0;
}