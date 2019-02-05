#include "cuda_config.cuh"
#include "cuda_return_wrapper.cuh"
#include "cuda_tokens.cuh"
#include "cuda_iterator.cuh"
#include "cuda_list.cuh"
#include "cuda_map.cuh"
#include "cuda_memory.cuh"
#include "cuda_stack.cuh"

#ifndef PARSER_H
#define PARSER_H
namespace cu {
	__device__ constexpr bool iswhitespace(char ch) noexcept
	{
		return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\0'; //see also std::isspace
	}
	//
	//__device__ constexpr bool isdigit(char ch) noexcept
	//{
	//	return ch >= '0' && ch <= '9';
	//};
	//
	//__device__ constexpr bool isalpha(char ch) noexcept
	//{
	//	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
	//};
	//
	//__device__ constexpr bool isalnum(char ch) noexcept
	//{
	//	return isdigit(ch) || isalpha(ch);
	//}

	template <class Iterator>
	__device__ Iterator skipSpaces(Iterator pBegin, Iterator pEnd) noexcept
	{
		auto pCurrent = pBegin;
		while (iswhitespace(*pCurrent) && ++pCurrent < pEnd);
		return pCurrent;
	}

	__device__ const char* skipSpaces(const char* input_string, std::size_t length) noexcept
	{
		return skipSpaces(input_string, input_string + length);
	}

	__device__ const char* skipSpaces(const char* input_string) noexcept
	{
		auto pCurrent = input_string;
		while (iswhitespace(*pCurrent) && *pCurrent++ != 0);
		return pCurrent;
	}

	__device__ inline char* skipSpaces(char* input_string, std::size_t length) noexcept
	{
		return const_cast<char*>(skipSpaces(const_cast<const char*>(input_string), length));
	}

	__device__ inline char* skipSpaces(char* input_string) noexcept
	{
		return const_cast<char*>(skipSpaces(const_cast<const char*>(input_string)));
	}

	struct token_string_entity
	{
		__device__ token_string_entity() = default;
		__device__ token_string_entity(const char* start_ptr, const char* end_ptr) :m_pStart(start_ptr), m_pEnd(end_ptr) {}

		//return a negative if the token < pString, 0 - if token == pString, a positive if token > pString
		__device__ int compare(const char* pString, std::size_t cbString) const noexcept
		{
			auto my_size = this->size();
			auto min_length = my_size < cbString ? my_size : cbString;
			for (std::size_t i = 0; i < min_length; ++i)
			{
				if (m_pStart[i] < pString[i])
					return -1;
				if (m_pStart[i] > pString[i])
					return -1;
			}
			return my_size < cbString ? -1 : (my_size > cbString ? 1 : 0);
		}
		__device__ inline int compare(const token_string_entity& right) const noexcept
		{
			return this->compare(right.begin(), right.size());
		}
		__device__ inline int compare(const char* pszString) const noexcept
		{
			return this->compare(pszString, cu::strlen(pszString));
		}
		template <std::size_t N>
		__device__ inline auto compare(const char(&strArray)[N]) const noexcept -> std::enable_if_t<N == 0, int>
		{
			return this->size() == 0;
		}
		template <std::size_t N>
		__device__ inline auto compare(const char(&strArray)[N]) const noexcept->std::enable_if_t<(N > 0), int>
		{
			return strArray[N - 1] == '\0' ?
				compare(strArray, N - 1) ://null terminated string, like "sin"
				compare(strArray, N); //generic array, like const char Array[] = {'s', 'i', 'n'}
		}
		__device__ inline const char* begin() const noexcept
		{
			return m_pStart;
		}
		__device__ inline const char* end() const noexcept
		{
			return m_pEnd;
		}
		__device__ inline std::size_t size() const noexcept
		{
			return std::size_t(this->end() - this->begin());
		}
	private:
		const char* m_pStart = nullptr, *m_pEnd = nullptr;
	};

	__device__ bool operator==(const token_string_entity& left, const token_string_entity& right) noexcept
	{
		return left.compare(right) == 0;
	}

	__device__ bool operator==(const token_string_entity& left, const char* pszRight) noexcept
	{
		return left.compare(pszRight) == 0;
	}

	template <std::size_t N>
	__device__ bool operator==(const token_string_entity& left, const char(&strArray)[N]) noexcept
	{
		return left.compare(strArray) == 0;
	}

	__device__ bool operator!=(const token_string_entity& left, const token_string_entity& right) noexcept
	{
		return left.compare(right) != 0;
	}

	__device__ bool operator!=(const token_string_entity& left, const char* pszRight) noexcept
	{
		return left.compare(pszRight) != 0;
	}

	template <std::size_t N>
	__device__ bool operator!=(const token_string_entity& left, const char(&strArray)[N]) noexcept
	{
		return left.compare(strArray) != 0;
	}

	__device__ bool operator==(const char* pszLeft, const token_string_entity& right) noexcept
	{
		return right == pszLeft;
	}

	template <std::size_t N>
	__device__ bool operator==(const char(&strArray)[N], const token_string_entity& right) noexcept
	{
		return right == strArray;
	}

	__device__ bool operator!=(const char* pszRight, const token_string_entity& right) noexcept
	{
		return right != pszRight;
	}

	template <std::size_t N>
	__device__ bool operator!=(const char(&strArray)[N], const token_string_entity& right) noexcept
	{
		return right != strArray;
	}

	//let's define token as a word.
	//One operator: = + - * / ^ ( ) ,
	//Or: something that only consists of digits and one comma
	//Or: something that starts with a letter and proceeds with letters or digits

	__device__ token_string_entity parse_string_token(const char* pExpression, std::size_t cbExpression)
	{
		auto pStart = skipSpaces(pExpression, cbExpression);
		if (pStart == pExpression + cbExpression)
			return token_string_entity();
		if (isdigit(*pStart) || *pStart == '.')
		{
			bool fFlPointFound = *pStart == '.';
			const char* pEnd = pStart;
			while (isdigit(*++pEnd) || (!fFlPointFound && *pEnd == '.')) continue;
			return token_string_entity(pStart, pEnd);
		}
		if (isalpha(*pStart))
		{
			const char* pEnd = pStart;
			while (isalnum(*++pEnd)) continue;
			return token_string_entity(pStart, pEnd);
		}
		return token_string_entity(pStart, pStart + 1);
	}

	template <class T>
	class TokenStorage
	{
		cuda_stack<cuda_device_unique_ptr<IToken<T>>> operationStack;
		cuda_list<cuda_device_unique_ptr<IToken<T>>> outputList;

	public:

		template <class TokenParamType>
		__device__ auto push_token(TokenParamType&& op) -> std::enable_if_t<
			std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
			std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value,
			IToken<T>*
		>
		{
			auto my_priority = op.getPriority();
			while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority() && op.type() != TokenType::bracket)
			{
				outputList.push_back(std::move(operationStack.top()));
				operationStack.pop();
			}
			operationStack.push(make_cuda_device_unique_ptr<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(op)));
			return operationStack.top().get();
		}

		template <class TokenParamType>
		__device__ auto push_token(TokenParamType&& value)->std::enable_if_t<
			!(std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
				std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value ||
				std::is_base_of<Bracket<T>, std::decay_t<TokenParamType>>::value),
			IToken<T>*
		>
		{
			outputList.push_back(make_cuda_device_unique_ptr<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(value)));
			return outputList.back().get();
		}

		__device__ return_wrapper_t<void> pop_bracket()
		{
			bool isOpeningBracket = false;
			while (operationStack.size() != 0)
			{
				if (operationStack.top().get()->type() != TokenType::bracket)
				{
					this->outputList.push_back(std::move(operationStack.top()));
					operationStack.pop();
				}
				else
				{
					isOpeningBracket = true;
					break;
				}
			}
			if (!isOpeningBracket)
				return return_wrapper_t<void>(CudaParserErrorCodes::InvalidNumberOfArguments);
			else
				operationStack.pop();
			return return_wrapper_t<void>();
		}

		__device__ return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>> finalize() &&
		{
			while (operationStack.size() != 0)
			{
				if (operationStack.top().get()->type() == TokenType::bracket) //checking enclosing brackets
					return return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>>(CudaParserErrorCodes::InvalidNumberOfArguments);
				else
				{
					outputList.push_back(std::move(operationStack.top()));
					operationStack.pop();
				}
			}
			return return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>>(std::move(outputList), CudaParserErrorCodes::Success);
		}

		__device__ cuda_device_unique_ptr<IToken<T>>& get_top_operation()
		{
			return operationStack.top();
		}

		__device__ return_wrapper_t<void> comma_parameter_replacement()
		{
			bool isOpeningBracket = false;

			while (!isOpeningBracket && operationStack.size() != 0) //while an opening bracket is not found or an operation stack is not empty
			{
				if (operationStack.top().get()->type() != TokenType::bracket) //if the cast to Bracket is not successfull, return NULL => it is not '('
				{
					outputList.push_back(std::move(operationStack.top()));
					operationStack.pop();
				}
				else
				{
					isOpeningBracket = true;
				}
			}

			if (!isOpeningBracket) //missing '('
				return return_wrapper_t<void>(CudaParserErrorCodes::InvalidNumberOfArguments);
			return return_wrapper_t<void>();
		}
	};

	template <class T>
	class expr_param_storage
	{
		cuda_list<cu::cuda_string> m_parameters;
		cuda_vector<cu::cuda_string*> m_sorted_params;
	public:
		__device__ expr_param_storage() = default;
		template <class ParameterNameType>
		__device__ cu::return_wrapper_t<void> add_parameter(ParameterNameType&& strParameterName)
		{
			using cu::swap;
			auto rv = m_parameters.push_back(std::forward<ParameterNameType>(strParameterName));
			if (!rv)
				return rv;
			m_sorted_params.emplace_back(&m_parameters.back());
			auto i = m_sorted_params.size() - 1;
			while (i > 0)
			{
				auto repl = i / 2 - 1;
				if (*m_sorted_params[repl] > *m_sorted_params[i])
					return cu::return_wrapper_t<void>();
				swap(m_sorted_params[repl], m_sorted_params[i]);
				i = repl;
			}
			return cu::return_wrapper_t<void>();
		}
		__device__ cu::return_wrapper_t<void> finalize()
		{
			using cu::swap;
			if (m_sorted_params.empty())
				return return_wrapper_t<void>();
			auto heap_end = m_sorted_params.size() - 1;
			while (heap_end > 0)
			{
				swap(m_sorted_params[0], m_sorted_params[--heap_end]);
				for (cuda_vector<cuda_string*>::size_type iStart = 0, iSwap; iStart < heap_end / 2 - 1; iStart = iSwap)
				{
					iSwap = iStart * 2 + 1;
					auto cmp = cu::strncmpnz(m_sorted_params[iSwap]->data(), m_sorted_params[iSwap]->size(), m_sorted_params[iSwap + 1]->data(), m_sorted_params[iSwap + 1]->size());
					if (!cmp)
						return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::ParameterIsNotUnique);
					if (cmp < 0)
						++iSwap;
					cmp = cu::strncmpnz(m_sorted_params[iStart]->data(), m_sorted_params[iStart]->size(), m_sorted_params[iSwap]->data(), m_sorted_params[iSwap]->size());
					if (!cmp)
						return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::ParameterIsNotUnique);
					if (cmp > 0)
						break;
					swap(m_sorted_params[iStart], m_sorted_params[iSwap]);
				}
			}
			return cu::return_wrapper_t<void>();
		}
		__device__ inline std::size_t size() const
		{
			return m_parameters.size();
		}
		template <class ArgIteratorBegin, class ArgIteratorEnd>
		__device__ return_wrapper_t<expr_param_init_block<T>> construct_init_block(ArgIteratorBegin arg_begin, ArgIteratorEnd arg_end) const
		{
			cuda_vector<cu::pair<const cu::cuda_string*, T>> v_init;
			auto itParam = m_parameters.begin();
			while (arg_begin != arg_end)
			{
				if (itParam == m_parameters.end())
					return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidNumberOfArguments);
				v_init.emplace_back(cu::make_pair(&*(itParam++), *(arg_begin++)));
			}
			return expr_param_init_block<T>(std::move(v_init));
		}
	};

	template <class T>
	class Header
	{
		expr_param_storage<T> m_strg;
		cu::cuda_string function_name;
		mutable return_wrapper_t<void> construction_success_code;
	public:
		__device__ Header() = default;
		__device__ Header(const char* expression, std::size_t expression_len, char** ppEndPtr)
		{
			auto endPtr = expression + expression_len;
			char* begPtr = (char*)(expression);
			cuda_list<cu::cuda_string> params;
			construction_success_code = return_wrapper_t<void>();

			bool isOpeningBracket = false;
			bool isClosingBracket = false;
			std::size_t commaCount = 0;

			while (begPtr < endPtr && *begPtr != '=')
			{
				if (isalpha(*begPtr))
				{
					auto l_endptr = begPtr + 1;
					while (l_endptr < endPtr && isalnum(*l_endptr)) ++l_endptr;
					if (this->function_name.empty())
						this->function_name = cu::cuda_string(begPtr, l_endptr);
					else
					{
						if (!isOpeningBracket)
						{
							construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedToken);
							return;
						}
						construction_success_code = m_strg.add_parameter(cu::cuda_string(begPtr, l_endptr));
						if (!construction_success_code)
							return;
					}
					begPtr = l_endptr;
				}

				if (*begPtr == ' ')
				{
					begPtr += 1;
					continue;
				}

				if (*begPtr == '(')
				{
					if (isOpeningBracket)
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidExpression);
					isOpeningBracket = true;
					begPtr += 1;
				}

				if (*begPtr == ',') //a-zA_Z0-9
				{
					commaCount += 1;
					begPtr += 1;
				}

				if (*begPtr == ')')
				{
					if (!isOpeningBracket || isClosingBracket)
						construction_success_code = return_wrapper_t<void>(CudaParserErrorCodes::InvalidExpression);
					isClosingBracket = true;
					begPtr += 1;
				}
			}
			*ppEndPtr = begPtr;
		}
		Header(const Header<T>&) = delete;
		Header& operator=(const Header<T>&) = delete;
		__device__ Header(Header&&) = default;
		__device__ Header& operator=(Header&&) = default;
		/*__device__ Header(const Header<T>& val)
		{
			std::size_t size = val.get_name_length();
			this->function_name = val.function_name;
			this->m_arguments = val.m_arguments;
			this->m_parameters = val.m_parameters;
			construction_success_code = val.construction_success_code;
		}*/
		__device__ inline std::size_t get_required_parameter_count() const
		{
			return m_strg.size();
		}
		__device__ inline const char* get_function_name() const
		{
			return function_name.c_str();
		}
		__device__ inline size_t get_name_length() const
		{
			return function_name.size();
		}
		template <class ArgIteratorBegin, class ArgIteratorEnd>
		__device__ inline return_wrapper_t<expr_param_init_block<T>> construct_argument_block(ArgIteratorBegin arg_begin, ArgIteratorEnd arg_end) const
		{
			return m_strg.construct_init_block(arg_begin, arg_end);
		}
	};

	template <class T>
	class Mathexpr
	{
	public:
		__device__ Mathexpr(const char* sMathExpr, std::size_t cbMathExpr);
		__device__ Mathexpr(const char* szMathExpr) :Mathexpr(szMathExpr, std::strlen(szMathExpr)) {}
		__device__ Mathexpr(const cuda_string& strMathExpr) : Mathexpr(strMathExpr.c_str(), strMathExpr.size()) {}
		template <class ArgIteratorBegin, class ArgIteratorEnd>
		__device__ auto operator()(ArgIteratorBegin arg_begin, ArgIteratorEnd arg_end) const
			-> std::enable_if_t<
				std::is_convertible<
					std::common_type_t<
						typename std::iterator_traits<ArgIteratorBegin>::value_type,
						typename std::iterator_traits<ArgIteratorEnd>::value_type
					>,
					T
				>::value,
			return_wrapper_t<T>>
		{
			auto rw_init_blck = header.construct_argument_block(arg_begin, arg_end);
			if (!rw_init_blck)
				return rw_init_blck;
			return body->compute(rw_init_blck.value());
		}
		template <class ArgSequenceContainer>
		__device__ inline auto operator()(const ArgSequenceContainer& container) const 
			-> decltype((*this)(cu::begin(std::declval<ArgSequenceContainer&>()), cu::end(std::declval<ArgSequenceContainer&>())))
		{
			return (*this)(cu::begin(container), cu::end(container));
		}
		__device__ inline return_wrapper_t<T> operator()(std::initializer_list<T> list) const
		{
			return (*this)(cu::begin(list), cu::end(list));
		}
		template <class ... Args>
		__device__ inline auto operator()(Args&& ... args) const -> decltype((*this)({T(std::declval<Args&&>()) ...}))
		{
			return (*this)({T(std::forward<Args>(args)) ...});
		}
	private:
		Header<T> header;
		cuda_device_unique_ptr<IToken<T>> body;

		template <class T>
		__device__ return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>> lexBody(const char* expr, std::size_t length)
		{
			char* begPtr = (char*)expr;
			std::size_t cbRest = length;
			TokenStorage<T> tokens;
			cuda_stack <cu::pair<cuda_device_unique_ptr<Function<T>>, std::size_t>> funcStack;
			int last_type_id = -1;

			while (cbRest > 0)
			{
				auto tkn = parse_string_token(begPtr, cbRest);
				if (tkn == "+")
				{
					if (last_type_id == -1 || IsOperatorTokenTypeId(TokenType(last_type_id))) //unary form
						last_type_id = int(tokens.push_token(UnaryPlus<T>())->type());
					else //binary form
						last_type_id = int(tokens.push_token(BinaryPlus<T>())->type());
				}
				else if (tkn == "-")
				{
					if (last_type_id == -1 || IsOperatorTokenTypeId(TokenType(last_type_id))) //unary form
						last_type_id = int(tokens.push_token(UnaryMinus<T>())->type());
					else //binary form
						last_type_id = int(tokens.push_token(BinaryMinus<T>())->type());
				}
				else if (tkn == "*")
					last_type_id = int(tokens.push_token(OperatorMul<T>())->type());
				else if (tkn == "/")
					last_type_id = int(tokens.push_token(OperatorDiv<T>())->type());
				else if (tkn == "^")
					last_type_id = int(tokens.push_token(OperatorPow<T>())->type());
				else if (tkn == ",")
				{
					tokens.comma_parameter_replacement();

					if (funcStack.top().first.get()->type() == TokenType::maxFunction ||
						funcStack.top().first.get()->type() == TokenType::minFunction)
						funcStack.top().second += 1;
				}
				else if (isdigit(*tkn.begin()))
				{
					char* conversion_end;
					static_assert(std::is_same<T, double>::value, "The following line is only applicable to double");
					auto value = cu::strtod(tkn.begin(), (char**)&conversion_end);
					if (conversion_end != tkn.end())
						return return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>>(CudaParserErrorCodes::InvalidExpression);
					last_type_id = int(tokens.push_token(Number<T>(value))->type());
				}
				else if (isalpha(*tkn.begin()))
				{
					if (tkn == "sin")
					{
						last_type_id = int(tokens.push_token(SinFunction<T>())->type());
						funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<SinFunction<T>>()), 1));
					}
					else if (tkn == "cos")
					{
						last_type_id = int(tokens.push_token(CosFunction<T>())->type());
						funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<CosFunction<T>>()), 1));
					}
					else if (tkn == "tg")
					{
						last_type_id = int(tokens.push_token(TgFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<TgFunction<T>>(), 1));
					}
					else if (tkn == "log10")
					{
						last_type_id = int(tokens.push_token(Log10Function<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Log10Function<T>>(), 1));
					}
					else if (tkn == "ln")
					{
						last_type_id = int(tokens.push_token(LnFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<LnFunction<T>>(), 1));
					}
					else if (tkn == "log")
					{
						last_type_id = int(tokens.push_token(LogFunction<T>())->type());
						funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<LogFunction<T>>()), 2));
					}
					else if (tkn == "j0")
					{
						last_type_id = int(tokens.push_token(J0Function<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<J0Function<T>>(), 1));
					}
					else if (tkn == "j1")
					{
						last_type_id = int(tokens.push_token(J1Function<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<J1Function<T>>(), 1));
					}
					else if (tkn == "jn")
					{
						last_type_id = int(tokens.push_token(JnFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<JnFunction<T>>(), 2));
					}
					else if (tkn == "y0")
					{
						last_type_id = int(tokens.push_token(Y0Function<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Y0Function<T>>(), 1));
					}
					else if (tkn == "y1")
					{
						last_type_id = int(tokens.push_token(Y1Function<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Y1Function<T>>(), 1));
					}
					else if (tkn == "yn")
					{
						last_type_id = int(tokens.push_token(YnFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<YnFunction<T>>(), 2));
					}
					else if (tkn == "max")
					{
						last_type_id = int(tokens.push_token(MaxFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<MaxFunction<T>>(), 0));
					}
					else if (tkn == "min")
					{
						last_type_id = int(tokens.push_token(MinFunction<T>())->type());
						funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<MinFunction<T>>(), 0));
					}
					else //if (this->header.get_param_index(cuda_string(tkn.begin(), tkn.end())).value() >= 0)
						last_type_id = int(tokens.push_token(Variable<T>(&*tkn.begin(), tkn.size()))->type());
				}
				else if (tkn == ")")
				{
					tokens.pop_bracket();
					if(!funcStack.empty())
					{
						switch (funcStack.top().first.get()->type())
						{
						case TokenType::maxFunction:
							static_cast<MaxFunction<T>&>(*tokens.get_top_operation()) = MaxFunction<T>(funcStack.top().second + 1);
							break;
						case TokenType::minFunction:
							static_cast<MinFunction<T>&>(*tokens.get_top_operation()) = MinFunction<T>(funcStack.top().second + 1);
							break;
						}
						funcStack.pop();
					}
				}
				else if (tkn == "(")
				{
					last_type_id = int(tokens.push_token(Bracket<T>())->type());
				}
				else
					return return_wrapper_t<cuda_list<cuda_device_unique_ptr<IToken<T>>>>(CudaParserErrorCodes::InvalidExpression);
				cbRest -= tkn.end() - begPtr;
				begPtr = (char*) tkn.end();
			}

			//auto formula = std::move(tokens).finalize();
			return std::move(tokens).finalize();
		}
	};

	template <class K>
	__device__ typename cuda_list<cuda_device_unique_ptr<IToken<K>>>::iterator simplify(cuda_list<cuda_device_unique_ptr<IToken<K>>>& body, typename cuda_list<cuda_device_unique_ptr<IToken<K>>>::iterator elem)
	{
		auto paramsCount = elem->get()->get_required_parameter_count();
		auto param_it = elem;
		for (auto i = paramsCount; i > 0; --i)
		{
			--param_it;
			elem->get()->push_argument(std::move(*param_it));
			//((*elem.data).get())->push_argument(*param_it); //here std::move must be
			param_it = body.erase(param_it);
		}
		if (elem->get()->is_ready())
			*elem = elem->get()->simplify().value();
		//*elem = *(elem->data.get()->simplify()).get();
		++elem;
		return elem;
	}

	template <class T>
	__device__ cuda_device_unique_ptr<IToken<T>> simplify_body(cuda_list<cuda_device_unique_ptr<IToken<T>>>&& listBody)
	{
		auto it = listBody.begin();
		while (listBody.size() > 1)
			it = simplify(listBody, it);
		return std::move(*listBody.begin());
		//When everything goes right, you are left with only one element within the list - the root of the tree.
	}
	template <class T>
	__device__ T compute(const cuda_list<cuda_device_unique_ptr<IToken<T>>>& body)
	{
		assert(body.size() == 1);
		return body.front()();
	}

	template <class T>
	__device__ Mathexpr<T>::Mathexpr(const char* sMathExpr, std::size_t cbMathExpr)
	{
		const char* endptr;
		header = Header<T>(sMathExpr, cbMathExpr, (char**)&endptr);
		++endptr;
		this->body = simplify_body(cuda_list<cuda_device_unique_ptr<IToken<T>>>(lexBody<T>(endptr, cbMathExpr - (endptr - sMathExpr)).value()));
	}
}

#endif // !PARSER_H
