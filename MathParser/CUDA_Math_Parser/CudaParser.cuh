#include "cuda_config.cuh"
#include "cuda_return_wrapper.cuh"
#include "cuda_tokens.cuh"
#include "cuda_iterator.cuh"
#include "cuda_list.cuh"
#include "cuda_map.cuh"
#include "cuda_memory.cuh"
#include "cuda_stack.cuh"
#include "cuda_vector.cuh"
#include "thrust/complex.h"

#ifndef PARSER_H
#define PARSER_H

CU_BEGIN

template<class T> struct is_complex : std::false_type {};
template<class T> struct is_complex<thrust::complex<T>> : std::true_type {};

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
		__device__ inline int compare(char ch) const noexcept
		{
			return this->compare(&ch, 1);
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

	__device__ bool operator==(const token_string_entity& left, char chTkn) noexcept
	{
		return left.compare(chTkn) == 0;
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

	__device__ bool operator!=(const token_string_entity& left, char chTkn) noexcept
	{
		return left.compare(chTkn) != 0;
	}

	//let's define token as a word.
	//One operator: = + - * / ^ ( ) ,
	//Or: something that only consists of digits and one comma
	//Or: something that starts with a letter and proceeds with letters or digits

	template <class T>
	__device__ std::enable_if_t<std::is_same<T, double>::value, token_string_entity> parse_string_token(const char* pExpression, std::size_t cbExpression)
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
	__device__ std::enable_if_t<cu::is_complex<T>::value, token_string_entity> parse_string_token(const char* pExpression, std::size_t cbExpression)
	{
		auto pStart = skipSpaces(pExpression, cbExpression);
		if (pStart == pExpression + cbExpression)
			return token_string_entity();
		if (isdigit(*pStart) || *pStart == '.')
		{
			bool fFlPointFound = *pStart == '.';
			const char* pEnd = pStart;
			while (isdigit(*++pEnd) || (!fFlPointFound && *pEnd == '.') || (*pEnd == 'j')) continue;
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
		cu::stack<cuda_device_unique_ptr<IToken<T>>> operationStack;
		cu::list<cuda_device_unique_ptr<IToken<T>>> outputList;

	public:

		template <class TokenParamType>
		__device__ auto push_token(TokenParamType&& op) -> std::enable_if_t<
			std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
			std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value,
			cu::return_wrapper_t<IToken<T>*>
		>
		{
			auto my_priority = op.getPriority();
			while (operationStack.size() != 0 && my_priority <= operationStack.top()->getPriority() && op.type() != TokenType::bracket)
			{
				outputList.push_back(std::move(operationStack.top()));
				operationStack.pop();
			}
			auto rw = operationStack.push(make_cuda_device_unique_ptr<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(op)));
			if (!rw)
				return rw;
			return operationStack.top().get();
		}

		template <class TokenParamType>
		__device__ auto push_token(TokenParamType&& value)->std::enable_if_t<
			!(std::is_base_of<Operator<T>, std::decay_t<TokenParamType>>::value ||
				std::is_base_of<Function<T>, std::decay_t<TokenParamType>>::value ||
				std::is_base_of<Bracket<T>, std::decay_t<TokenParamType>>::value),
			cu::return_wrapper_t<IToken<T>*>
		>
		{
			outputList.push_back(make_cuda_device_unique_ptr<std::decay_t<TokenParamType>>(std::forward<TokenParamType>(value)));
			return cu::return_wrapper_t<IToken<T>*>(outputList.back().get());
		}

		__device__ cu::return_wrapper_t<void> pop_bracket()
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
				return cu::return_wrapper_t<void>(CudaParserErrorCodes::InvalidNumberOfArguments);
			else
				operationStack.pop();
			return cu::return_wrapper_t<void>();
		}

		__device__ cu::return_wrapper_t<cu::list<cuda_device_unique_ptr<IToken<T>>>> finalize() &&
		{
			while (operationStack.size() != 0)
			{
				if (operationStack.top().get()->type() == TokenType::bracket) //checking enclosing brackets
					return cu::return_wrapper_t<cu::list<cuda_device_unique_ptr<IToken<T>>>>(CudaParserErrorCodes::InvalidNumberOfArguments);
				else
				{
					outputList.push_back(std::move(operationStack.top()));
					operationStack.pop();
				}
			}
			return cu::return_wrapper_t<cu::list<cuda_device_unique_ptr<IToken<T>>>>(std::move(outputList), CudaParserErrorCodes::Success);
		}

		__device__ cuda_device_unique_ptr<IToken<T>>& get_top_operation()
		{
			return operationStack.top();
		}

		__device__ cu::return_wrapper_t<void> comma_parameter_replacement()
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
				return cu::return_wrapper_t<void>(CudaParserErrorCodes::InvalidNumberOfArguments);
			return cu::return_wrapper_t<void>();
		}
	};

	template <class T>
	class expr_param_storage
	{
		cu::list<cu::string> m_parameters;
		cu::vector<cu::string*> m_sorted_params;
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
				return cu::return_wrapper_t<void>();
			auto heap_end = m_sorted_params.size() - 1;
			while (heap_end > 0)
			{
				swap(m_sorted_params[0], m_sorted_params[--heap_end]);
				for (cu::vector<cu::string*>::size_type iStart = 0, iSwap; iStart < heap_end / 2 - 1; iStart = iSwap)
				{
					iSwap = iStart * 2 + 1;
					auto cmp = cu::strncmpnz(m_sorted_params[iSwap]->data(), m_sorted_params[iSwap]->size(), m_sorted_params[iSwap + 1]->data(), m_sorted_params[iSwap + 1]->size());
					if (!cmp)
						return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::ParameterIsNotUnique);
					if (cmp.value() < 0)
						++iSwap;
					cmp = cu::strncmpnz(m_sorted_params[iStart]->data(), m_sorted_params[iStart]->size(), m_sorted_params[iSwap]->data(), m_sorted_params[iSwap]->size());
					if (!cmp)
						return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::ParameterIsNotUnique);
					if (cmp.value() > 0)
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
		__device__ cu::return_wrapper_t<expr_param_init_block<T>> construct_init_block(ArgIteratorBegin arg_begin, ArgIteratorEnd arg_end) const
		{
			cu::vector<cu::pair<const cu::string*, T>> v_init;
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
		cu::string function_name;
		mutable cu::return_wrapper_t<void> construction_success_code;
	public:
		__device__ Header() = default;
		__device__ Header(const char* expression, std::size_t expression_len, char** ppEndPtr)
		{
			auto endPtr = expression + expression_len;
			char* begPtr = (char*)(expression);
			cu::list<cu::string> params;
			construction_success_code = cu::return_wrapper_t<void>();

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
						this->function_name = cu::string(begPtr, l_endptr);
					else
					{
						if (!isOpeningBracket)
						{
							construction_success_code = cu::return_wrapper_t<void>(CudaParserErrorCodes::UnexpectedToken);
							return;
						}
						if (cu::strcmp(cu::string(begPtr, l_endptr).c_str(), "j") == 0 && cu::is_complex<T>::value)
						{
							construction_success_code = cu::return_wrapper_t<void>(cu::CudaParserErrorCodes::InvalidArgument);
							return;
						}
						construction_success_code = m_strg.add_parameter(cu::string(begPtr, l_endptr));
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
						construction_success_code = cu::return_wrapper_t<void>(CudaParserErrorCodes::InvalidExpression);
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
						construction_success_code = cu::return_wrapper_t<void>(CudaParserErrorCodes::InvalidExpression);
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
		__device__ inline cu::return_wrapper_t<expr_param_init_block<T>> construct_argument_block(ArgIteratorBegin arg_begin, ArgIteratorEnd arg_end) const
		{
			return m_strg.construct_init_block(arg_begin, arg_end);
		}
	};

	template <class T>
	class Mathexpr
	{
		static constexpr char IMAGINARY_UNIT = 'j';
		template <class U = T>
		__device__ static constexpr auto is_imaginary_unit(char) -> std::enable_if_t<std::is_same<U, double>::value, bool>
		{
			return false;
		}
		template <class U = T>
		__device__ static constexpr auto is_imaginary_unit(char ch) -> std::enable_if_t<cu::is_complex<U>::value, bool>
		{
			return ch == IMAGINARY_UNIT;
		}
		template <class U = T>
		__device__ static auto parse_imaginary_unit() -> std::enable_if_t<cu::is_complex<U>::value, Number<U>>
		{
			return U(0, 1);
		}
		template <class U = T, class = void>
		__device__ static auto parse_imaginary_unit() -> std::enable_if_t<std::is_same<U, double>::value, Variable<U>>
		{
			auto ch = IMAGINARY_UNIT;
			return Variable<U>(&ch, 1);
		}
		template <class U = T>
		__device__ static auto parse_val(const token_string_entity& tkn) -> std::enable_if_t<std::is_same<U, double>::value, return_wrapper_t<U>>
		{
			char* conversion_end;
			auto value = cu::strtod(tkn.begin(), (char**)&conversion_end);
			if (conversion_end != tkn.end())
				return cu::make_return_wrapper_error(CudaParserErrorCodes::InvalidExpression);
			return return_wrapper_t<U>(value);
		}
		template <class U = T>
		__device__ static auto parse_val(const token_string_entity& tkn) -> std::enable_if_t<cu::is_complex<U>::value, return_wrapper_t<U>>
		{
			char* conversion_end;
			auto value = cu::strtod(tkn.begin(), (char**)&conversion_end);
			if (conversion_end == tkn.end())
				return return_wrapper_t<U>(U(value));
			//auto i_marker = conversion_end;
			//if (*conversion_end == ' ')
			//	i_marker = cu::skipSpaces(i_marker);
			if (*conversion_end == IMAGINARY_UNIT)
				return return_wrapper_t<U>(U(0, value));
			return cu::make_return_wrapper_error(CudaParserErrorCodes::InvalidExpression);
		}
	public:
		__device__ Mathexpr(cu::CudaParserErrorCodes* pCode, const char* sMathExpr, std::size_t cbMathExpr);
		__device__ Mathexpr(cu::CudaParserErrorCodes* pCode, const char* szMathExpr) :Mathexpr(pCode, szMathExpr, std::strlen(szMathExpr)) {}
		__device__ Mathexpr(cu::CudaParserErrorCodes* pCode, const cu::string& strMathExpr) : Mathexpr(pCode, strMathExpr.c_str(), strMathExpr.size()) {}
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
			cu::return_wrapper_t<T>>
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
		__device__ inline cu::return_wrapper_t<T> operator()(std::initializer_list<T> list) const
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

		enum class LastParsedId
		{
			Unspecified,
			Literal,
			Function,
			UnaryOperator,
			BinaryOperator,
			OpenBracket,
			ClosingBracket,
			Comma
		};
		__device__ static inline bool verify_unary_operator(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::Unspecified:
			case LastParsedId::BinaryOperator:
			case LastParsedId::Comma:
			case LastParsedId::OpenBracket:
				return true;
			default:
				return false;
			}
		}
		__device__ static inline bool verify_binary_operator(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::ClosingBracket:
			case LastParsedId::Literal:
				return true;
			default:
				return false;
			}
		}
		__device__ static inline bool verify_comma(LastParsedId last_parsed_entity)
		{
			return verify_binary_operator(last_parsed_entity);
		}
		__device__ static inline bool verify_literal(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::Unspecified:
			case LastParsedId::UnaryOperator:
			case LastParsedId::BinaryOperator:
			case LastParsedId::Comma:
			case LastParsedId::OpenBracket:
				return true;
			default:
				return false;
			}
		}
		__device__ static inline bool verify_open_bracket(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::Unspecified:
			case LastParsedId::Function:
			case LastParsedId::UnaryOperator:
			case LastParsedId::BinaryOperator:
			case LastParsedId::OpenBracket:
				return true;
			default:
				return false;
			}
		}
		__device__ static inline bool verify_closing_bracket(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::Literal:
			case LastParsedId::ClosingBracket:
				return true;
			default:
				return false;
			}
		}
		__device__ static inline bool verify_end(LastParsedId last_parsed_entity)
		{
			switch (last_parsed_entity)
			{
			case LastParsedId::Literal:
			case LastParsedId::ClosingBracket:
				return true;
			default:
				return false;
			}
		}
		__device__ cu::return_wrapper_t<cu::list<cuda_device_unique_ptr<IToken<T>>>> lexBody(const char* expr, std::size_t length)
		{
			char* begPtr = (char*)expr;
			std::size_t cbRest = length;
			TokenStorage<T> tokens;
			cu::stack <cu::pair<cuda_device_unique_ptr<Function<T>>, std::size_t>> funcStack;
			int last_type_id = -1;
			LastParsedId last_parse_entity = LastParsedId::Unspecified;

			while (cbRest > 0)
			{
				auto tkn = parse_string_token<T>(begPtr, cbRest);
				if (tkn == "+")
				{
					if (verify_unary_operator(last_parse_entity))
					{
						auto rw = tokens.push_token(UnaryPlus<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::UnaryOperator;
					}else if (verify_binary_operator(last_parse_entity))
					{
						auto rw = tokens.push_token(BinaryPlus<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::BinaryOperator;
					}else
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
				}
				else if (tkn == "-")
				{
					
					if (verify_unary_operator(last_parse_entity))
					{
						auto rw = tokens.push_token(UnaryMinus<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::UnaryOperator;
					}else if (verify_binary_operator(last_parse_entity))
					{
						auto rw = tokens.push_token(BinaryMinus<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::BinaryOperator;
					}else
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
				}
				else if (tkn == "*")
				{
					if (!verify_binary_operator(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					auto rw = tokens.push_token(OperatorMul<T>());
					if (!rw)
						return rw;
					last_type_id = int(rw.value()->type());
					last_parse_entity = LastParsedId::BinaryOperator;
				}else if (tkn == "/")
				{
					if (!verify_binary_operator(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					auto rw = tokens.push_token(OperatorDiv<T>());
					if (!rw)
						return rw;
					last_type_id = int(rw.value()->type());
					last_parse_entity = LastParsedId::BinaryOperator;
				}else if (tkn == "^")
				{
					if (!verify_binary_operator(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					auto rw = tokens.push_token(OperatorPow<T>());
					if (!rw)
						return rw;
					last_type_id = int(rw.value()->type());
					last_parse_entity = LastParsedId::BinaryOperator;
				}else if (tkn == ",")
				{
					if (!verify_comma(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					tokens.comma_parameter_replacement();

					if (funcStack.top().first.get()->type() == TokenType::maxFunction ||
						funcStack.top().first.get()->type() == TokenType::minFunction)
						funcStack.top().second += 1;
					last_parse_entity = LastParsedId::Comma;
				}
				else if (isdigit(*tkn.begin()))
				{
					if (!verify_literal(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					auto rw_val = parse_val(tkn);
					if (!rw_val)
						return rw_val;
					auto rw_tkn = tokens.push_token(Number<T>(rw_val.value()));
					if (!rw_tkn)
						return rw_tkn;
					last_type_id = int(rw_tkn.value()->type());
					last_parse_entity = LastParsedId::Literal;
				}
				else if (isalpha(*tkn.begin()))
				{
					if (!verify_literal(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					if (is_imaginary_unit(*tkn.begin()))
					{
						auto rw = tokens.push_token(parse_imaginary_unit());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
					}
					else if (tkn == "PI")
					{
						auto rw = tokens.push_token(PI<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type()); // TODO: Set last_parse_entity for every entity
						last_parse_entity = LastParsedId::Literal;
					}
					else if (tkn == "EULER")
					{
						auto rw = tokens.push_token(Euler<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
					}
					else if (tkn == "arg")
					{
						auto rw = tokens.push_token(ArgFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<ArgFunction<T>>()), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "sin")
					{
						auto rw = tokens.push_token(SinFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<SinFunction<T>>()), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "cos")
					{
						auto rw = tokens.push_token(CosFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<CosFunction<T>>()), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "tg")
					{
						auto rw = tokens.push_token(TgFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<TgFunction<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "log10")
					{
						auto rw = tokens.push_token(Log10Function<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Log10Function<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "ln")
					{
						auto rw = tokens.push_token(LnFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<LnFunction<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "log")
					{
						auto rw = tokens.push_token(LogFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(std::move(make_cuda_device_unique_ptr<LogFunction<T>>()), 2));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "j0")
					{
						auto rw = tokens.push_token(J0Function<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<J0Function<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "j1")
					{
						auto rw = tokens.push_token(J1Function<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<J1Function<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "jn")
					{
						auto rw = tokens.push_token(JnFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<JnFunction<T>>(), 2));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "y0")
					{
						auto rw = tokens.push_token(Y0Function<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Y0Function<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "y1")
					{
						auto rw = tokens.push_token(Y1Function<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<Y1Function<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "yn")
					{
						auto rw = tokens.push_token(YnFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<YnFunction<T>>(), 2));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "gamma")
					{
						auto rw = tokens.push_token(GammaFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<GammaFunction<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "abs")
					{
						auto rw = tokens.push_token(AbsFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<AbsFunction<T>>(), 1));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "polar")
					{
						auto rw = tokens.push_token(PolarFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<PolarFunction<T>>(), 2));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "max")
					{
						auto rw = tokens.push_token(MaxFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<MaxFunction<T>>(), 0));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else if (tkn == "min")
					{
						auto rw = tokens.push_token(MinFunction<T>());
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
						rw = funcStack.push(cu::make_pair(make_cuda_device_unique_ptr<MinFunction<T>>(), 0));
						if(!rw)
							return make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory); 
					}
					else //if (this->header.get_param_index(cu::string(tkn.begin(), tkn.end())).value() >= 0)
					{
						auto rw = tokens.push_token(Variable<T>(&*tkn.begin(), tkn.size()));
						if (!rw)
							return rw;
						last_type_id = int(rw.value()->type());
						last_parse_entity = LastParsedId::Literal;
					}
				}
				else if (tkn == ")")
				{
					if (!verify_open_bracket(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
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
					last_parse_entity = LastParsedId::OpenBracket;
				}
				else if (tkn == "(")
				{
					if (!verify_closing_bracket(last_parse_entity))
						return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
					auto rw = tokens.push_token(Bracket<T>());
					if (!rw)
						return rw;
					last_type_id = int(rw.value()->type());
					last_parse_entity = LastParsedId::ClosingBracket;
				}
				else
					return cu::return_wrapper_t<cu::list<cuda_device_unique_ptr<IToken<T>>>>(CudaParserErrorCodes::InvalidExpression);
				cbRest -= tkn.end() - begPtr;
				begPtr = (char*) tkn.end();
			}
			if (!verify_end(last_parse_entity))
				return make_return_wrapper_error(cu::CudaParserErrorCodes::InvalidExpression);
			//auto formula = std::move(tokens).finalize();
			return std::move(tokens).finalize();
		}
	};

	template <class K>
	__device__ typename cu::list<cuda_device_unique_ptr<IToken<K>>>::iterator simplify(cu::list<cuda_device_unique_ptr<IToken<K>>>& body, typename cu::list<cuda_device_unique_ptr<IToken<K>>>::iterator elem)
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
	__device__ cuda_device_unique_ptr<IToken<T>> simplify_body(cu::list<cuda_device_unique_ptr<IToken<T>>>&& listBody)
	{
		auto it = listBody.begin();
		while (listBody.size() > 1)
			it = simplify(listBody, it);
		return std::move(*listBody.begin());
		//When everything goes right, you are left with only one element within the list - the root of the tree.
	}
	template <class T>
	__device__ T compute(const cu::list<cuda_device_unique_ptr<IToken<T>>>& body)
	{
		assert(body.size() == 1);
		return body.front()();
	}

	template <class T>
	__device__ Mathexpr<T>::Mathexpr(cu::CudaParserErrorCodes* pCode, const char* sMathExpr, std::size_t cbMathExpr)
	{
		const char* endptr;
		header = Header<T>(sMathExpr, cbMathExpr, (char**)&endptr);
		++endptr;
		auto rwFormula = lexBody(endptr, cbMathExpr - (endptr - sMathExpr));
		if(bool(rwFormula))
			this->body = simplify_body(std::move(rwFormula.value()));
		*pCode = rwFormula.return_code();
	}
CU_END

#endif // !PARSER_H
