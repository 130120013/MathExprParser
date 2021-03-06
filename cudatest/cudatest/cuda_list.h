#include <iostream>
#include <type_traits>
//#include <stdexcept>
//#include "cuda_memory.h"
#include "cuda_config.h"
#include "cuda_return_wrapper.h"

#ifndef CUDA_LIST_CUH
#define CUDA_LIST_CUH

CU_BEGIN

template <class DerivedType, class NodeType>
class list_iterator_base;

template <typename T>
class list;

template <class T>
class list_iterator;

template <class T>
class list_const_iterator;

template <class DerivedType, class NodeType>
class list_iterator_base//:impl_list_iterator_proxy<DerivedType, NodeType>
{
protected:
	typedef std::remove_cv_t<NodeType> internal_node_type;
private:
	internal_node_type* it_value = nullptr;
	template <class Derived2, class NodeArg>
	friend class list_iterator_base;
public:
	typedef std::conditional_t<std::is_const<NodeType>::value, const typename NodeType::value_type, typename NodeType::value_type> value_type;
	typedef value_type& reference;
	typedef value_type* pointer;
	typedef std::ptrdiff_t difference_type;
	typedef std::bidirectional_iterator_tag iterator_category;

	__device__ list_iterator_base() = default;
	template <class DerivedArg>
	__device__ list_iterator_base(const list_iterator_base<DerivedArg, std::remove_cv_t<NodeType>>& right) : it_value(right.it_val()) {}
	__device__ explicit list_iterator_base(internal_node_type* pNode) :it_value(pNode) {}

	__device__ reference operator*() const
	{
		return it_value->data;
	}
	__device__ pointer operator->() const
	{
		return &it_value->data;
	}
	__device__ DerivedType& operator++()
	{
		it_value = it_value->next;
		return static_cast<DerivedType&>(*this);
	}
	__device__ DerivedType operator++(int)
	{
		auto old = static_cast<DerivedType&>(*this);
		++static_cast<DerivedType&>(*this);
		return old;
	}
	__device__ DerivedType& operator--()
	{
		it_value = it_value->prev;
		return static_cast<DerivedType&>(*this);
	}
	__device__ DerivedType operator--(int)
	{
		auto old = static_cast<DerivedType&>(*this);
		--static_cast<DerivedType&>(*this);
		return old;
	}
	__device__ bool operator==(const list_iterator_base& right) const
	{
		return it_value == right.it_value;
	}
	__device__ bool operator!=(const list_iterator_base& right) const
	{
		return it_value != right.it_value;
	}
protected:
	__device__ internal_node_type* it_val() const
	{
		return it_value;
	}
	//TODO: methods needed for the list
};

template <class T>
struct node
{
	typedef T value_type;
	T data;
	node *next, *prev;
	template <class U, class = std::enable_if_t<std::is_constructible<T, U&&>::value>>
	__device__ node(U&& data, node* next, node* prev) : data(std::forward<U>(data)), next(next), prev(prev) {} /*@*/
	__device__ ~node()
	{
		node<T>* pCurr = this;
		node<T>* pNodeNext = pCurr->next, *pNodePrev = pCurr->prev;
		while (pNodeNext != nullptr)
		{
			auto tmp = pNodeNext;
			pNodeNext = pNodeNext->next;
			if (pNodeNext != nullptr)
				pNodeNext->prev = nullptr;
			tmp->prev = tmp->next = nullptr;
			delete tmp;
		}
		while (pNodePrev != nullptr)
		{
			auto tmp = pNodePrev;
			pNodePrev = pNodePrev->prev;
			if (pNodePrev != nullptr)
				pNodePrev->next = nullptr;
			tmp->prev = tmp->next = nullptr;
			delete tmp;
		}
	}
};

template <typename T>
class list;

template <class T>
class list_iterator;

template <class T>
class list_const_iterator :public list_iterator_base<list_const_iterator<T>, const node<T>>
{
	friend class list<T>;
public:
	using list_iterator_base<list_const_iterator<T>, const node<T>>::list_iterator_base;
};

template <class T>
class list_iterator :public list_iterator_base<list_iterator<T>, node<T>>
{
	friend class list<T>;
public:
	using list_iterator_base<list_iterator<T>, node<T>>::list_iterator_base;
};

template <class T,
	bool is_move_constr = std::is_move_constructible<T>::value,
	bool is_copy_constr = std::is_copy_constructible<T>::value,
	bool is_move_assign = std::is_move_assignable<T>::value,
	bool is_copy_assign = std::is_copy_assignable<T>::value>
	struct list_proxy;

template <class T, bool /*ignored*/, class derived_class = list_proxy<T>>
struct list_proxy_move_constr
{
	__device__ list_proxy_move_constr() = default;
	__device__ list_proxy_move_constr(const list_proxy_move_constr&) = default;
	__device__ list_proxy_move_constr(list_proxy_move_constr&& right)
	{
		this->derived().m_head = right.derived().m_head;
		this->derived().m_tail = right.derived().m_tail;
		this->derived().m_elements = right.derived().m_elements;

		right.derived().m_head = nullptr;
		right.derived().m_tail = nullptr;
		right.derived().m_elements = 0;
	}
	__device__ list_proxy_move_constr& operator=(const list_proxy_move_constr&) = default;
	__device__ list_proxy_move_constr& operator=(list_proxy_move_constr&&) = default;
private:
	__device__ const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	__device__ derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

template <class T, bool /*ignore*/, class derived_class = list_proxy<T>>
struct list_proxy_move_assign
{
	__device__ list_proxy_move_assign() = default;
	__device__ list_proxy_move_assign(const list_proxy_move_assign&) = default;
	__device__ list_proxy_move_assign(list_proxy_move_assign&& right) = default;
	__device__ list_proxy_move_assign& operator=(const list_proxy_move_assign&) = default;
	__device__ list_proxy_move_assign& operator=(list_proxy_move_assign&& right)
	{
		if (&this->derived() != &right.derived())
		{
			this->derived().m_head = right.derived().m_head;
			this->derived().m_tail = right.derived().m_tail;
			this->derived().m_elements = right.derived().m_elements;

			right.derived().m_head = nullptr;
			right.derived().m_tail = nullptr;
			right.derived().m_elements = 0;
		}
		return *this;
	}
private:
	__device__ const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	__device__ derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

template <class T, bool is_copy_constr, class derived_class = list_proxy<T>> struct list_proxy_copy_constr
{
	__device__ list_proxy_copy_constr() = default;
	__device__ list_proxy_copy_constr(const list_proxy_copy_constr&) = delete;
	__device__ list_proxy_copy_constr(list_proxy_copy_constr&&) = default;
	__device__ list_proxy_copy_constr& operator=(const list_proxy_copy_constr&) = default;
	__device__ list_proxy_copy_constr& operator=(list_proxy_copy_constr&&) = default;
};

template <class T, class derived_class> struct list_proxy_copy_constr<T, true, derived_class>
{
	__device__ list_proxy_copy_constr() = default;
	__device__ list_proxy_copy_constr(const list_proxy_copy_constr& right)
	{
		if (!right.derived().m_elements)
		{
			this->derived().m_head = this->derived().m_tail = nullptr;
			this->derived().m_elements = 0;
		} else
		{
			node<T> *pNode = right.derived().m_head, *pHead = make_cuda_device_unique_ptr<node<T>>(pNode->data, nullptr, nullptr).release(), *pTail = pHead;
			while ((pNode = pNode->next) != nullptr)
			{
				pTail->next = make_cuda_device_unique_ptr<node<T>>(pNode->data, nullptr, pTail).release();
				pTail = pTail->next;
			}
			this->derived().m_head = pHead;
			this->derived().m_tail = pTail;
		}
		this->derived().m_elements = right.derived().m_elements;
	}
	__device__ list_proxy_copy_constr(list_proxy_copy_constr&&) = default;
	__device__ list_proxy_copy_constr& operator=(const list_proxy_copy_constr&) = default;
	__device__ list_proxy_copy_constr& operator=(list_proxy_copy_constr&&) = default;
private:
	__device__ const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	__device__ derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

template <class T, bool is_copy_assign, class derived_class = list_proxy<T>> struct list_proxy_copy_assign
{
	__device__ list_proxy_copy_assign() = default;
	__device__ list_proxy_copy_assign(const list_proxy_copy_assign&) = default;
	__device__ list_proxy_copy_assign(list_proxy_copy_assign&&) = default;
	__device__ list_proxy_copy_assign& operator=(const list_proxy_copy_assign&) = delete;
	__device__ list_proxy_copy_assign& operator=(list_proxy_copy_assign&&) = default;
};

template <class T, class derived_class> struct list_proxy_copy_assign<T, true, derived_class>
{
	__device__ list_proxy_copy_assign() = default;
	__device__ list_proxy_copy_assign(const list_proxy_copy_assign&) = default;
	__device__ list_proxy_copy_assign(list_proxy_copy_assign&&) = default;
	__device__ list_proxy_copy_assign operator=(const list_proxy_copy_assign& right)
	{
		if (&this->derived() != &right.derived())
		{
			auto pLeft = this->derived().m_head, pRight = right.derived().m_head;
			if (pLeft != nullptr && pRight != nullptr)
			{
				while (true)
				{
					pLeft->data = pRight->data;
					pLeft = pLeft->next;
					pRight = pRight->next;
					if (pLeft->next == nullptr || pRight->next == nullptr)
						break;
					pLeft = pLeft->next;
					pRight = pRight->next;
				}
			}
			if (!pLeft->next)
			{
				while (pRight->next)
				{
					pRight = pRight->next;
					pLeft->next = make_cuda_device_unique_ptr<node<T>>(pRight->data, nullptr, pLeft);
					pLeft = pLeft->next;
				}
			} else if (!pRight->next)
			{
				for (auto pLeftNext = pLeft->next; pLeftNext != nullptr; )
				{
					auto pCurrent = pLeftNext;
					pLeftNext = pLeftNext->next;
					delete pCurrent;
				}
				pLeft->next = nullptr;
			}
			pLeft->m_elements = pRight->m_elements;
		}
		return *this;
	}
	__device__ list_proxy_copy_assign& operator=(list_proxy_copy_assign&&) = default;
private:
	__device__ const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	__device__ derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

template <class T>
struct list_fields
{
	node<T>* m_head = nullptr, *m_tail = nullptr;
	std::size_t m_elements = 0;
};

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable:4584)
#endif

template <class T, bool is_move_constr, bool is_copy_constr, bool is_move_assign, bool is_copy_assign>
struct list_proxy
	:list_fields<T>, list_proxy_move_constr<T, is_move_constr>, list_proxy_move_assign<T, is_move_assign>,
	list_proxy_copy_constr<T, is_copy_constr>, list_proxy_copy_assign<T, is_copy_assign>
{
};

#ifdef _MSC_VER
#pragma warning (pop)
#endif

template <typename T>
class list :public list_proxy<T>
{
public:
	typedef list_iterator<T> iterator;
	typedef list_const_iterator<T> const_iterator;
	__device__ list() = default;
	__device__ list(const list&) = default;
	__device__ list(list&&) = default;
	__device__ list& operator=(const list&) = default;
	__device__ list& operator=(list&&) = default;
	__device__ ~list();

	template <class U>
	__device__ cu::return_wrapper_t<void> push_back(U&& data);
	template <class U>
	__device__ cu::return_wrapper_t<void> push_front(U&& data);
	//__device__ void pop_back();
	//__device__ void pop_front();
	__device__ void swap(list &x);
	__device__ void clear();
	__device__ iterator erase(const_iterator pos);

	__device__ const_iterator begin() const;
	__device__ iterator begin();

	__device__ const_iterator end() const;
	__device__ iterator end();

	//__device__ const_iterator rbegin() const; //reverse_iterator!
	//__device__ iterator rbegin();

	//__device__ const_iterator rend() const;
	//__device__ iterator rend();

	__device__ size_t size() const;
	__device__ bool empty() const;

	__device__ T& front();
	__device__ T const& front() const;

	__device__ T& back();
	__device__ T const& back() const;
};

template <typename T>
__device__ list <T>::~list()
{
	if (this->m_head)
		delete this->m_head;
}

template <typename T>
__device__ T& list<T>::front() {

	return this->m_head->data;
}

template <typename T>
__device__ T const& list<T>::front() const {
	return this->m_head->data;
}

template <typename T>
__device__ T& list<T>::back() {
	return this->m_tail->data;
}

template <typename T>
__device__ T const& list<T>::back() const {

	return this->m_tail->data;
}

template <typename T> template <class U>
__device__ cu::return_wrapper_t<void> list<T>::push_back(U&& data)
{
	node<T>* newNode = new node<T>(std::forward<U>(data), nullptr, this->m_tail);
	if (!newNode)
		return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
	if (this->m_head == nullptr)
		this->m_head = newNode;
	if (this->m_tail != nullptr)
		this->m_tail->next = newNode;
	this->m_tail = newNode;
	++this->m_elements;
	return cu::return_wrapper_t<void>();
}

template <typename T>
__device__ typename list<T>::iterator list<T>::erase(typename list<T>::const_iterator pos)
{
	if (pos.it_val() == nullptr) // (this->size() == 0) == true only when (pos.it_val() == nullptr) is true.
		return list<T>::iterator();
	auto pNode = pos.it_val();
	auto pNodePrev = pNode->prev;
	auto pNodeNext = pNode->next;
	if (pNodePrev != nullptr)
		pNodePrev->next = pNodeNext;
	if (pNodeNext != nullptr)
		pNodeNext->prev = pNodePrev;
	pNode->prev = pNode->next = nullptr;
	delete pNode;
	if (pNode == this->m_head)
		this->m_head = pNodeNext;
	else if (pNode == this->m_tail)
		this->m_tail = pNodePrev;
	--this->m_elements;
	return list<T>::iterator(pNodeNext);
}
//template <typename T>
//__device__ void list<T>::push_back(T&& data)
//{
//	node* newNode = new node(std::move(data), nullptr, this->m_tail);
//	if (this->m_head == nullptr)
//		this->m_head = newNode;
//	if (this->m_tail != nullptr)
//		this->m_tail->next = newNode;
//	this->m_tail = newNode;
//	++this->m_elements;
//}

template <typename T> template <class U>
__device__ cu::return_wrapper_t<void> list<T>::push_front(U&& data) {
	node<T>* newNode = new node<T>(std::forward<U>(data), this->m_head, nullptr);
	if (!newNode)
		return cu::make_return_wrapper_error(cu::CudaParserErrorCodes::NotEnoughMemory);
	if (this->m_tail == nullptr)
		this->m_tail = newNode;
	if (this->m_head != nullptr)
		this->m_head->prev = newNode;
	this->m_head = newNode;
	++this->m_elements;
	return cu::return_wrapper_t<void>();
}

//template <typename T>
//__device__ void list<T>::pop_front() 
//{
//	node *tmp = this->m_head;
//	if(this->size() == 1)
//	{
//		this->m_head = this->m_tail = nullptr;
//	}
//	else
//	{
//		this->m_head = this->m_head->next;
//		this->m_head->prev = nullptr;
//	}
//	--this->m_elements;
//	delete tmp;
//}
//
//template <typename T>
//__device__ void list<T>::pop_back() {
//
//	node *tmp = this->m_tail;
//	if(this->size() == 1)
//	{
//		this->m_head = this->m_tail = nullptr;
//	}
//	else
//	{
//		this->m_tail = this->m_tail->prev;
//		this->m_tail->next = nullptr;
//	}
//	--this->m_elements;
//	delete tmp;
//}

template <typename T>
__device__ bool list<T>::empty() const {
	return this->m_head == nullptr;
}

template <typename T>
__device__ size_t list<T>::size() const {
	return this->m_elements;
}

template <typename T>
__device__ typename list<T>::const_iterator list<T>::begin() const {
	return list<T>::const_iterator(this->m_head);
}

template <typename T>
__device__ typename list<T>::iterator list<T>::begin() {
	return list<T>::iterator(this->m_head);
}

template <typename T>
__device__ typename list<T>::const_iterator list<T>::end() const {
	return list<T>::const_iterator();
}

template <typename T>
__device__ typename list<T>::iterator list<T>::end() {
	return list<T>::iterator();
}

//
//template <typename T>
//__device__ typename list<T>::const_iterator list<T>::rbegin() const {
//	return list<T>::const_iterator(this->m_tail);
//}
//template <typename T>
//__device__ typename list<T>::iterator list<T>::rbegin() {
//	return list<T>::iterator(this->m_tail);
//}
//template <typename T>
//__device__ typename list<T>::const_iterator list<T>::rend() const {
//	return list<T>::const_iterator();
//}
//
//template <typename T>
//__device__ typename list<T>::iterator list<T>::rend() {
//	return list<T>::iterator();
//}
template <typename T>
__device__ void list<T>::swap(list &that) {
	std::swap(this->m_head, that.m_head);
	std::swap(this->m_tail, that.m_tail);
	std::swap(this->m_elements, that.m_elements);
}

template <typename T>
__device__ void list<T>::clear() {
	node<T>* curr = this->m_head;
	while (this->m_head)
	{
		curr = this->m_head;
		this->m_head = this->m_head->next;
		delete curr;
	}
	this->m_tail = nullptr;
	this->m_elements = 0;
}

CU_END

#endif // !CUDA_LIST_CUH
