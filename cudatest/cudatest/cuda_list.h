#include <iostream>
#include <type_traits>
//#include <stdexcept>
//#include "cuda_memory.h"
#include "cuda_config.h"

#ifndef CUDA_LIST_CUH
#define CUDA_LIST_CUH


template <class Derived, class Node>
class cuda_list_iterator_base;

template <typename T>
class cuda_list;

template <class T>
class cuda_list_iterator;

template <class T>
class cuda_list_const_iterator;

template <class Derived, class Node>
class cuda_list_iterator_base//:impl_cuda_list_iterator_proxy<Derived, Node>
{
	Node* it_value = nullptr;
public:
	//typedef typename std::conditional<std::is_const<Node>::value, const typename Node::value_type, typename Node::value_type>::type value_type;
	typedef std::conditional_t<std::is_const<Node>::value, const typename Node::value_type, typename Node::value_type> value_type;
	//typedef std::conditional_t<std::is_const_v<Node>, const typename Node::value_type, typename Node::value_type> value_type;
	typedef value_type& reference;
	typedef value_type* pointer;
	typedef std::ptrdiff_t difference_type;
	typedef std::bidirectional_iterator_tag iterator_category;

	__device__ cuda_list_iterator_base() = default;
	__device__ explicit cuda_list_iterator_base(Node* pNode) :it_value(pNode) {}
	//__device__ cuda_list_iterator_base(const cuda_list_iterator_base& right) : it_value(right.it_value) {}

	__device__ reference operator*() const
	{
		return it_value->data;
	}
	__device__ pointer operator->() const
	{
		return &it_value->data;
	}
	__device__ Derived& operator++()
	{
		it_value = it_value->next;
		return static_cast<Derived&>(*this);
	}
	__device__ Derived operator++(int)
	{
		auto old = static_cast<Derived&>(*this);
		++static_cast<Derived&>(*this);
		return old;
	}
	__device__ Derived& operator--()
	{
		it_value = it_value->prev;
		return static_cast<Derived&>(*this);
	}
	__device__ Derived operator--(int)
	{
		auto old = static_cast<Derived&>(*this);
		--static_cast<Derived&>(*this);
		return old;
	}
	__device__ bool operator==(const cuda_list_iterator_base& right) const
	{
		return it_value == right.it_value;
	}
	__device__ bool operator!=(const cuda_list_iterator_base& right) const
	{
		return it_value != right.it_value;
	}
protected:
	__device__ Node* it_val() const
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
		while(pNodeNext != nullptr)
		{
			auto tmp = pNodeNext;
			pNodeNext = pNodeNext->next;
			pNodeNext->prev = nullptr;
			delete tmp;
		}
		while (pNodePrev != nullptr)
		{
			auto tmp = pNodePrev;
			pNodePrev = pNodePrev->prev;
			pNodePrev->next = nullptr;
			delete tmp;
		}
	}
};

template <typename T>
class cuda_list;

template <class T>
class cuda_list_iterator;

template <class T>
class cuda_list_const_iterator :public cuda_list_iterator_base<cuda_list_const_iterator<T>, const node<T>>
{
	friend class cuda_list<T>;
public:
	//typedef node<T> value_type;
	using cuda_list_iterator_base<cuda_list_const_iterator<T>, const node<T>>::cuda_list_iterator_base;
	__device__ cuda_list_const_iterator(const cuda_list_iterator<T>& right) :cuda_list_iterator_base(right) {}
};

template <class T>
class cuda_list_iterator :public cuda_list_iterator_base<cuda_list_iterator<T>, node<T>>
{
	friend class cuda_list<T>;
public:
	//typedef node<T> value_type;
	using cuda_list_iterator_base<cuda_list_iterator<T>, node<T>>::cuda_list_iterator_base;
};

template <class T,
	bool is_move_constr = std::is_move_constructible<T>::value,
	bool is_copy_constr = std::is_copy_constructible<T>::value,
	bool is_move_assign = std::is_move_assignable<T>::value, 
	bool is_copy_assign = std::is_copy_assignable<T>::value>
struct cuda_list_proxy;

template <class T, bool /*ignored*/, class derived_class = cuda_list_proxy<T>>
struct cuda_list_proxy_move_constr
{
	cuda_list_proxy_move_constr() = default;
	cuda_list_proxy_move_constr(const cuda_list_proxy_move_constr&) = default;
	cuda_list_proxy_move_constr(cuda_list_proxy_move_constr&& right)
	{
		*this = std::move(right);
	}
	cuda_list_proxy_move_constr& operator=(const cuda_list_proxy_move_constr&) = default;
	cuda_list_proxy_move_constr& operator=(cuda_list_proxy_move_constr&& right)
	{
		if (this != &right)
		{
			this->derived().head = right.derived().head;
			this->derived().tail = right.derived().tail;
			this->derived().elements = right.derived().elements;

			right.derived().head = nullptr;
			right.derived().tail = nullptr;
			right.derived().elements = 0;
		}
		return *this;
	}
private:
	const derived_class& derived() const noexcept {return static_cast<const derived_class&>(*this);}
	derived_class& derived() noexcept {return static_cast<derived_class&>(*this);}
};

template <class T, bool /*ignore*/, class derived_class = cuda_list_proxy<T>>
struct cuda_list_proxy_move_assign:cuda_list_proxy_move_constr<T, true, derived_class> {};

template <class T, bool is_copy_constr, class derived_class = cuda_list_proxy<T>> struct cuda_list_proxy_copy_constr {};

template <class T, class derived_class> struct cuda_list_proxy_copy_constr<T, true, derived_class> 
{
	cuda_list_proxy_copy_constr() = default;
	cuda_list_proxy_copy_constr(const cuda_list_proxy_copy_constr& right)
	{
		if (!right.derived().elements)
		{
			this->derived().head = this->derived().tail = nullptr;
			this->derived().elements = 0;
		}
		else
		{
			node<T> *pNode = right.derived().head, *pHead = make_cuda_device_unique_ptr<node<T>>(pNode->data, nullptr, nullptr).release(), *pTail = pHead;
			while ((pNode = pNode->next) != nullptr)
			{
				pTail->next = make_cuda_device_unique_ptr<node<T>>(pNode->data, nullptr, pTail).release();
				pTail = pTail->next;
			}
			this->derived().head = pHead;
			this->derived().tail = pTail;
		}
		this->derived().elements = right.derived().elements;
	}
	cuda_list_proxy_copy_constr(cuda_list_proxy_copy_constr&&) = default;
	cuda_list_proxy_copy_constr& operator=(const cuda_list_proxy_copy_constr&) = default;
	cuda_list_proxy_copy_constr& operator=(cuda_list_proxy_copy_constr&&) = default;
private:
	const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

template <class T, bool is_copy_assign, class derived_class = cuda_list_proxy<T>> struct cuda_list_proxy_copy_assign {};

template <class T, class derived_class> struct cuda_list_proxy_copy_assign<T, true, derived_class> 
{
	cuda_list_proxy_copy_assign() = default;
	cuda_list_proxy_copy_assign(const cuda_list_proxy_copy_assign&) = default;
	cuda_list_proxy_copy_assign(cuda_list_proxy_copy_assign&&) = default;
	cuda_list_proxy_copy_assign operator=(const cuda_list_proxy_copy_assign& right)
	{
		if (this != &right)
		{

			auto pLeft = this->derived().head, pRight = right.derived().head;
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
			}else if (!pRight->next)
			{
				for (auto pLeftNext = pLeft->next; pLeftNext != nullptr; )
				{
					auto pCurrent = pLeftNext;
					pLeftNext = pLeftNext->next;
					delete pCurrent;
				}
				pLeft->next = nullptr;
			}
			pLeft->elements = pRight->elements;
		}
		return *this;
	}
	cuda_list_proxy_copy_assign& operator=(cuda_list_proxy_copy_assign&&) = default;
private:
	const derived_class& derived() const noexcept { return static_cast<const derived_class&>(*this); }
	derived_class& derived() noexcept { return static_cast<derived_class&>(*this); }
};

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable:4584)
#endif

template <class T, bool is_move_constr, bool is_copy_constr, bool is_move_assign, bool is_copy_assign>
struct cuda_list_proxy
	:cuda_list_proxy_move_constr<T, is_move_constr>, cuda_list_proxy_move_assign<T, is_move_assign>,
	cuda_list_proxy_copy_constr<T, is_copy_constr>, cuda_list_proxy_copy_assign<T, is_copy_assign>
{
	node<T>* head = nullptr, *tail = nullptr;
	std::size_t elements = 0;
};

#ifdef _MSC_VER
#pragma warning (pop)
#endif

template <typename T>
class cuda_list:public cuda_list_proxy<T>
{
public:
	typedef cuda_list_iterator<T> iterator;
	typedef cuda_list_const_iterator<T> const_iterator;
	//__host__ __device__ cuda_list() = default;
	__host__ __device__ ~cuda_list();

	template <class U>
	__host__ __device__ void push_back(U&& data);
	template <class U>
	__host__ __device__ void push_front(U&& data);
	//__host__ __device__ void pop_back();
	//__host__ __device__ void pop_front();
	__host__ __device__ void swap(cuda_list &x);
	__host__ __device__ void clear();
	__device__ iterator erase(iterator pos);

	__host__ __device__ const_iterator begin() const;
	__host__ __device__ iterator begin();

	__host__ __device__ const_iterator end() const;
	__host__ __device__ iterator end();

	__host__ __device__ const_iterator rbegin() const;
	__host__ __device__ iterator rbegin();

	__host__ __device__ const_iterator rend() const;
	__host__ __device__ iterator rend();

	__host__ __device__ size_t size() const;
	__host__ __device__ bool empty() const;

	__host__ __device__ T& front();
	__host__ __device__ T const& front() const;

	__host__ __device__ T& back();
	__host__ __device__ T const& back() const;
};

template <typename T>
__host__ __device__ cuda_list <T>::~cuda_list() 
{
	while (this->head)
		delete this->head;
}

template <typename T>
__host__ __device__ T& cuda_list<T>::front() {

	return this->head->data;
}

template <typename T>
__host__ __device__ T const& cuda_list<T>::front() const {
	return this->head->data;
}

template <typename T>
__host__ __device__ T& cuda_list<T>::back() {
	return this->tail->data;
}

template <typename T>
__host__ __device__ T const& cuda_list<T>::back() const {

	return this->tail->data;
}

template <typename T> template <class U>
__host__ __device__ void cuda_list<T>::push_back(U&& data) {
	node<T>* newNode = new node<T>(std::forward<U>(data), nullptr, this->tail);
	if (this->head == nullptr)
		this->head = newNode;
	if (this->tail != nullptr)
		this->tail->next = newNode;
	this->tail = newNode;
	++this->elements;
}

template <typename T>
__device__ typename cuda_list<T>::iterator cuda_list<T>::erase(typename cuda_list<T>::iterator pos)
{
	if (this->size() == 0)
		return cuda_list<T>::iterator();

	auto tmp = std::move(pos);

	if (tmp == this->begin())
	{
		this->head = (++tmp).it_val();
		//tmp->next->prev = nullptr;
		--tmp = cuda_list<T>::iterator();
	}
	else
	{
		if (tmp != this->end())
		{
			this->tail = (--tmp).it_val();
			//tmp->prev->next = nullptr;
			++tmp = cuda_list<T>::iterator();
		}
		else
		{
			//pos->next->prev = tmp->prev;
			++pos;
			--pos = --tmp;

			//pos->prev->next = tmp->next;
			--pos;
			--pos;
			++tmp;
			++pos = ++tmp;


		}
	}

	--this->elements;
	//node* nxt = tmp->next;
	//delete tmp;
	++tmp;
	return tmp;
}
//template <typename T>
//__host__ __device__ void cuda_list<T>::push_back(T&& data)
//{
//	node* newNode = new node(std::move(data), nullptr, this->tail);
//	if (this->head == nullptr)
//		this->head = newNode;
//	if (this->tail != nullptr)
//		this->tail->next = newNode;
//	this->tail = newNode;
//	++this->elements;
//}

template <typename T> template <class U>
__host__ __device__ void cuda_list<T>::push_front(U&& data) {
	node<T>* newNode = new node<T>(std::forward<U>(data), this->head, nullptr);
	if (this->tail == nullptr)
		this->tail = newNode;
	if (this->head != nullptr)
		this->head->prev = newNode;
	this->head = newNode;
	++this->elements;
}

//template <typename T>
//__host__ __device__ void cuda_list<T>::pop_front() 
//{
//	node *tmp = this->head;
//	if(this->size() == 1)
//	{
//		this->head = this->tail = nullptr;
//	}
//	else
//	{
//		this->head = this->head->next;
//		this->head->prev = nullptr;
//	}
//	--this->elements;
//	delete tmp;
//}
//
//template <typename T>
//__host__ __device__ void cuda_list<T>::pop_back() {
//
//	node *tmp = this->tail;
//	if(this->size() == 1)
//	{
//		this->head = this->tail = nullptr;
//	}
//	else
//	{
//		this->tail = this->tail->prev;
//		this->tail->next = nullptr;
//	}
//	--this->elements;
//	delete tmp;
//}

template <typename T>
__host__ __device__ bool cuda_list<T>::empty() const {
	return this->head == nullptr;
}

template <typename T>
__host__ __device__ size_t cuda_list<T>::size() const {
	return this->elements;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::begin() const {
	return cuda_list<T>::const_iterator(this->head);
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::begin() {
	return cuda_list<T>::iterator(this->head);
}

template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::end() const {
	return cuda_list<T>::const_iterator();
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::end() {
	return cuda_list<T>::iterator();
}


template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::rbegin() const {
	return cuda_list<T>::const_iterator(this->tail);
}
template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::rbegin() {
	return cuda_list<T>::iterator(this->tail);
}
template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::rend() const {
	return cuda_list<T>::const_iterator();
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::rend() {
	return cuda_list<T>::iterator();
}
template <typename T>
__host__ __device__ void cuda_list<T>::swap(cuda_list &that) {
	std::swap(this->head, that.head);
	std::swap(this->tail, that.tail);
	std::swap(this->elements, that.elements);
}

template <typename T>
__host__ __device__ void cuda_list<T>::clear() {
	node<T>* curr = this->head;
	while (this->head) {
		curr = this->head;
		this->head = this->head->next;
		delete curr;
	}
	this->tail = nullptr;
	this->elements = 0;
}

#endif // !CUDA_LIST_CUH
