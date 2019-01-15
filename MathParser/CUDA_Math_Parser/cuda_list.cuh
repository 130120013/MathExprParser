#ifndef CUDA_LIST_CUH
#define CUDA_LIST_CUH

#include <iostream>
#include <type_traits>
//#include <stdexcept>
#include "cuda_memory.cuh"

//TODO: make cuda_list iterator for which:
// std::is_same<typename std::iterator_traits<typename cuda_list<T>::iterator>::value_type, T>::value
// is true

template <class Derived, class Node>
class cuda_list_iterator_base
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
	//__host__ __device__ node(T const& data, node* next, node* prev) : data(data), next(next), prev(prev) {}
	__host__ __device__ node(T&& data, node* next, node* prev) : data(std::move(data)), next(next), prev(prev) {}
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
	using cuda_list_iterator_base::cuda_list_iterator_base;
	__device__ cuda_list_const_iterator(const cuda_list_iterator<T>& right) :cuda_list_iterator_base(right) {}
};

template <class T>
class cuda_list_iterator :public cuda_list_iterator_base<cuda_list_iterator<T>, node<T>>
{
	friend class cuda_list<T>;
public:
	using cuda_list_iterator_base::cuda_list_iterator_base;
};

template <typename T>
class cuda_list 
{
	/*struct node 
	{
		T data;
		node *next, *prev;
		__host__ __device__ node(T const& data, node* next, node* prev): data(data), next(next), prev(prev) {}
		__host__ __device__ node(T&& data, node* next, node* prev): data(std::move(data)), next(next), prev(prev) {}
	};*/
public:
	typedef cuda_list_iterator<T> iterator;
	typedef cuda_list_const_iterator<T> const_iterator;

	__host__ __device__ cuda_list<T>& operator= (const cuda_list<T> &);
	__host__ __device__ ~cuda_list();

	//__host__ __device__ void push_back(T&& data);
	__host__ __device__ void push_back(T&& data);
	__host__ __device__ void push_front(T&& data);
	__host__ __device__ void push_front(T const& data);
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

private:
	int elements = 0;
	node<T> *head = nullptr;
	node<T> *tail = nullptr;
};

template <typename T>
__host__ __device__ cuda_list<T>& cuda_list<T>::operator= (const cuda_list<T> & that) {
	node<T>* tmp = head;
	while (head) {
		tmp = head;
		head = head->next;
		delete tmp;
	}
	elements = that.elements;
	head = that.head;
	tail = that.tail;
	return *this;
}

template <typename T>
__host__ __device__ cuda_list <T>::~cuda_list() {
	node<T>* tmp;
	while (head) {
		tmp = head;
		head = head->next;
		delete tmp;
	}
}

template <typename T>
__host__ __device__ T& cuda_list<T>::front() {

	return head->data;
}

template <typename T>
__host__ __device__ T const& cuda_list<T>::front() const {
	return head->data;
}

template <typename T>
__host__ __device__ T& cuda_list<T>::back() {
	return tail->data;
}

template <typename T>
__host__ __device__ T const& cuda_list<T>::back() const {

	return tail->data;
}

template <typename T>
__host__ __device__ void cuda_list<T>::push_back(T&& data) {
	node<T>* newNode = new node<T>(std::move(data), nullptr, tail);
	if (head == nullptr)
		head = newNode;
	if (tail != nullptr)
		tail->next = newNode;
	tail = newNode;
	++elements;
}

template <typename T>
__device__ typename::cuda_list<T>::iterator cuda_list<T>::erase(typename::cuda_list<T>::iterator pos)
{
	if(this->size() == 0)
		return cuda_list<T>::iterator();;

	auto tmp = std::move(pos);

	if(tmp == this->begin())
	{
		this->head = (++tmp).it_val();
		//tmp->next->prev = nullptr;
		--tmp = cuda_list<T>::iterator();
	}
	else
	{
		if(tmp != this->end())
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
	
	--elements;
	//node* nxt = tmp->next;
	//delete tmp;
	++tmp;
	return tmp;
}
//template <typename T>
//__host__ __device__ void cuda_list<T>::push_back(T&& data)
//{
//	node* newNode = new node(std::move(data), nullptr, tail);
//	if (head == nullptr)
//		head = newNode;
//	if (tail != nullptr)
//		tail->next = newNode;
//	tail = newNode;
//	++elements;
//}

template <typename T>
__host__ __device__ void cuda_list<T>::push_front(T const& data) {
	node<T>* newNode = new node<T>(data, head, nullptr);
	if (tail == nullptr)
		tail = newNode;
	if (head != nullptr)
		head->prev = newNode;
	head = newNode;
	++elements;
}

template <typename T>
__host__ __device__ void cuda_list<T>::push_front(T&& data) {
	node<T>* newNode = new node<T>(data, head, nullptr);
	if (tail == nullptr)
		tail = newNode;
	if (head != nullptr)
		head->prev = newNode;
	head = newNode;
	++elements;
}

//template <typename T>
//__host__ __device__ void cuda_list<T>::pop_front() 
//{
//	node *tmp = head;
//	if(this->size() == 1)
//	{
//		head = tail = nullptr;
//	}
//	else
//	{
//		head = head->next;
//		head->prev = nullptr;
//	}
//	--elements;
//	delete tmp;
//}
//
//template <typename T>
//__host__ __device__ void cuda_list<T>::pop_back() {
//
//	node *tmp = tail;
//	if(this->size() == 1)
//	{
//		head = tail = nullptr;
//	}
//	else
//	{
//		tail = tail->prev;
//		tail->next = nullptr;
//	}
//	--elements;
//	delete tmp;
//}

template <typename T>
__host__ __device__ bool cuda_list<T>::empty() const {
	return head == nullptr;
}

template <typename T>
__host__ __device__ size_t cuda_list<T>::size() const {
	return elements;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::begin() const {
	return cuda_list<T>::const_iterator(head);
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::begin() {
	return cuda_list<T>::iterator(head);
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
	return cuda_list<T>::const_iterator(tail);
}
template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::rbegin() {
	return cuda_list<T>::iterator(tail);
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
	std::swap(head, that.head);
	std::swap(tail, that.tail);
	std::swap(elements, that.elements);
}

template <typename T>
__host__ __device__ void cuda_list<T>::clear() {
	node<T>* curr = head;
	while (head) {
		curr = head;
		head = head->next;
		delete curr;
	}
	head = tail = nullptr;
	elements = 0;
}

#endif // !CUDA_LIST_CUH
