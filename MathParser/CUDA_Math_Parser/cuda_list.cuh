#ifndef CUDA_LIST_CUH
#define CUDA_LIST_CUH

#include <iostream>
//#include <stdexcept>
#include "cuda_memory.cuh"

//TODO: make cuda_list iterator for which:
// std::is_same<typename std::iterator_traits<typename cuda_list<T>::iterator>::value_type, T>::value
// is true

template <typename T>
class cuda_list 
{
	struct node 
	{
		T data;
		node *next, *prev;
		__host__ __device__ node(T const& data, node* next, node* prev): data(data), next(next), prev(prev) {}
		__host__ __device__ node(T&& data, node* next, node* prev): data(std::move(data)), next(next), prev(prev) {}
	};
public:
	typedef node* iterator;
	typedef node* const const_iterator;

	__host__ __device__ cuda_list<T>& operator= (const cuda_list<T> &);
	__host__ __device__ ~cuda_list();

	//__host__ __device__ void push_back(T&& data);
	__host__ __device__ void push_back(T const& data);
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
	node *head = nullptr;
	node *tail = nullptr;
};

template <typename T>
__host__ __device__ cuda_list<T>& cuda_list<T>::operator= (const cuda_list<T> & that) {
	node* tmp = head;
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
	node* tmp;
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
__host__ __device__ void cuda_list<T>::push_back(T const& data) {
	node* newNode = new node(data, nullptr, tail);
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
		return nullptr;

	auto tmp = cuda_device_unique_ptr<node>(pos);

	if(tmp.get() == this->begin())
	{
		this->head = tmp->next;
		tmp->next->prev = nullptr;
	}
	else
	{
		if(tmp.get() != this->end())
		{
			this->tail = tmp->prev;
			tmp->prev->next = nullptr;
		}
		else
		{
			pos->next->prev = tmp->prev;
			pos->prev->next = tmp->next;
		}
	}
	
	--elements;
	//node* nxt = tmp->next;
	//delete tmp;
	return tmp->next;
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
	node* newNode = new node(data, head, nullptr);
	if (tail == nullptr)
		tail = newNode;
	if (head != nullptr)
		head->prev = newNode;
	head = newNode;
	++elements;
}

template <typename T>
__host__ __device__ void cuda_list<T>::push_front(T&& data) {
	node* newNode = new node(data, head, nullptr);
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
	return head;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::begin() {
	return head;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::end() const {
	return tail;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::rbegin() const {
	return tail;
}
template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::rbegin() {
	return tail;
}
template <typename T>
__host__ __device__ typename cuda_list<T>::const_iterator cuda_list<T>::rend() const {
	return head;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::rend() {
	return head;
}

template <typename T>
__host__ __device__ typename cuda_list<T>::iterator cuda_list<T>::end() {
	return tail;
}

template <typename T>
__host__ __device__ void cuda_list<T>::swap(cuda_list &that) {
	std::swap(head, that.head);
	std::swap(tail, that.tail);
	std::swap(elements, that.elements);
}

template <typename T>
__host__ __device__ void cuda_list<T>::clear() {
	node* curr = head;
	while (head) {
		curr = head;
		head = head->next;
		delete curr;
	}
	head = tail = nullptr;
	elements = 0;
}

#endif // !CUDA_LIST_CUH
