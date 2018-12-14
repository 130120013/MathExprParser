//#ifndef list_H
//#define list_H
//
//#include <iostream>
//#include <stdexcept>
//
//template <typename T>
//class list {
//public:
//	__host__ __device__ list <T> & operator = (const list<T> &);
//	__host__ __device__ ~list();
//	/* Modifiers */
//	__host__ __device__ void push_back(T&& data);
//	__host__ __device__ void push_back(T const& data);
//	__host__ __device__ void push_front(T&& data);
//	__host__ __device__ void push_front(T const& data);
//	__host__ __device__ void pop_back();
//	__host__ __device__ void pop_front();
//	__host__ __device__ void swap(list &x);
//	__host__ __device__ void clear();
//
//	/* Iterators */
//	typedef T* iterator;
//	typedef T* const const_iterator;
//
//	__host__ __device__ const_iterator begin() const; // cbegin
//	__host__ __device__ iterator begin();
//
//	__host__ __device__ const_iterator end() const; //cend()
//	__host__ __device__ iterator end();
//
//	__host__ __device__ const_iterator rbegin() const;
//	__host__ __device__ iterator rbegin();
//
//	__host__ __device__ const_iterator rend() const;
//	__host__ __device__ iterator rend();
//
//	/* Capacity */
//	__host__ __device__ size_t size() const;
//	__host__ __device__ bool empty() const;
//
//	/* Element Access */
//	__host__ __device__ T& front();
//	__host__ __device__ T const& front() const;
//
//	__host__ __device__ T& back();
//	__host__ __device__ T const& back() const;
//
//	__host__ __device__ T& at(T const indx);
//	__host__ __device__ T const& at(T const indx) const;
//
//	__host__ __device__ T& operator[] (T const indx);
//	__host__ __device__ T const& operator[] (T const indx) const;
//
//private:
//	struct node {
//		int data;
//		node *next, *prev;
//		__host__ __device__ node(T const& data, node* next, node* prev)
//			: data(data)
//			, next(next)
//			, prev(prev) {
//		}
//		__host__ __device__ node(T&& data, node* next, node* prev)
//			: data(std::move(data))
//			, next(next)
//			, prev(prev) {
//		}
//	};
//	int elements = 0;
//	node *head = nullptr;
//	node *tail = nullptr;
//};
//
//template <typename T>
//list <T> & list<T>::operator = (const list<T> & that) {
//	node* tmp = head;
//	while (head) {
//		tmp = head;
//		head = head->next;
//		delete tmp;
//	}
//	elements = that.elements;
//	head = that.head;
//	tail = that.tail;
//}
//
//template <typename T>
//list <T>::~list() {
//	node* tmp;
//	while (head) {
//		tmp = head;
//		head = head->next;
//		delete tmp;
//	}
//}
//
//
//template <typename T>
//T& list<T>::front() {
//	if (head == nullptr)
//		throw std::runtime_error("Invalid Action!");
//	return head->data;
//}
//
//template <typename T>
//T const& list<T>::front() const {
//	if (head == nullptr)
//		throw std::runtime_error("Invalid Action!");
//	return head->data;
//}
//
//template <typename T>
//T& list<T>::back() {
//	if (tail == nullptr)
//		throw std::runtime_error("Invalid Action!");
//	return tail->data;
//}
//
//template <typename T>
//T const& list<T>::back() const {
//	if (tail == nullptr)
//		throw std::runtime_error("Invalid Action!");
//	return tail->data;
//}
//
//template <typename T>
//void list<T>::push_back(T const& data) {
//	node* newNode = new node(data, nullptr, tail);
//	if (head == nullptr)
//		head = newNode;
//	if (tail != nullptr)
//		tail->next = newNode;
//	tail = newNode;
//	++elements;
//}
//
//template <typename T>
//void list<T>::push_back(T&& data) {
//	node* newNode = new node(std::move(data), nullptr, tail);
//	if (head == nullptr)
//		head = newNode;
//	if (tail != nullptr)
//		tail->next = newNode;
//	tail = newNode;
//	++elements;
//}
//
//template <typename T>
//void list<T>::push_front(T const& data) {
//	node* newNode = new node(data, head, nullptr);
//	if (tail == nullptr)
//		tail = newNode;
//	if (head != nullptr)
//		head->prev = newNode;
//	head = newNode;
//	++elements;
//}
//
//template <typename T>
//void list<T>::push_front(T&& data) {
//	node* newNode = new node(data, head, nullptr);
//	if (tail == nullptr)
//		tail = newNode;
//	if (head != nullptr)
//		head->prev = newNode;
//	head = newNode;
//	++elements;
//}
//
//template <typename T>
//void list<T>::pop_front() {
//	if (head == nullptr)
//		throw std::runtime_error("Invalid Action");
//	node *tmp = head;
//	head = head->next;
//	if (head != nullptr)
//		head->prev = nullptr;
//	--elements;
//	delete tmp;
//}
//
//template <typename T>
//void list<T>::pop_back() {
//	if (tail == nullptr)
//		throw std::runtime_error("Invalid Action");
//	node *tmp = tail;
//	tail = tail->prev;
//	if (tail != nullptr)
//		tail->next = nullptr;
//	--elements;
//	delete tmp;
//}
//
//template <typename T>
//bool list<T>::empty() const {
//	return head == nullptr;
//}
//
//template <typename T>
//size_t list<T>::size() const {
//	return elements;
//}
//
//template <typename T>
//T& list<T>::operator[] (T const indx) {
//	int cont = 0;
//	node *curr = head;
//	while (curr) {
//		if (cont == indx)
//			return curr->data;
//		curr = curr->next;
//		++cont;
//	}
//	return nullptr;
//}
//
//template <typename T>
//T const& list<T>::operator[] (T const indx) const {
//	int cont = 0;
//	node *curr = head;
//	while (curr) {
//		if (cont == indx)
//			return curr->data;
//		curr = curr->next;
//		++cont;
//	}
//	return nullptr;
//}
//
//template <typename T>
//T& list<T>::at(T const indx) {
//	int cont = 0;
//	node *curr = head;
//	while (curr) {
//		if (cont == indx)
//			return curr->data;
//		curr = curr->next;
//	}
//	return nullptr;
//}
//
//template <typename T>
//T const& list<T>::at(T const indx) const {
//	int cont = 0;
//	node *curr = head;
//	while (curr) {
//		if (cont == indx)
//			return curr->data;
//		curr = curr->next;
//	}
//	return nullptr;
//}
//
//template <typename T>
//typename list<T>::const_iterator list<T>::begin() const {
//	return head;
//}
//
//template <typename T>
//typename list<T>::iterator list<T>::begin() {
//	return head;
//}
//
//
//template <typename T>
//typename list<T>::const_iterator list<T>::end() const {
//	return tail;
//}
//
//template <typename T>
//typename list<T>::const_iterator list<T>::rbegin() const {
//	return tail;
//}
//template <typename T>
//typename list<T>::iterator list<T>::rbegin() {
//	return tail;
//}
//template <typename T>
//typename list<T>::const_iterator list<T>::rend() const {
//	return head;
//}
//
//template <typename T>
//typename list<T>::iterator list<T>::rend() {
//	return head;
//}
//
//template <typename T>
//typename list<T>::iterator list<T>::end() {
//	return tail;
//}
//
//template <typename T>
//void list<T>::swap(list &that) {
//	std::swap(head, that.head);
//	std::swap(tail, that.tail);
//	std::swap(elements, that.elements);
//}
//
//template <typename T>
//void list<T>::clear() {
//	node* curr = head;
//	while (head) {
//		curr = head;
//		head = head->next;
//		delete curr;
//	}
//	head = tail = nullptr;
//	elements = 0;
//}
//
//#endif // list_H
