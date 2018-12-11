//#ifndef list_H
//#define list_H
//
//#include <iostream>
//#include <stdexcept>
//
//template <typename T>
//class list {
//public:
//	list <T> & operator = (const list<T> &);
//	~list();
//	/* Modifiers */
//	void push_back(T&& data);
//	void push_back(T const& data);
//	void push_front(T&& data);
//	void push_front(T const& data);
//	void pop_back();
//	void pop_front();
//	void swap(list &x);
//	void clear();
//
//	/* Iterators */
//	typedef T* iterator;
//	typedef T* const const_iterator;
//
//	const_iterator begin() const; // cbegin
//	iterator begin();
//
//	const_iterator end() const; //cend()
//	iterator end();
//
//	const_iterator rbegin() const;
//	iterator rbegin();
//
//	const_iterator rend() const;
//	iterator rend();
//
//	/* Capacity */
//	size_t size() const;
//	bool empty() const;
//
//	/* Element Access */
//	T& front();
//	T const& front() const;
//
//	T& back();
//	T const& back() const;
//
//	T& at(T const indx);
//	T const& at(T const indx) const;
//
//	T& operator[] (T const indx);
//	T const& operator[] (T const indx) const;
//
//private:
//	struct node {
//		int data;
//		node *next, *prev;
//		node(T const& data, node* next, node* prev)
//			: data(data)
//			, next(next)
//			, prev(prev) {
//		}
//		node(T&& data, node* next, node* prev)
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
//
//struct node {
//	void* value;
//
//	node* next;
//};
//
//struct stack {
//	node* first;
//	int elemSize;
//	int length;
//};
//
//stack* StackNew(int elemSize)
//{
//	stack* s = (stack*)malloc(sizeof(stack));
//	s->elemSize = elemSize;
//	s->length = 0;
//	return s;
//}
//
//void Push(stack* s, void* value)
//{
//	node* oldFirst = s->first;
//	s->first = (node*)malloc(sizeof(node));
//	s->first->value = malloc(s->elemSize);
//	memcpy(s->first->value, value, s->elemSize);
//	s->first->next = oldFirst;
//	s->length++;
//}
//
//void* Pop(stack* s)
//{
//	void* value = malloc(s->elemSize);
//	memcpy(value, s->first->value, s->elemSize);
//	node* oldFirst = s->first;
//	s->first = s->first->next;
//	free(oldFirst);
//	s->length--;
//	return value;
//}
//
//void Destory(stack* s)
//{
//	while (s->length > 0) Pop(s);
//	free(s);
//}