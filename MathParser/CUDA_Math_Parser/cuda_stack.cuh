#include "cuda_config.cuh"

#ifndef CUDA_STACK_CUH
#define CUDA_STACK_CUH

CU_BEGIN
template <typename T>
class stack
{
	struct node
	{
		T data;
		node* next;

		__device__ node(T const& data, node* next) : data(data), next(next) {}
		__device__ node(T&& data, node* next) : data(std::move(data)), next(next) {}
	};

public:
	__device__ ~stack();
	__device__ void push(T const& data);
	__device__ void push(T&& data);
	__device__ bool empty() const;
	__device__ int size() const;
	__device__ T& top();
	__device__ const T& top() const;
	__device__ void pop();

private:
	node* root = nullptr;
	int elements = 0;
};

template<typename T>
__device__ stack<T>::~stack()
{
	node* next;
	for (node* loop = root; loop != nullptr; loop = next)
	{
		next = loop->next;
		delete loop;
	}
}
template<typename T>
__device__ void stack<T>::push(const T& data)
{
	root = new node(data, root);
	++elements;
}
template<typename T>
__device__ void stack<T>::push(T&& data)
{
	root = new node(std::move(data), root);
	++elements;
}
template<typename T>
__device__ bool stack<T>::empty() const
{
	return root == nullptr;
}
template<typename T>
__device__ int stack<T>::size() const
{
	return elements;
}
template<typename T>
__device__ const T& stack<T>::top() const
{
	return root->data;
}
template<typename T>
__device__ T& stack<T>::top()
{
	return root->data;
}
template<typename T>
__device__ void stack<T>::pop()
{
	if (root == nullptr)
	{
		return;
	}
	node* tmp = root;
	root = root->next;
	--elements;
	delete tmp;
}

CU_END

#endif // !CUDA_STACK_CUH
