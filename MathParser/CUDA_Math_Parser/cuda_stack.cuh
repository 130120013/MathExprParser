#ifndef CUDA_STACK_CUH
#define CUDA_STACK_CUH
#include <iostream>

template <typename T>
class cuda_stack 
{
	struct node 
	{
		T data;
		node* next;

		__host__ __device__ node(T const& data, node* next): data(data), next(next) {}
		__host__ __device__ node(T&& data, node* next): data(std::move(data)), next(next) {}
	};

public:
	__host__ __device__ ~cuda_stack();
	__host__ __device__ void push(T const& data);
	__host__ __device__ void push(T&& data);
	__host__ __device__ bool empty() const;
	__host__ __device__ int size() const;
	__host__ __device__ T& top() ;
	__host__ __device__ const T& top() const;
	__host__ __device__ void pop();

private:
	node* root = nullptr;
	int elements = 0;
};

template<typename T>
cuda_stack<T>::~cuda_stack()
{
	node* next;
	for (node* loop = root; loop != nullptr; loop = next)
	{
		next = loop->next;
		delete loop;
	}
}
template<typename T>
void cuda_stack<T>::push(T const& data)
{
	root = new node(data, root);
	++elements;
}
template<typename T>
void cuda_stack<T>::push(T&& data)
{
	root = new node(std::move(data), root);
	++elements;
}
template<typename T>
bool cuda_stack<T>::empty() const
{
	return root == nullptr;
}
template<typename T>
int cuda_stack<T>::size() const
{
	return elements;
}
template<typename T>
const T& cuda_stack<T>::top() const
{
	return root->data;
}
template<typename T>
T& cuda_stack<T>::top() 
{
	return root->data;
}
template<typename T>
void cuda_stack<T>::pop()
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

#endif // !CUDA_STACK_CUH
