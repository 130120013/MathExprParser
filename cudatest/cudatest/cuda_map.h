#include "cuda_iterator.h"
#include "cuda_pair.h"
#include "impl_rb_tree.h"

#include "cuda_config.h"

#ifndef CUDAMAP_CUH
#define CUDAMAP_CUH

CU_BEGIN

	template <class Key, class Tp>
	class cuda_map_value_compare
	{
		template <class PairKey, class PairValue>
		using pair = cu::cuda_pair<PairKey, PairValue>;
	public:
		struct key_compare
		{
			__device__ bool operator() (const Key& left, const Key& right) const
			{
				return left < right;
			}
		};
		__device__ key_compare key_comp() const { return key_compare(); }

		template <class KeyLeft, class ValueLeft, class KeyRight, class ValueRight>
		__device__ bool operator()(const pair<KeyLeft, ValueLeft>& l, const pair<KeyRight, ValueRight>& r) const
		{
			return this->key_comp()(l.first, r.first);
		}

		template <class PairKey, class PairValue>
		__device__ bool operator()(const pair<PairKey, PairValue>& l, const Key& r) const
		{
			return this->key_comp()(l.first, r);
		}

		template <class PairKey, class PairValue>
		__device__ bool operator()(const Key& l, const pair<PairKey, PairValue>& r) const
		{
			return this->key_comp()(l, r.first);
		}
	};

	template <class _Allocator>
	class cuda_map_node_destructor
	{
		typedef _Allocator                          allocator_type;
		typedef std::allocator_traits<allocator_type>    __alloc_traits;
		typedef typename __alloc_traits::value_type::value_type value_type;
	public:
		typedef typename __alloc_traits::pointer    pointer;
	private:
		typedef typename value_type::first_type     first_type;
		typedef typename value_type::second_type    second_type;

		allocator_type& __na_;

		__device__ cuda_map_node_destructor& operator=(const cuda_map_node_destructor&) = default;

	public:
		bool __first_constructed;
		bool __second_constructed;

		explicit __device__ cuda_map_node_destructor(allocator_type& __na)
			: __na_(__na),
			__first_constructed(false),
			__second_constructed(false)
		{}

		__device__ void operator()(pointer ptr)
		{
			if (__second_constructed)
				__alloc_traits::destroy(__na_, addressof(ptr->value.second));
			if (__first_constructed)
				__alloc_traits::destroy(__na_, addressof(ptr->value.first));
			if (ptr)
				__alloc_traits::deallocate(__na_, ptr, 1);
		}
	};

	template <class _TreeIterator> class cuda_map_const_iterator;

	template <class _TreeIterator>
	class cuda_map_iterator
	{
		_TreeIterator m_itTree;

		typedef const typename _TreeIterator::value_type::first_type key_type;
		typedef typename _TreeIterator::value_type::second_type      mapped_type;
	public:
		typedef std::bidirectional_iterator_tag                           iterator_category;
		typedef cu::cuda_pair<key_type, mapped_type>                      value_type;
		typedef typename _TreeIterator::difference_type              difference_type;
		typedef value_type&                                          reference;
		typedef typename std::pointer_traits<typename _TreeIterator::pointer>::template
			rebind<value_type>                      pointer;

		__device__ cuda_map_iterator() {}

		__device__ cuda_map_iterator(_TreeIterator itTree) : m_itTree(itTree) {}

		__device__ reference operator*() const { return *operator->(); }
		__device__ pointer operator->() const { return (pointer)m_itTree.operator->(); }

		__device__ cuda_map_iterator& operator++() { ++m_itTree; return *this; }
		__device__ cuda_map_iterator operator++(int)
		{
			auto old = *this;
			++(*this);
			return old;
		}

		__device__ cuda_map_iterator& operator--() { --m_itTree; return *this; }
		__device__ cuda_map_iterator operator--(int)
		{
			auto old = *this;
			--(*this);
			return old;
		}

		friend __device__ bool operator==(const cuda_map_iterator& left, const cuda_map_iterator& right)
		{
			return left.m_itTree == right.m_itTree;
		}
		friend
			__device__ bool operator!=(const cuda_map_iterator& left, const cuda_map_iterator& right)
		{
			return left.m_itTree != right.m_itTree;
		}

		template <class, class, class> friend class cuda_map;
		template <class> friend class cuda_map_const_iterator;
	};

	template <class _TreeIterator>
	class cuda_map_const_iterator
	{
		_TreeIterator m_itTree;

		typedef typename _TreeIterator::pointer_traits             pointer_traits;
		typedef const typename _TreeIterator::value_type::first_type __key_type;
		typedef typename _TreeIterator::value_type::second_type      __mapped_type;
	public:
		typedef std::bidirectional_iterator_tag                           iterator_category;
		typedef cu::cuda_pair<__key_type, __mapped_type>                      value_type;
		typedef typename _TreeIterator::difference_type              difference_type;
		typedef const value_type&                                    reference;
		typedef typename pointer_traits::template
			rebind<const value_type>                      pointer;

		__device__ cuda_map_const_iterator() {}

		__device__ cuda_map_const_iterator(_TreeIterator itTree) : m_itTree(itTree) {}
		__device__ cuda_map_const_iterator(
			cuda_map_iterator<typename _TreeIterator::non_const_iterator> itTree)

			: m_itTree(itTree.m_itTree) {}

		__device__ reference operator*() const { return *operator->(); }
		__device__ pointer operator->() const { return (pointer)m_itTree.operator->(); }

		__device__ cuda_map_const_iterator& operator++() { ++m_itTree; return *this; }
		__device__ cuda_map_const_iterator operator++(int)
		{
			auto old = *this;
			++(*this);
			return old;
		}

		__device__ cuda_map_const_iterator& operator--() { --m_itTree; return *this; }
		__device__ cuda_map_const_iterator operator--(int)
		{
			auto old = *this;
			--(*this);
			return old;
		}

		friend __device__ bool operator==(const cuda_map_const_iterator& left, const cuda_map_const_iterator& right)
		{
			return left.m_itTree == right.m_itTree;
		}
		friend
			__device__ bool operator!=(const cuda_map_const_iterator& left, const cuda_map_const_iterator& right)
		{
			return left.m_itTree != right.m_itTree;
		}

		template <class, class, class> friend class cuda_map;
		template <class, class, class> friend class tree_const_iterator;
	};


	template <class Key, class T, class Compare>
	class cuda_map
	{
	public:
		// types:
		typedef Key key_type;
		typedef T mapped_type;
		typedef cu::cuda_pair<key_type, mapped_type> value_type;
		typedef cuda_map_value_compare<key_type, mapped_type> value_compare;
		typedef typename value_compare::key_compare key_compare;
		typedef value_type& reference;
		typedef const value_type& const_reference;

	private:
		typedef cu::cuda_pair<key_type, mapped_type> my_value_type;
		typedef cuda_map_value_compare<key_type, mapped_type> my_value_compare;
		typedef cuda_red_black_tree<my_value_type, my_value_compare>   my_base;
		//typedef typename my_base::__node_traits                 __node_traits;

	public:
		typedef typename my_base::pointer            pointer;
		typedef typename my_base::const_pointer      const_pointer;
		typedef typename my_base::size_type          size_type;
		typedef typename my_base::difference_type    difference_type;
		typedef cuda_map_iterator<typename my_base::iterator>   iterator;
		typedef cuda_map_const_iterator<typename my_base::const_iterator> const_iterator;
		typedef cuda_reverse_iterator<iterator>                  reverse_iterator;
		typedef cuda_reverse_iterator<const_iterator>            const_reverse_iterator;

	public:
		explicit __device__ cuda_map(const key_compare& = key_compare()) : tree(my_value_compare()) {}

		template <class _InputIterator>
		__device__ cuda_map(_InputIterator itBegin, _InputIterator itEnd,
			const key_compare& comp = key_compare())
			: tree(my_value_compare(comp))
		{
			insert(itBegin, itEnd);
		}

		__device__ cuda_map(const cuda_map& m)
			: tree(m.tree)
		{
			insert(m.begin(), m.end());
		}

		__device__ cuda_map& operator=(const cuda_map& m)
		{
			tree = m.tree;
			return *this;
		}
	public:
		// Iteration
		__device__ iterator begin() { return tree.begin(); }
		__device__  const_iterator begin() const { return tree.begin(); }
		__device__ iterator end() { return tree.end(); }
		__device__ const_iterator end() const { return tree.end(); }

		__device__ reverse_iterator rbegin() { return reverse_iterator(end()); }
		__device__ const_reverse_iterator rbegin() const
		{
			return const_reverse_iterator(end());
		}
		__device__ reverse_iterator rend()
		{
			return       reverse_iterator(begin());
		}
		__device__ const_reverse_iterator rend() const
		{
			return const_reverse_iterator(begin());
		}

	public:
		// Size
		__device__ bool      empty() const { return tree.size() == 0; }
		__device__ size_type size() const { return tree.size(); }
		__device__ size_type max_size() const { return tree.max_size(); }

	public:
		// Element Access
		__device__ mapped_type& operator[](const key_type& key);

		__device__ mapped_type& at(const key_type& key);
		__device__ const mapped_type& at(const key_type& key) const;

		__device__ cu::cuda_pair<iterator, bool>
			insert(const value_type& val) { return tree.insert_unique(val); }

		__device__ iterator
			insert(const_iterator itWhere, const value_type& val)
		{
			return tree.insert_unique(itWhere.m_itTree, val);
		}

		template <class _InputIterator>
		__device__ void insert(_InputIterator itBegin, _InputIterator itEnd)
		{
			for (const_iterator e = end(); itBegin != itEnd; ++itBegin)
				insert(e.m_itTree, *itBegin);
		}

		__device__ iterator erase(const_iterator itWhere) { return tree.erase(itWhere.m_itTree); }
		__device__ size_type erase(const key_type& key)
		{
			return tree.erase_unique(key);
		}
		__device__ iterator  erase(const_iterator itBegin, const_iterator itEnd)
		{
			return tree.erase(itBegin.m_itTree, itEnd.m_itTree);
		}
		__device__ void clear() { tree.clear(); }

		__device__ void swap(cuda_map& m)
		{
			tree.swap(m.tree);
		}

		__device__ iterator find(const key_type& key) { return tree.find(key); }
		__device__ const_iterator find(const key_type& key) const { return tree.find(key); }
		__device__ size_type      count(const key_type& key) const
		{
			return tree.count_unique(key);
		}
		__device__ iterator lower_bound(const key_type& key)
		{
			return tree.lower_bound(key);
		}
		__device__ const_iterator lower_bound(const key_type& key) const
		{
			return tree.lower_bound(key);
		}
		__device__ iterator upper_bound(const key_type& key)
		{
			return tree.upper_bound(key);
		}
		__device__ const_iterator upper_bound(const key_type& key) const
		{
			return tree.upper_bound(key);
		}
		__device__ cu::cuda_pair<iterator, iterator> equal_range(const key_type& key)
		{
			return tree.equal_range_unique(key);
		}
		__device__  cu::cuda_pair<const_iterator, const_iterator> equal_range(const key_type& key) const
		{
			return tree.equal_range_unique(key);
		}
		template <class ... Tk>
		__device__ cu::cuda_pair<iterator, bool> emplace(Tk&&... keys);

	public:
		// Member access
		__device__ key_compare    key_comp()      const { return tree.value_comp().key_comp(); }
		__device__ value_compare  value_comp()    const { return value_compare(tree.value_comp().key_comp()); }


	private:
		my_base tree;

	private:
		typedef typename my_base::node node;
		typedef typename my_base::node_pointer node_pointer;
		typedef typename my_base::node_const_pointer node_const_pointer;

		//typedef cuda_map_node_destructor<__node_allocator> _Dp;
		typedef cuda_device_unique_ptr<node> node_holder;

	private:
		template <class ... Tk, class = std::enable_if_t<std::is_constructible<cu::cuda_pair<Key, T>, Tk&&...>::value>>
		__device__ node_holder construct_node(Tk&& ... key);
		__device__ node_holder construct_node(const key_type& key);
		__device__ node_pointer& find_equal_key(node_pointer& parent, const key_type& key);
		__device__ node_pointer& find_equal_key(const_iterator hint,
			node_pointer& parent, const key_type& key);
		__device__ node_const_pointer
			find_equal_key(node_const_pointer& parent, const key_type& key) const;

	};

	template <class _Key, class _Tp, class _Compare>
	template <class ... Tk>
	__device__ cu::cuda_pair<typename cuda_map<_Key, _Tp, _Compare>::iterator, bool> cuda_map<_Key, _Tp, _Compare>::emplace(Tk&&... keys)
	{
		node_holder h = construct_node(std::forward<Tk>(keys)...);
		node_pointer parent;
		bool inserted = false;
		node_pointer& child = find_equal_key(parent, h->value.first);
		if (child == 0)
		{
			this->tree.insert_node_at(parent, child, h.release());
			inserted = true;
		}
		return cu::make_cuda_pair(iterator(typename my_base::iterator(child)), inserted);
	}

	template <class _Key, class _Tp, class _Compare>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::mapped_type& cuda_map<_Key, _Tp, _Compare>::at(const key_type& key)
	{
		node_pointer p;
		auto res = find_equal_key(p, key);
		/*if (res == nullptr)
			cuda_abort_with_error(CHSVERROR_NOT_FOUND);*/
		return res->value.second;
	}

	template <class _Key, class _Tp, class _Compare>
	__device__ const typename cuda_map<_Key, _Tp, _Compare>::mapped_type& cuda_map<_Key, _Tp, _Compare>::at(const key_type& key) const
	{
		node_const_pointer p;
		auto res = find_equal_key(p, key);
		/*if (res == nullptr)
			cuda_abort_with_error(CHSVERROR_NOT_FOUND);*/
		return res->value.second;
	}

	// Find place to insert if key doesn't exist
	// Set parent to parent of null leaf
	// Return reference to null leaf
	// If key exists, set parent to node of key and return reference to node of key
	template <class _Key, class _Tp, class _Compare>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::node_pointer&
		cuda_map<_Key, _Tp, _Compare>::find_equal_key(node_pointer& parent,
			const key_type& key)
	{
		node_pointer nd = tree.root();
		if (nd != 0)
		{
			while (true)
			{
				if (tree.value_comp().key_comp()(key, nd->value.first))
				{
					if (nd->left != 0)
						nd = static_cast<node_pointer>(nd->left);
					else
					{
						parent = nd;
						return parent->left;
					}
				}
				else if (tree.value_comp().key_comp()(nd->value.first, key))
				{
					if (nd->right != 0)
						nd = static_cast<node_pointer>(nd->right);
					else
					{
						parent = nd;
						return parent->right;
					}
				}
				else
				{
					parent = nd;
					return parent;
				}
			}
		}
		parent = tree.end_node();
		return parent->left;
	}

	// Find place to insert if key doesn't exist
	// First check prior to hint.
	// Next check after hint.
	// Next do O(log N) search.
	// Set parent to parent of null leaf
	// Return reference to null leaf
	// If key exists, set parent to node of key and return reference to node of key
	template <class _Key, class _Tp, class _Compare>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::node_pointer&
		cuda_map<_Key, _Tp, _Compare>::find_equal_key(const_iterator hint,
			node_pointer& parent,
			const key_type& key)
	{
		if (hint == end() || tree.value_comp().key_comp()(key, hint->first))  // check before
		{
			// key < *hint
			const_iterator prior = hint;
			if (prior == begin() || tree.value_comp().key_comp()((--prior)->first, key))
			{
				// *prev(hint) < key < *hint
				if (hint.ptr->left == nullptr)
				{
					parent = const_cast<node_pointer&>(hint.ptr);
					return parent->left;
				}
				else
				{
					parent = const_cast<node_pointer&>(prior.ptr);
					return parent->right;
				}
			}
			// key <= *prev(hint)
			return find_equal_key(parent, key);
		}
		else if (tree.value_comp().key_comp()(hint->first, key))  // check after
		{
			// *hint < key
			const_iterator itNext = next(hint);
			if (itNext == end() || tree.value_comp().key_comp()(key, itNext->first))
			{
				// *hint < key < *next(hint)
				if (hint.ptr->right == 0)
				{
					parent = const_cast<node_pointer&>(hint.ptr);
					return parent->right;
				}
				else
				{
					parent = const_cast<node_pointer&>(itNext.ptr);
					return parent->left;
				}
			}
			// *next(hint) <= key
			return find_equal_key(parent, key);
		}
		// else key == *hint
		parent = const_cast<node_pointer&>(hint.ptr);
		return parent;
	}

	// Find key
	// Set parent to parent of null leaf and
	//    return reference to null leaf iv key does not exist.
	// If key exists, set parent to node of key and return reference to node of key
	template <class _Key, class _Tp, class _Compare>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::node_const_pointer
		cuda_map<_Key, _Tp, _Compare>::find_equal_key(node_const_pointer& parent,
			const key_type& key) const
	{
		node_const_pointer nd = tree.root();
		if (nd != 0)
		{
			while (true)
			{
				if (tree.value_comp().key_comp()(key, nd->value.first))
				{
					if (nd->left != nullptr)
						nd = static_cast<node_pointer>(nd->left);
					else
					{
						parent = nd;
						return const_cast<const node_const_pointer&>(parent->left);
					}
				}
				else if (tree.value_comp().key_comp()(nd->value.first, key))
				{
					if (nd->right != nullptr)
						nd = static_cast<node_pointer>(nd->right);
					else
					{
						parent = nd;
						return const_cast<const node_const_pointer&>(parent->right);
					}
				}
				else
				{
					parent = nd;
					return parent;
				}
			}
		}
		parent = tree.end_node();
		return const_cast<const node_const_pointer&>(parent->left);
	}

	template <class _Key, class _Tp, class _Compare>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::node_holder
		cuda_map<_Key, _Tp, _Compare>::construct_node(const key_type& key)
	{
		return node_holder(new node(cu::make_cuda_pair(key, mapped_type())));
	}

	template <class _Key, class _Tp, class _Compare>
	template <class ... Tk, class>
	__device__ typename cuda_map<_Key, _Tp, _Compare>::node_holder
		cuda_map<_Key, _Tp, _Compare>::construct_node(Tk&& ... key)
	{
		return node_holder(new node(std::forward<Tk>(key)...));
	}

	template <class _Key, class _Tp, class _Compare>
	__device__ _Tp& cuda_map<_Key, _Tp, _Compare>::operator[](const key_type& key)
	{
		node_pointer parent;
		node_pointer& child = find_equal_key(parent, key);
		node_pointer r = static_cast<node_pointer>(child);
		if (child == 0)
		{
			node_holder h = construct_node(key);
			tree.insert_node_at(parent, child, h.get());
			r = h.release();
		}
		return r->value.second;
	}

CU_END

#endif //CUDAMAP_CUH
