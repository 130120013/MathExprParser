#include <type_traits>
#include <memory>
#include <thrust/memory.h>
//#include <cuda/utils.h>
#include <thrust/swap.h>
//#include <chsvlib/chsvmetafun.h>
#include "cuda_memory.cuh"
#include "cuda_string.cuh"
#include "cuda_iterator.cuh"
#include "cuda_pair.cuh"

#ifndef CUDAREDBLACKTREE_H
#define CUDAREDBLACKTREE_H

namespace cu
{

	template <class Tp, class _Compare> class tree;
		template <class Tp, class NodePtr, class DiffType>
		class tree_iterator;
		template <class Tp, class _ConstNodePtr, class DiffType>
		class tree_const_iterator;
		template <class Key, class Tp>
		class cuda_map_value_compare;
		template <class Key, class Tp, class _Compare = typename  cuda_map_value_compare<Key, Tp>::key_compare>
		class cuda_map;
		/* template <class Key, class _Compare= typename  cuda_set_value_compare<Key, T>::key_compare>
		class cuda_set;*/

		/*

		NodePtr algorithms

		The algorithms taking NodePtr are red black tree algorithms.  Those
		algorithms taking a parameter named root should assume that root
		points to a proper red black tree (unless otherwise specified).

		Each algorithm herein assumes that root->parent points to a non-null
		structure which has a member left which points back to root.  No other
		member is read or written to at root->parent.

		root->parent will be referred to below (in comments only) as end_node.
		end_node->left is an externably accessible lvalue for root, and can be
		changed by node insertion and removal (without explicit reference to end_node).

		All nodes (with the exception of end_node), even the node referred to as
		root, have a non-null parent field.

		*/

		// Returns:  true if x is a left child of its parent, else false
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ inline bool
			tree_is_left_child(NodePtr x)
		{
			return x->parent != nullptr && x == x->parent->left;
		}

		// Determintes if the subtree rooted at x is a proper red black subtree.  If
		//    x is a proper subtree, returns the black height (null counts as 1).  If
		//    x is an improper subtree, returns 0.
		template <class NodePtr>
		__device__ unsigned
			tree_sub_invariant(NodePtr x)
		{
			if (x == 0)
				return 1;
			// parent consistency checked by caller
			// check x->left consistency
			if (x->left != 0 && x->left->parent != x)
				return 0;
			// check x->right consistency
			if (x->right != 0 && x->right->parent != x)
				return 0;
			// check x->left != x->right unless both are 0
			if (x->left == x->right && x->left != 0)
				return 0;
			// If this is red, neither child can be red
			if (!x->is_black)
			{
				if (x->left && !x->left->is_black)
					return 0;
				if (x->right && !x->right->is_black)
					return 0;
			}
			unsigned h = tree_sub_invariant(x->left);
			if (h == 0)
				return 0;  // invalid left subtree
			if (h != tree_sub_invariant(x->right))
				return 0;  // invalid or different height right subtree
			return h + x->is_black;  // return black height of this node
		}

		// Determintes if the red black tree rooted at root is a proper red black tree.
		//    root == 0 is a proper tree.  Returns true is root is a proper
		//    red black tree, else returns false.
		template <class NodePtr>
		__device__ bool
			tree_invariant(NodePtr root)
		{
			if (root == 0)
				return true;
			// check x->parent consistency
			if (root->parent == 0)
				return false;
			if (!tree_is_left_child(root))
				return false;
			// root must be black
			if (!root->is_black)
				return false;
			// do normal node checks
			return tree_sub_invariant(root) != 0;
		}

		// Returns:  pointer to the left-most node under x.
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ inline NodePtr
			tree_min(NodePtr x)
		{
			while (x->left != 0)
				x = x->left;
			return x;
		}

		// Returns:  pointer to the right-most node under x.
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ inline NodePtr
			tree_max(NodePtr x)
		{
			while (x->right != 0)
				x = x->right;
			return x;
		}

		// Returns:  pointer to the next in-order node after x.
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ NodePtr
			tree_next(NodePtr x)
		{
			if (x->right != 0)
				return tree_min(x->right);
			while (!tree_is_left_child(x))
				x = x->parent;
			return x->parent;
		}

		// Returns:  pointer to the previous in-order node before x.
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ NodePtr
			tree_prev(NodePtr x)
		{
			if (x->left != 0)
				return tree_max(x->left);
			while (tree_is_left_child(x))
				x = x->parent;
			return x->parent;
		}

		// Returns:  pointer to a node which has no children
		// Precondition:  x != 0.
		template <class NodePtr>
		__device__ NodePtr
			tree_leaf(NodePtr x)
		{
			while (true)
			{
				if (x->left != 0)
				{
					x = x->left;
					continue;
				}
				if (x->right != 0)
				{
					x = x->right;
					continue;
				}
				break;
			}
			return x;
		}

		// Effects:  Makes x->right the subtree root with x as its left child
		//           while preserving in-order order.
		// Precondition:  x->right != 0
		template <class NodePtr>
		__device__ void
			tree_left_rotate(NodePtr x)
		{
			NodePtr y = x->right;
			x->right = y->left;
			if (x->right != 0)
				x->right->parent = x;
			y->parent = x->parent;
			if (tree_is_left_child(x))
				x->parent->left = y;
			else
				x->parent->right = y;
			y->left = x;
			x->parent = y;
		}

		// Effects:  Makes x->left the subtree root with x as its right child
		//           while preserving in-order order.
		// Precondition:  x->left != 0
		template <class NodePtr>
		__device__ void
			tree_right_rotate(NodePtr x)
		{
			NodePtr y = x->left;
			x->left = y->right;
			if (x->left != 0)
				x->left->parent = x;
			y->parent = x->parent;
			if (tree_is_left_child(x))
				x->parent->left = y;
			else
				x->parent->right = y;
			y->right = x;
			x->parent = y;
		}

		// Effects:  Rebalances root after attaching x to a leaf.
		// Precondition:  root != nulptr && x != 0.
		//                x has no children.
		//                x == root or == a direct or indirect child of root.
		//                If x were to be unlinked from root (setting root to
		//                  0 if root == x), tree_invariant(root) == true.
		// Postcondition: tree_invariant(end_node->left) == true.  end_node->left
		//                may be different than the value passed in as root.
		template <class NodePtr>
		__device__ void
			tree_balance_after_insert(NodePtr root, NodePtr x)
		{
			x->is_black = x == root;
			while (x != root && !x->parent->is_black)
			{
				// x->parent != root because x->parent->is_black == false
				if (tree_is_left_child(x->parent))
				{
					NodePtr y = x->parent->parent->right;
					if (y != 0 && !y->is_black)
					{
						x = x->parent;
						x->is_black = true;
						x = x->parent;
						x->is_black = x == root;
						y->is_black = true;
					}
					else
					{
						if (!tree_is_left_child(x))
						{
							x = x->parent;
							tree_left_rotate(x);
						}
						x = x->parent;
						x->is_black = true;
						x = x->parent;
						x->is_black = false;
						tree_right_rotate(x);
						break;
					}
				}
				else
				{
					NodePtr y = x->parent->parent->left;
					if (y != 0 && !y->is_black)
					{
						x = x->parent;
						x->is_black = true;
						x = x->parent;
						x->is_black = x == root;
						y->is_black = true;
					}
					else
					{
						if (tree_is_left_child(x))
						{
							x = x->parent;
							tree_right_rotate(x);
						}
						x = x->parent;
						x->is_black = true;
						x = x->parent;
						x->is_black = false;
						tree_left_rotate(x);
						break;
					}
				}
			}
		}

		// Precondition:  root != 0 && z != 0.
		//                tree_invariant(root) == true.
		//                z == root or == a direct or indirect child of root.
		// Effects:  unlinks z from the tree rooted at root, rebalancing as needed.
		// Postcondition: tree_invariant(end_node->left) == true && end_node->left
		//                nor any of its children refer to z.  end_node->left
		//                may be different than the value passed in as root.
		template <class NodePtr>
		__device__ void
			tree_remove(NodePtr root, NodePtr z)
		{
			// z will be removed from the tree.  Client still needs to destruct/deallocate it
			// y is either z, or if z has two children, tree_next(z).
			// y will have at most one child.
			// y will be the initial hole in the tree (make the hole at a leaf)
			NodePtr y = (z->left == 0 || z->right == 0) ?
				z : tree_next(z);
			// x is y's possibly null single child
			NodePtr x = y->left != 0 ? y->left : y->right;
			// w is x's possibly null uncle (will become x's sibling)
			NodePtr w = 0;
			// link x to y's parent, and find wr
			if (x != 0)
				x->parent = y->parent;
			if (tree_is_left_child(y))
			{
				y->parent->left = x;
				if (y != root)
					w = y->parent->right;
				else
					root = x;  // w == 0
			}
			else
			{
				y->parent->right = x;
				// y can't be root if it is a right child
				w = y->parent->left;
			}
			bool removed_black = y->is_black;
			// If we didn't remove z, do so now by splicing in y for z,
			//    but copy z's color.  This does not impact x or w.
			if (y != z)
			{
				// z->left != nulptr but z->right might == x == 0
				y->parent = z->parent;
				if (tree_is_left_child(z))
					y->parent->left = y;
				else
					y->parent->right = y;
				y->left = z->left;
				y->left->parent = y;
				y->right = z->right;
				if (y->right != 0)
					y->right->parent = y;
				y->is_black = z->is_black;
				if (root == z)
					root = y;
			}
			// There is no need to rebalance if we removed a red, or if we removed
			//     the last node.
			if (removed_black && root != 0)
			{
				// Rebalance:
				// x has an implicit black color (transferred from the removed y)
				//    associated with it, no matter what its color is.
				// If x is root (in which case it can't be null), it is supposed
				//    to be black anyway, and if it is doubly black, then the double
				//    can just be ignored.
				// If x is red (in which case it can't be null), then it can absorb
				//    the implicit black just by setting its color to black.
				// Since y was black and only had one child (which x points to), x
				//   is either red with no children, else null, otherwise y would have
				//   different black heights under left and right pointers.
				// if (x == root || x != 0 && !x->is_black)
				if (x != 0)
					x->is_black = true;
				else
				{
					//  Else x isn't root, and is "doubly black", even though it may
					//     be null.  w can not be null here, else the parent would
					//     see a black height >= 2 on the x side and a black height
					//     of 1 on the w side (w must be a non-null black or a red
					//     with a non-null black child).
					while (true)
					{
						if (!tree_is_left_child(w))  // if x is left child
						{
							if (!w->is_black)
							{
								w->is_black = true;
								w->parent->is_black = false;
								tree_left_rotate(w->parent);
								// x is still valid
								// reset root only if necessary
								if (root == w->left)
									root = w;
								// reset sibling, and it still can't be null
								w = w->left->right;
							}
							// w->is_black is now true, w may have null children
							if ((w->left == 0 || w->left->is_black) &&
								(w->right == 0 || w->right->is_black))
							{
								w->is_black = false;
								x = w->parent;
								// x can no longer be null
								if (x == root || !x->is_black)
								{
									x->is_black = true;
									break;
								}
								// reset sibling, and it still can't be null
								w = tree_is_left_child(x) ?
									x->parent->right :
									x->parent->left;
								// continue;
							}
							else  // w has a red child
							{
								if (w->right == 0 || w->right->is_black)
								{
									// w left child is non-null and red
									w->left->is_black = true;
									w->is_black = false;
									tree_right_rotate(w);
									// w is known not to be root, so root hasn't changed
									// reset sibling, and it still can't be null
									w = w->parent;
								}
								// w has a right red child, left child may be null
								w->is_black = w->parent->is_black;
								w->parent->is_black = true;
								w->right->is_black = true;
								tree_left_rotate(w->parent);
								break;
							}
						}
						else
						{
							if (!w->is_black)
							{
								w->is_black = true;
								w->parent->is_black = false;
								tree_right_rotate(w->parent);
								// x is still valid
								// reset root only if necessary
								if (root == w->right)
									root = w;
								// reset sibling, and it still can't be null
								w = w->right->left;
							}
							// w->is_black is now true, w may have null children
							if ((w->left == 0 || w->left->is_black) &&
								(w->right == 0 || w->right->is_black))
							{
								w->is_black = false;
								x = w->parent;
								// x can no longer be null
								if (!x->is_black || x == root)
								{
									x->is_black = true;
									break;
								}
								// reset sibling, and it still can't be null
								w = tree_is_left_child(x) ?
									x->parent->right :
									x->parent->left;
								// continue;
							}
							else  // w has a red child
							{
								if (w->left == 0 || w->left->is_black)
								{
									// w right child is non-null and red
									w->right->is_black = true;
									w->is_black = false;
									tree_left_rotate(w);
									// w is known not to be root, so root hasn't changed
									// reset sibling, and it still can't be null
									w = w->parent;
								}
								// w has a left red child, right child may be null
								w->is_black = w->parent->is_black;
								w->parent->is_black = true;
								w->left->is_black = true;
								tree_right_rotate(w->parent);
								break;
							}
						}
					}
				}
			}
		}

		template <class _Allocator> class map_node_destructor;

		// node

		//template <class _Pointer>
		//class tree_end_node
		//{
		//public:
		//	typedef _Pointer pointer;
		//	pointer left;

		//	__device__ tree_end_node() : left() {}
		//};

		template <class Tp, class VoidPtr>
		class tree_node
			//_base
			//: public tree_end_node
			//<
			//typename std::pointer_traits<VoidPtr>::template
			//rebind<tree_node_base<VoidPtr> >::other
			//>
		{
			__device__ tree_node(const tree_node&) = default;
			__device__ tree_node& operator=(const tree_node&) = default;
			
			__device__ tree_node(tree_node&&) = default;
			__device__ tree_node& operator=(tree_node&&) = default;
		public:
			typedef typename std::pointer_traits<VoidPtr>::template
				rebind<tree_node>
				pointer;
			typedef typename std::pointer_traits<VoidPtr>::template
				rebind<const tree_node>
				const_pointer;
			typedef Tp value_type;

			pointer right;
			pointer left;
			pointer parent;
			bool is_black;
			value_type value;

			template <class ... Args, class = std::enable_if<std::is_constructible<Tp, Args&&...>::value>>
			__device__ tree_node(Args&& ... args) :right(), left(), parent(), value(std::forward<Args>(args)...), is_black(false) {}

			/*__device__ explicit tree_node(const value_type& v)
				: right(), left(), parent(), value(v), is_black(false) {}*/
		};

		//template <class Tp, class VoidPtr>
		//class tree_node
		//	: public tree_node_base<VoidPtr>
		//{
		//public:
		//	typedef tree_node_base<VoidPtr> base;
		//	typedef Tp value_type;

		//	value_type value;

		//	__device__ explicit tree_node(const value_type& v)
		//		: value(v) {}
		//};

		template <class _TreeIterator> class map_iterator;
		template <class _TreeIterator> class map_const_iterator;

		template <class Tp, class NodePtr, class DiffType>
		class tree_iterator
		{
			typedef NodePtr node_pointer;
			typedef typename std::pointer_traits<node_pointer>::element_type node;
			//typedef typename std::pointer_traits<node_pointer>::pointer_type node_pointer;

			node_pointer ptr;

			//typedef std::pointer_traits<node_pointer> __std::pointer_traits;
		public:
			typedef std::bidirectional_iterator_tag iterator_category;
			typedef Tp value_type;
			typedef DiffType difference_type;
			typedef value_type& reference;
			typedef typename std::pointer_traits<node_pointer>::template
				rebind<value_type> pointer;

			__device__ tree_iterator() {}

			__device__ reference operator*() const { return ptr->value; }
			__device__ pointer operator->() const { return &ptr->value; }

			__device__ tree_iterator& operator++()
			{
				ptr = tree_next(ptr);
				return *this;
			}
			__device__ tree_iterator operator++(int)
			{
				tree_iterator t(*this); ++(*this); return t;
			}

			__device__ tree_iterator& operator--()
			{
				ptr = tree_prev(ptr);
				return *this;
			}
			__device__ tree_iterator operator--(int)
			{
				tree_iterator t(*this); --(*this); return t;
			}

			friend
				__device__ bool operator==(const tree_iterator& x, const tree_iterator& y)
			{
				return x.ptr == y.ptr;
			}
			friend __device__ bool operator!=(const tree_iterator& x, const tree_iterator& y)
			{
				return !(x == y);
			}

		private:
			__device__ explicit tree_iterator(node_pointer p) : ptr( p) {}
			template <class, class> friend class tree;
			template <class, class, class> friend class tree_const_iterator;
			template <class, class, class> friend class cuda_map;
			template <class> friend class cuda_map_iterator;
			template <class, class> friend class cuda_red_black_tree;
			template <class, class> friend class cuda_set;
			template <class> friend class cuda_set_iterator;
		};

		template <class Tp, class _ConstNodePtr, class DiffType>
		class tree_const_iterator
		{
			typedef _ConstNodePtr node_pointer;
			typedef typename std::pointer_traits<node_pointer>::element_type node;
			typedef typename std::pointer_traits<node_pointer>::pointer const_node_pointer;

			const_node_pointer ptr;

			typedef std::pointer_traits<node_pointer> pointer_traits;
		public:
			typedef std::bidirectional_iterator_tag iterator_category;
			typedef Tp value_type;
			typedef DiffType difference_type;
			typedef const value_type& reference;
			typedef typename std::pointer_traits<node_pointer>::template
				rebind<const value_type> pointer;

			__device__ tree_const_iterator() {}
		private:
			typedef typename std::remove_const<node>::type non_const_node;
			typedef typename std::pointer_traits<node_pointer>::template
				rebind<non_const_node> non_const_node_pointer;
			typedef tree_iterator<value_type, non_const_node_pointer, difference_type>
				non_const_iterator;
		public:
			__device__ tree_const_iterator(non_const_iterator p)
				: ptr(p.ptr) {}

			__device__ reference operator*() const { return ptr->value; }
			__device__ pointer operator->() const { return &ptr->value; }

			__device__ tree_const_iterator& operator++()
			{
				ptr = tree_next(ptr);
				return *this;
			}
			__device__ tree_const_iterator operator++(int)
			{
				tree_const_iterator t(*this); ++(*this); return t;
			}

			__device__ tree_const_iterator& operator--()
			{
				ptr = tree_prev(ptr);
				return *this;
			}
			__device__ tree_const_iterator operator--(int)
			{
				tree_const_iterator t(*this); --(*this); return t;
			}

			friend __device__ bool operator==(const tree_const_iterator& x, const tree_const_iterator& y)
			{
				return x.ptr == y.ptr;
			}
			friend __device__ bool operator!=(const tree_const_iterator& x, const tree_const_iterator& y)
			{
				return !(x == y);
			}

		private:
			__device__ explicit tree_const_iterator(const_node_pointer p)
				: ptr(p) {}
			template <class, class> friend class cuda_red_black_tree;
			template <class, class, class> friend class cuda_map;
			template <class, class> friend class cuda_set;
			template <class> friend class cuda_map_const_iterator;
			template <class> friend class cuda_set_const_iterator;
		};

		template <class Tp, class Compare>
		class cuda_red_black_tree
		{
		public:
			typedef Tp                                     value_type;
			typedef Tp*                                    pointer;
			typedef const Tp*                              const_pointer;
			typedef Tp&                                    reference;
			typedef const Tp&                              const_reference;
			typedef std::size_t								size_type;
			typedef std::ptrdiff_t							difference_type;

			typedef tree_node<value_type, void*> node;
			typedef typename node::pointer node_pointer;
			typedef typename node::const_pointer node_const_pointer;
			typedef Compare value_compare;
		/*	typedef tree_node_base<typename __alloc_traits::void_pointer> __node_base;
			typedef typename __alloc_traits::template
				rebind_alloc<__node>::other

				__node_allocator;
			typedef allocator_traits<__node_allocator>       __node_traits;
			typedef typename __node_traits::pointer          node_pointer;
			typedef typename __node_traits::const_pointer    node_const_pointer;
			typedef typename __node_base::pointer            __node_base_pointer;
			typedef typename __node_base::const_pointer      __node_base_const_pointer;*/
		private:
			//typedef typename __node_base::base end_node_t;

			typedef typename std::pointer_traits<node_pointer>::template
				rebind<node>
				node_ptr;
			typedef typename std::pointer_traits<node_pointer>::template
				rebind<const node>
				node_const_ptr;

			node_pointer m_begin_node;
			node m_end_node;
			size_type count;
			Compare val_compare;
			//pair<node, void*>  pair1_;
			//pair<size_type, value_compare>        pair3_;

		public:
			__device__ node_pointer end_node()
			{
				return &m_end_node;
			}
			__device__ node_const_pointer end_node() const
			{
				return &m_end_node;
			}

		private:
			__device__ node_const_pointer begin_node() const { return m_begin_node; }
			__device__ node_pointer& begin_node() { return m_begin_node; }
		private:
			__device__ size_type& size() { return count; }
		public:
			__device__ const size_type& size() const { return count; }
			__device__ value_compare& value_comp() {return val_compare;}
			__device__ const value_compare& value_comp() const
        	{return val_compare;}
		public:
			__device__ node_pointer root()
			{
				return end_node()->left;
			}
			__device__ node_const_pointer root() const
			{
				return end_node()->left;
			}

			typedef tree_iterator<value_type, node_pointer, difference_type> iterator;
			typedef tree_const_iterator<value_type, node_const_pointer, difference_type> const_iterator;

			__device__ inline cuda_red_black_tree(const cuda_red_black_tree& t) :cuda_red_black_tree(t.val_compare)
			{
				*this = t;
			}
			__device__ cuda_red_black_tree(const Compare& comp);

			__device__ cuda_red_black_tree& operator=(const cuda_red_black_tree& t);
			template <class InputIterator>
			__device__ void assign_unique(InputIterator first, InputIterator last);
			template <class InputIterator>
			__device__ void assign_multi(InputIterator first, InputIterator last);

			__device__ ~cuda_red_black_tree();

			__device__ iterator begin() { return iterator(begin_node()); }
			__device__ const_iterator begin() const { return const_iterator(begin_node()); }
			__device__ iterator end() { return iterator(end_node()); }
			__device__ const_iterator end() const { return const_iterator(end_node()); }

			__device__ size_type max_size() const
			{
				return cuda_numeric_limits<std::size_t>::max();
			}

			__device__ void clear();

			__device__ void swap(cuda_red_black_tree&  t);

			__device__ cu::cuda_pair<iterator, bool> insert_unique(const value_type& v);
			__device__ cu::cuda_pair<iterator, bool> insert_unique(value_type&& v);
			__device__ iterator insert_unique(const_iterator p, const value_type& v);
			__device__ iterator insert_multi(const value_type& v);
			__device__ iterator insert_multi(const_iterator p, const value_type& v);

			__device__ cu::cuda_pair<iterator, bool> node_insert_unique( node_pointer nd);
			__device__ iterator node_insert_unique(const_iterator p, node_pointer nd);

			__device__ iterator node_insert_multi( node_pointer nd);
			__device__ iterator node_insert_multi(const_iterator p, node_pointer nd);

			__device__ iterator erase(const_iterator p);
			__device__ iterator erase(const_iterator f, const_iterator l);
			template <class Key>
			__device__ size_type erase_unique(const Key& k);
			template <class Key>
			__device__ size_type erase_multi(const Key& k);

			__device__ void insert_node_at(node_pointer parent, node_pointer& child,
							node_pointer new_node);

			template <class Key>
			__device__ iterator find(const Key& v);
			template <class Key>
			__device__ const_iterator find(const Key& v) const;

			template <class Key>
			__device__ size_type count_unique(const Key& k) const;
			template <class Key>
			__device__ size_type count_multi(const Key& k) const;

			template <class Key>
			__device__ iterator lower_bound(const Key& v)
			{
				return lower_bound(v, root(), end_node());
			}
			template <class Key>
			__device__ iterator lower_bound(const Key& v,
						node_pointer root, node_pointer result);
			template <class Key>
			__device__ const_iterator lower_bound(const Key& v) const
			{
				return lower_bound(v, root(), end_node());
			}
			template <class Key>
			__device__ const_iterator lower_bound(const Key& v,
						node_const_pointer root,
						node_const_pointer result) const;
			template <class Key>
			__device__ iterator upper_bound(const Key& v)
			{
				return upper_bound(v, root(), end_node());
			}
			template <class Key>
			__device__ iterator upper_bound(const Key& v,
				node_pointer root,
				node_pointer result);
			template <class Key>
			__device__ const_iterator upper_bound(const Key& v) const
			{
				return upper_bound(v, root(), end_node());
			}
			template <class Key>
			__device__ const_iterator upper_bound(const Key& v,
				node_const_pointer root,
				node_const_pointer result) const;
			template <class Key>
			__device__ cu::cuda_pair<iterator, iterator>
				equal_range_unique(const Key& k);
			template <class Key>
			__device__ cu::cuda_pair<const_iterator, const_iterator>
				equal_range_unique(const Key& k) const;

			template <class Key>
			__device__ cu::cuda_pair<iterator, iterator>
				equal_range_multi(const Key& k);
			template <class Key>
			__device__ cu::cuda_pair<const_iterator, const_iterator>
				equal_range_multi(const Key& k) const;

			typedef cuda_device_unique_ptr<node> node_holder;

			__device__ node_holder remove(const_iterator p);
		private:
			__device__ typename node::pointer& find_leaf_low(typename node::pointer& parent, const value_type& v);
			__device__ typename node::pointer& find_leaf_high(typename node::pointer& parent, const value_type& v);
			__device__ typename node::pointer& find_leaf(const_iterator hint, typename node::pointer& parent, const value_type& v);

			template <class Key>
			__device__ typename node::pointer& find_equal(typename tree_node<Tp, void*>::pointer& parent, const Key& v);
			template <class Key>
			__device__ typename node::pointer& find_equal(const_iterator hint, typename node::pointer& parent,const Key& v);

			__device__ node_holder construct_node(const value_type& v);

			template <class ... Args, class = std::enable_if_t<std::is_constructible<node, Args&&...>::value>>
			__device__ node_holder construct_node(Args&& ... args);

			__device__ void destroy(node_pointer nd);

		/*	__device__ void copy_assign_alloc(const cuda_red_black_tree& t)
			{
				copy_assign_alloc(t, integral_constant<bool,
					node_traits::propagate_on_container_copy_assignment::value>());
			}

			__device__ void copy_assign_alloc(const cuda_red_black_tree& t, true_type)
			{
				node_alloc() = t.node_alloc();
			}
			void __copy_assign_alloc(const cuda_red_black_tree& t, false_type) {}

			__device__ void move_assign(cuda_red_black_tree& t, false_type);
			__device__ void move_assign(cuda_red_black_tree& t, true_type);

			__device__ void move_assign_alloc(cuda_red_black_tree& t)
			{
				move_assign_alloc(t, integral_constant<bool,
					node_traits::propagate_on_container_move_assignment::value>());
			}

			__device__ void __move_assign_alloc(cuda_red_black_tree& t, true_type)
			{
				__node_alloc() = util::move(t.__node_alloc());
			}
			__device__ void __move_assign_alloc(cuda_red_black_tree& t, false_type) {}*/
			__device__ node_pointer detach();
			__device__ static node_pointer detach(node_pointer nd);
		};

		template <class _Tp, class _Compare>
		__device__ cuda_red_black_tree<_Tp, _Compare>::cuda_red_black_tree(const _Compare& comp)
			: m_begin_node(nullptr), m_end_node(value_type()), count(0), val_compare(comp)
		{
			 begin_node() = end_node();
		}

		//template <class Tp, class _Compare>
		template <class Tp, class _Compare>
		template <class ... Args, class>
		__device__ typename cuda_red_black_tree<Tp, _Compare>::node_holder cuda_red_black_tree<Tp, _Compare>::construct_node(Args&& ... args)
		{
			return node_holder(new node(std::forward<Args>(args)...));
		}
		// Precondition:  size() != 0
		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer cuda_red_black_tree<Tp, Compare>::detach()
		{
			node_pointer cache = begin_node();
			begin_node() = end_node();
			m_end_node.left->parent = 0;
			m_end_node.left = 0;
			count = 0;
			// cache->left == 0
			if (cache->right != 0)
				cache = cache->right;
			// cache->left == 0
			// cache->right == 0
			return cache;
		}

		// Precondition:  cache != 0
		//    cache->left_ == 0
		//    cache->right_ == 0
		//    This is no longer a red-black tree
		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer cuda_red_black_tree<Tp, Compare>::detach(node_pointer cache)
		{
			if (cache->parent == 0)
				return 0;
			if (tree_is_left_child(cache))
			{
				cache->parent->left = 0;
				cache = cache->parent;
				if (cache->right == 0)
					return cache;
				return tree_leaf(cache->right);
			}
			// cache is right child
			cache->parent->right = 0;
			cache = cache->parent;
			if (cache->left == 0)
				return cache;
			return tree_leaf(cache->left);
		}

		template <class Tp, class Compare>
		__device__ cuda_red_black_tree<Tp, Compare>& cuda_red_black_tree<Tp, Compare>::operator=(const cuda_red_black_tree& t)
		{
			if (this != &t)
			{
				//value_comp() = t.value_comp();
				/*__copy_assign_alloc(t);*/
				assign_multi(t.begin(), t.end());
			}
			return *this;
		}

		template <class Tp, class Compare>
		template <class InputIterator>
		__device__ void cuda_red_black_tree<Tp, Compare>::assign_unique(InputIterator first, InputIterator last)
		{
			if (size() != 0)
			{
				node_pointer cache = detach();
				for (; cache != 0 && first != last; ++first)
				{
					cache->value = *first;
					node_pointer next = detach(cache);
					node_insert_unique(cache);
					cache = next;
				}
				if (cache != 0)
				{
					while (cache->parent != 0)
						cache = cache->parent;
					destroy(cache);
				}
			}
			for (; first != last; ++first)
				insert_unique(*first);
		}

		template <class Tp, class Compare>
		template <class InputIterator>
		__device__ void cuda_red_black_tree<Tp, Compare>::assign_multi(InputIterator first, InputIterator last)
		{
			if (size() != 0)
			{
				node_pointer cache = detach();
				for (; cache != 0 && first != last; ++first)
				{
					cache->value = *first;
					node_pointer next = detach(cache);
					node_insert_multi(cache);
					cache = next;
				}
				if (cache != 0)
				{
					while (cache->parent != 0)
						cache = cache->parent;
					destroy(cache);
				}
			}
			for (; first != last; ++first)
				insert_multi(*first);
		}

		//template <class Tp, class Compare>
		//__device__ cuda_red_black_tree<Tp, Compare>::cuda_red_black_tree(const cuda_red_black_tree& t)
		//	: begin_node(t.begin_node()),
		//	//pair1_(__node_traits::select_on_container_copy_construction(t.__node_alloc())),
		//	count(0)
		//{
		//	begin_node() = end_node();
		//}

		template <class Tp, class Compare>
		__device__ cuda_red_black_tree<Tp, Compare>::~cuda_red_black_tree()
		{
			destroy(root());
		}

		template <class Tp, class Compare>
		__device__ void cuda_red_black_tree<Tp, Compare>::destroy(node_pointer nd)
		{
			if (nd != 0)
			{
				destroy(nd->left);
				destroy(nd->right);
				
				/*__node_traits::destroy(na, &nd->value);
				__node_traits::deallocate(na, nd, 1);*/
			}
		}

		template <class Tp, class Compare>
		__device__ void cuda_red_black_tree<Tp, Compare>::swap(cuda_red_black_tree& t)
		{
			using thrust::swap;
			swap(m_begin_node, t.m_begin_node);
			swap(m_end_node, t.m_end_node);
			swap(count, t.count);
			/*__swap_alloc(__node_alloc(), t.__node_alloc());*/
			//pair3_.swap(t.pair3_);
			if (size() == 0)
				begin_node() = end_node();
			else
				end_node()->left->parent = end_node();
			if (t.size() == 0)
				begin_node() = end_node();
			else
				t.end_node()->left->parent = t.end_node();
		}

		template <class Tp, class Compare>
		__device__ void cuda_red_black_tree<Tp, Compare>::clear()
		{
			destroy(root());
			count = 0;
			begin_node() = end_node();
			end_node()->left = nullptr;
		}

		// Find lower_bound place to insert
		// Set parent to parent of null leaf
		// Return reference to null leaf
		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer&
			cuda_red_black_tree<Tp, Compare>::find_leaf_low(typename tree_node<Tp, void*>::pointer& parent,
				const value_type& v)
		{
			node_pointer nd = root();
			if (nd != 0)
			{
				while (true)
				{
					if (this->value_comp()(nd->value, v))
					{
						if (nd->right != 0)
							nd = nd->right;
						else
						{
							parent = nd;
							return parent->right;
						}
					}
					else
					{
						if (nd->left != 0)
							nd = nd->left;
						else
						{
							parent = nd;
							return parent->left;
						}
					}
				}
			}
			parent = end_node;
			return parent->left;
		}

		// Find upper_bound place to insert
		// Set parent to parent of null leaf
		// Return reference to null leaf
		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer&
			cuda_red_black_tree<Tp, Compare>::find_leaf_high(typename tree_node<Tp, void*>::pointer& parent,
				const value_type& v)
		{
			node_pointer nd = root();
			if (nd != 0)
			{
				while (true)
				{
					if (this->value_comp()(v, nd->value))
					{
						if (nd->left != 0)
							nd = nd->left;
						else
						{
							parent = nd;
							return parent->left;
						}
					}
					else
					{
						if (nd->right != 0)
							nd = nd->right;
						else
						{
							parent = nd;
							return parent->right;
						}
					}
				}
			}
			parent = end_node();
			return parent->left;
		}

		// Find leaf place to insert closest to hint
		// First check prior to hint.
		// Next check after hint.
		// Next do O(log N) search.
		// Set parent to parent of null leaf
		// Return reference to null leaf
		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer&
			cuda_red_black_tree<Tp, Compare>::find_leaf(const_iterator hint, typename tree_node<Tp, void*>::pointer& parent, const value_type& v)
		{
			if (hint == end() || !this->value_comp()(*hint, v))  // check before
			{
				// v <= *hint
				const_iterator prior = hint;
				if (prior == begin() || v >= *--prior)
				{
					// *prev(hint) <= v <= *hint
					if (hint.ptr->left == 0)
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
				// v < *prev(hint)
				return find_leaf_high(parent, v);
			}
			// else v > *hint
			return find_leaf_low(parent, v);
		}

		// Find place to insert if v doesn't exist
		// Set parent to parent of null leaf
		// Return reference to null leaf
		// If v exists, set parent to node of v and return reference to node of v
		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer&
			cuda_red_black_tree<Tp, Compare>::find_equal(typename tree_node<Tp, void*>::pointer& parent, const Key& v)
		{
			node_pointer nd = root();
			if (nd != 0)
			{
				while (true)
				{
					if (this->value_comp()(v, nd->value))
					{
						if (nd->left != 0)
							nd = nd->left;
						else
						{
							parent = nd;
							return parent->left;
						}
					}
					else if (this->value_comp()(nd->value, v))
					{
						if (nd->right != 0)
							nd = nd->right;
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
			parent = end_node();
			return parent->left;
		}

		// Find place to insert if v doesn't exist
		// First check prior to hint.
		// Next check after hint.
		// Next do O(log N) search.
		// Set parent to parent of null leaf
		// Return reference to null leaf
		// If v exists, set parent to node of v and return reference to node of v
		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_pointer&
			cuda_red_black_tree<Tp, Compare>::find_equal(const_iterator hint, typename tree_node<Tp, void*>::pointer& parent, const Key& v)
		{
			if (hint == end() || this->value_comp()(v, *hint)) // check before
			{
				// v < *hint
				const_iterator prior = hint;
				if (prior == begin() || this->value_comp()(*--prior, v))
				{
					// *prev(hint) < v < *hint
					if (hint.ptr->left == 0)
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
				// v <= *prev(hint)
				return find_equal(parent, v);
			}
			else if (this->value_comp()(*hint, v))  // check after
			{
				// *hint < v
				const_iterator next = next_it(hint);
				if (next == end() || this->value_comp()(v, *next))
				{
					// *hint < v < *util::next(hint)
					if (hint.ptr->right == 0)
					{
						parent = const_cast<node_pointer&>(hint.ptr);
						return parent->right;
					}
					else
					{
						parent = const_cast<node_pointer&>(next.ptr);
						return parent->left;
					}
				}
				// *next(hint) <= v
				return find_equal(parent, v);
			}
			// else v == *hint
			parent = const_cast<node_pointer&>(hint.ptr);
			return parent;
		}

		template <class Tp, class Compare>
		__device__ void cuda_red_black_tree<Tp, Compare>::insert_node_at(typename tree_node<Tp, void*>::pointer parent,
			typename tree_node<Tp, void*>::pointer& child,
			typename tree_node<Tp, void*>::pointer new_node)
		{
			new_node->left = 0;
			new_node->right = 0;
			new_node->parent = parent;
			child = new_node;
			if (m_begin_node->left != 0)
				m_begin_node = m_begin_node->left;
			tree_balance_after_insert(m_end_node.left, child);
			++size();
		}

		template <class Tp, class Compare>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::iterator, bool>
			cuda_red_black_tree<Tp, Compare>::insert_unique(const value_type& v)
		{
			node_pointer parent;
			node_pointer& child = find_equal(parent, v);
			node_pointer r = child;
			bool inserted = false;
			if (child == 0)
			{
				node_holder h = construct_node(v);
				insert_node_at(parent, child, h.get());
				r = h.release();
				inserted = true;
			}
			return cu::cuda_pair<iterator, bool>(iterator(r), inserted);
		}


		template <class Tp, class Compare>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::iterator, bool>
			cuda_red_black_tree<Tp, Compare>::insert_unique(value_type&& v)
		{
			node_pointer parent;
			node_pointer& child = find_equal(parent, std::move(v));
			node_pointer r = child;
			bool inserted = false;
			if (child == 0)
			{
				node_holder h = construct_node(std::move(v));
				insert_node_at(parent, child, h.get());
				r = h.release();
				inserted = true;
			}
			return cu::cuda_pair<iterator, bool>(iterator(r), inserted);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::insert_unique(const_iterator p, const value_type& v)
		{
			node_pointer parent;
			node_pointer& child = find_equal(p, parent, v);
			node_pointer r = child;
			if (child == 0)
			{
				node_holder h = construct_node(v);
				insert_node_at(parent, child, h.get());
				r = h.release();
			}
			return iterator(r);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator cuda_red_black_tree<Tp, Compare>::insert_multi(const value_type& v)
		{
			node_pointer parent;
			node_pointer& child = find_leaf_high(parent, v);
			node_holder h = construct_node(v);
			insert_node_at(parent, child, h.get());
			return iterator(h.release());
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::insert_multi(const_iterator p, const value_type& v)
		{
			node_pointer parent;
			node_pointer& child = find_leaf(p, parent, v);
			node_holder h = construct_node(v);
			insert_node_at(parent, child, h.get());
			return iterator(h.release());
		}

		template <class Tp, class Compare>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::iterator, bool>
			cuda_red_black_tree<Tp, Compare>::node_insert_unique(node_pointer nd)
		{
			node_pointer parent;
			node_pointer& child = find_equal(parent, nd->value);
			node_pointer r = child;
			bool inserted = false;
			if (child == 0)
			{
				insert_node_at(parent, child, nd);
				r = nd;
				inserted = true;
			}
			return cu::cuda_pair<iterator, bool>(iterator(r), inserted);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::node_insert_unique(const_iterator p, node_pointer nd)
		{
			node_pointer parent;
			node_pointer& child = find_equal(p, parent, nd->value);
			node_pointer r = child;
			if (child == 0)
			{
				insert_node_at(parent, child, nd);
				r = nd;
			}
			return iterator(r);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::node_insert_multi(node_pointer nd)
		{
			node_pointer parent;
			node_pointer& child = find_leaf_high(parent, nd->value);
			insert_node_at(parent, child, nd);
			return iterator(nd);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::node_insert_multi(const_iterator p, node_pointer nd)
		{
			node_pointer parent;
			node_pointer& child = find_leaf(p, parent, nd->value);
			insert_node_at(parent, child, nd);
			return iterator(nd);
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::erase(const_iterator p)
		{
			node_pointer np = const_cast<node_pointer>(p.ptr);
			iterator r(np);
			++r;
			if (begin_node() == np)
				begin_node() = r.ptr;
			--size();
			//__node_traits::destroy(na, const_cast<value_type*>(util::addressof(*p)));
			tree_remove(end_node()->left, np);
			free(np);
			//__node_traits::deallocate(na, np, 1);
			return r;
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::erase(const_iterator f, const_iterator l)
		{
			while (f != l)
				f = erase(f);
			return iterator(const_cast<node_pointer>(l.ptr));
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::size_type
			cuda_red_black_tree<Tp, Compare>::erase_unique(const Key& k)
		{
			iterator i = find(k);
			if (i == end())
				return 0;
			erase(i);
			return 1;
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::size_type
			cuda_red_black_tree<Tp, Compare>::erase_multi(const Key& k)
		{
			cu::cuda_pair<iterator, iterator> p = equal_range_multi(k);
			size_type r = 0;
			for (; p.first != p.second; ++r)
				p.first = erase(p.first);
			return r;
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::find(const Key& v)
		{
			iterator p = lower_bound(v, root(), end_node());
			if (p != end() && !this->value_comp()(v, *p))
				return p;
			return end();
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::const_iterator
			cuda_red_black_tree<Tp, Compare>::find(const Key& v) const
		{
			const_iterator p = lower_bound(v, root(), end_node());
			if (p != end() && !this->value_comp()(v, *p))
				return p;
			return end();
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::size_type
			cuda_red_black_tree<Tp, Compare>::count_unique(const Key& k) const
		{
			node_const_pointer result = end_node();
			node_const_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = static_cast<node_const_pointer>(rt->left);
				}
				else if (this->value_comp()(rt->value, k))
					rt = static_cast<node_const_pointer>(rt->right);
				else
					return 1;
			}
			return 0;
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::size_type
			cuda_red_black_tree<Tp, Compare>::count_multi(const Key& k) const
		{
			typedef cu::cuda_pair<const_iterator, const_iterator> Pp;
			node_const_pointer result = end_node();
			node_const_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = rt->left;
				}
				else if (this->value_comp()(rt->value, k))
					rt = rt->right;
				else
					return thrust::distance(
						lower_bound(k, rt->left, rt),
						upper_bound(k, rt->right, result)
					);
			}
			return 0;
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::lower_bound(const Key& v, node_pointer root, node_pointer result)
		{
			while (root != 0)
			{
				if (!this->value_comp()(root->value, v))
				{
					result = root;
					root = root->left;
				}
				else
					root = root->right;
			}
			return iterator(result);
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::const_iterator
			cuda_red_black_tree<Tp, Compare>::lower_bound(const Key& v, node_const_pointer root,
				node_const_pointer result) const
		{
			while (root != 0)
			{
				if (!this->value_comp()(root->value, v))
				{
					result = root;
					root = static_cast<node_const_pointer>(root->left);
				}
				else
					root = static_cast<node_const_pointer>(root->right);
			}
			return const_iterator(result);
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::iterator
			cuda_red_black_tree<Tp, Compare>::upper_bound(const Key& v, node_pointer root,
				node_pointer result)
		{
			while (root != 0)
			{
				if (this->value_comp()(v, root->value))
				{
					result = root;
					root = root->left;
				}
				else
					root = root->right;
			}
			return iterator(result);
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ typename cuda_red_black_tree<Tp, Compare>::const_iterator
			cuda_red_black_tree<Tp, Compare>::upper_bound(const Key& v, node_const_pointer root,
				node_const_pointer result) const
		{
			while (root != 0)
			{
				if (this->value_comp()(v, root->value))
				{
					result = root;
					root = static_cast<node_const_pointer>(root->left);
				}
				else
					root = static_cast<node_const_pointer>(root->right);
			}
			return const_iterator(result);
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::iterator,
			typename cuda_red_black_tree<Tp, Compare>::iterator> cuda_red_black_tree<Tp, Compare>::equal_range_unique(const Key& k)
		{
			typedef cu::cuda_pair<iterator, iterator> _Pp;
			node_pointer result = end_node();
			node_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = rt->left;
				}
				else if (this->value_comp()(rt->value, k))
					rt = rt->right;
				else
					return _Pp(iterator(rt),
						iterator(
							rt->right != 0 ?
							tree_min(rt->right)
							: result));
			}
			return _Pp(iterator(result), iterator(result));
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::const_iterator,
			typename cuda_red_black_tree<Tp, Compare>::const_iterator> cuda_red_black_tree<Tp, Compare>::equal_range_unique(const Key& k) const
		{
			typedef cu::cuda_pair<const_iterator, const_iterator> _Pp;
			node_const_pointer result = end_node();
			node_const_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = static_cast<node_const_pointer>(rt->left);
				}
				else if (this->value_comp()(rt->value, k))
					rt = static_cast<node_const_pointer>(rt->right);
				else
					return _Pp(const_iterator(rt),
						const_iterator(
							rt->right != 0 ?
							static_cast<node_const_pointer>(tree_min(rt->right))
							: result));
			}
			return _Pp(const_iterator(result), const_iterator(result));
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::iterator,
			typename cuda_red_black_tree<Tp, Compare>::iterator> cuda_red_black_tree<Tp, Compare>::equal_range_multi(const Key& k)
		{
			typedef cu::cuda_pair<iterator, iterator> _Pp;
			node_pointer result = end_node();
			node_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = rt->left;
				}
				else if (this->value_comp()(rt->value, k))
					rt = rt->right;
				else
					return _Pp(lower_bound(k, rt->left, rt),
						upper_bound(k, rt->right, result));
			}
			return _Pp(iterator(result), iterator(result));
		}

		template <class Tp, class Compare>
		template <class Key>
		__device__ cu::cuda_pair<typename cuda_red_black_tree<Tp, Compare>::const_iterator,
			typename cuda_red_black_tree<Tp, Compare>::const_iterator> cuda_red_black_tree<Tp, Compare>::equal_range_multi(const Key& k) const
		{
			typedef cu::cuda_pair<const_iterator, const_iterator> _Pp;
			node_const_pointer result = end_node();
			node_const_pointer rt = root();
			while (rt != 0)
			{
				if (this->value_comp()(k, rt->value))
				{
					result = rt;
					rt = static_cast<node_const_pointer>(rt->left);
				}
				else if (this->value_comp()(rt->value, k))
					rt = static_cast<node_const_pointer>(rt->right);
				else
					return _Pp(lower_bound(k, static_cast<node_const_pointer>(rt->left), rt),
						upper_bound(k, static_cast<node_const_pointer>(rt->right), result));
			}
			return _Pp(const_iterator(result), const_iterator(result));
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_holder
			cuda_red_black_tree<Tp, Compare>::construct_node(const value_type& v)
		{
			return node_holder(new node(v));
		}

		template <class Tp, class Compare>
		__device__ typename cuda_red_black_tree<Tp, Compare>::node_holder
			cuda_red_black_tree<Tp, Compare>::remove(const_iterator p)
		{
			node_pointer np = const_cast<node_pointer>(p.ptr);
			if (begin_node() == np)
			{
				if (np->right != 0)
					begin_node() = np->right;
				else
					begin_node() = np->parent;
			}
			--size();
			tree_remove(end_node()->left, np);
			return node_holder(np);
		}

		template <class Tp, class Compare>
		__device__ inline void swap(cuda_red_black_tree<Tp, Compare>& x, cuda_red_black_tree<Tp, Compare>& y)
		{
			x.swap(y);
		}

} //namespace cu

#endif //CUDAREDBLACKTREE_H
