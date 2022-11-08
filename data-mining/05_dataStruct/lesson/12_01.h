#include <iostream>
using namespace std;
 
/************************/
/*  12_01.h 文件
/************************/
 
class BSTNode
{
public:
	int key;  //关键字
	BSTNode *left;  //左子节点
	BSTNode *right; //右子节点
	BSTNode *parent;  //父节点
	 
	BSTNode(int k = 0, BSTNode *l = NULL, BSTNode *r = NULL, BSTNode *p = NULL) : key(k), left(l), right(r), parent(p) {};  //初始化列表
};
 
 
class BSTree
{
public:
	BSTree();  //构造函数
	~BSTree();  //析构函数
 
	void insert(int key);  //将key节点插入到二叉树中
 
	void PreOrder();  //前序二叉树遍历
	void InOrder();  //中序二叉树遍历
	void PostOrder();  //后序二叉树遍历
 
	BSTNode *search(int key);  //递归实现，在二叉树中查找key节点
	BSTNode *IteratorSearch(int key);  //迭代实现，在二叉树中查找key节点
 
	BSTNode *successor(BSTNode *x);  //找节点(x)的后继节点。即，查找"二叉树中数据值大于该节点"的"最小节点"
	BSTNode *predecessor(BSTNode *x);  //找节点(x)的前驱节点。即，查找"二叉树中数据值小于该节点"的"最大节点"
 
	void remove(int key);  //删除key节点
 
	void destroy();  //销毁二叉树
 
 
private:
	BSTNode *root;  //根节点
	void PreOrder(BSTNode *tree);  //前序二叉树遍历
	void InOrder(BSTNode *tree);  //中序二叉树遍历
	void PostOrder(BSTNode *tree);  //后序二叉树遍历
 
	BSTNode *search(BSTNode *x, int key);  //递归实现，在”二叉树x“中查找key节点
	BSTNode *IteratorSearch(BSTNode *x, int key);  //迭代实现，在“二叉树x”中查找key节点
 
	BSTNode *minimum(BSTNode *tree);  //查找最小节点：返回tree为根节点的二叉树的最小节点
	BSTNode *maximum(BSTNode *tree);  //查找最大节点：返回tree为根节点的二叉树的最大节点
 
	void insert(BSTNode *&tree, BSTNode *z);  // 将节点(z)插入到二叉树(tree)中
 
	BSTNode *remove(BSTNode *tree, BSTNode *z);  // 删除二叉树(tree)中的节点(z)，并返回被删除的节点
 
	void destroy(BSTNode *&tree);  //销毁二叉树
};