#include <iostream>
#include <stdlib.h>
#include "12_01.h"
using namespace std;
 
/***********************************************************/
/*  二叉查找树 —— 插入、遍历、查找、删除、销毁
/**********************************************************/
 
int main(void)
{
	/************************/
	/*          插入
	/************************/
	BSTree *tree = new BSTree();
	int array[6] = {11, 33, 18, 24, 44, 66};
	cout << "二叉树数值：" << endl;
	for (int i = 0; i < 6; i++)
	{
		cout << array[i] << " ";
		tree->insert(array[i]);  //调用插入函数，生成二叉查找树
	}
 
	cout << endl << endl;
 
 
	/************************/
	/*          遍历           
	/************************/
	cout << "前序遍历：";
	tree->PreOrder();
	cout << endl;
 
	cout << "中序遍历：";
	tree->InOrder();
	cout << endl;
 
	cout << "后序遍历：";
	tree->PostOrder();
	cout << endl << endl;
 
 
	/************************/
	/*          查找           
	/************************/
	int keyword;  //查找节点的关键字
	cout << "请输入要查找的节点：";
	cin >> keyword;
	cout << endl;
	BSTNode *node = tree->IteratorSearch(keyword);  //获取数值的地址
	if (node)  //判断有没有地址
	{
		cout << "关键字为“" << keyword << "”的节点，存在。" << endl ;
	}
	else
	{
		cout << "关键字为“" << keyword << "”的节点，不存在。" << endl;
	}
	cout << endl << endl;
 
 
	/************************/
	/*          删除
	/************************/
	int DelNode;  //要删除的节点
	cout << "请输入要删除的节点：";
	cin >> DelNode;
	tree->remove(DelNode);
	cout << endl;
 
	cout << "删除操作后，(前序)遍历：";
	tree->PreOrder();
	cout << endl;
	cout << "删除操作后，(中序)遍历：";
	tree->InOrder();
	cout << endl;
	cout << "删除操作后，(后序)遍历：";
	tree->PostOrder();
	cout << endl << endl;
	
 
	/************************/
	/*          销毁
	/************************/
	tree->destroy();
 
 
	system("pause");
	return 0;
}