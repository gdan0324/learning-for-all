// 用分治法设计算法，求解一个数组中的最大元素。

#include<iostream>
using namespace std;

//[) 左闭右开
int findMax(int a[], int l, int r)
{
	int x;
	int m = l + (r - l) / 2;		//防止过大溢出
	if (r - l == 1)return a[l];	//只有一个元素
	else
	{
		//分治法
		int u = findMax(a, l, m);
		int v = findMax(a, m, r);
		x = max(u, v);
	}
	return x;
}

int main()
{
	int N = 10;
	int a[] = { 16,2,3,4,48,6,7,8,29,10 };
	cout << findMax(a, 0, N) << endl;
	return 0;
}
