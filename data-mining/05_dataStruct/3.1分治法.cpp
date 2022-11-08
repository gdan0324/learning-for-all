// 几何问题中的分治法
// 设p1=(X1,y1)，p2 =(2 y2)，…,pn= (Xn,yn)是平面上n个点构成的集合S，最近对问题就是找出集合S中距离最近的点对（为简单起见，只需找出其中一对）。

#include <iostream>
#include<cmath>
#include <math.h>
using namespace std;
typedef struct {
	int x;
	int y;
}Point;
void sort(Point*& point, int n)			//对点进行排序
{
	Point c;
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			if (point[i].x > point[j].x)
			{
				c = point[i];
				point[i] = point[j];
				point[j] = c;
			}
		}
	}
}
double fun2(Point a, Point b)	//计算两点之间距离
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
double mmin(double a, double b) //两个数的最小值
{
	if (a < b)
		return a;
	return b;
}
double fun(Point* point, int n)		//分治法
{
	if (n == 2)
		return fun2(point[0], point[1]);
	double min;
	double c;
	if (n == 3)
	{
		min = fun2(point[0], point[1]);
		c = fun2(point[0], point[2]);
		min = mmin(min, c);
				c = fun2(point[2], point[1]);
		min = mmin(min, c);
		return min;
	}
	Point* lpoint = new Point[n / 2];		//左边部分
	Point* rpoint = new Point[n - n / 2];	//右边部分
	Point* mpoint = new Point[n];			//左右边界部分
		//对两侧进行赋值
	for (int i = 0; i < n / 2; i++)
		lpoint[i] = point[i];
	for (int i = n / 2; i < n; i++)
		rpoint[i - n / 2] = point[i];
	min = fun(lpoint, n / 2);			//求左侧
	c = fun(rpoint, n - n / 2);			//求右侧
	min = mmin(min, c);					//两侧中的最小值
		//对于中间的部分，求解需要用暴力求解，不然程序可能会因为递归太多层造成内存溢出
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (abs(point[i].x - point[n / 2].x)<min)
		{	//如果光x的距离相差就比min大，那么这两点的距离一定大于等于min
			mpoint[count] = point[i];
			count++;
		}
		if (point[i].x - point[n / 2].x > min)
			break;
	}
	for (int i = 0; i < count; i++)
	{
		for (int j = i + 1; j < count; j++)
		{	//暴力求解
			c = fun2(mpoint[i], mpoint[j]);
			min = mmin(min, c);
		}
	}
	return min;
}
int main()
{
	int n, m;
	cin >> n;
	while (n--)
	{
		cin >> m;
		Point* point = new Point[m];
		for (int i = 0; i < m; i++)
		{
			cin >> point[i].x >> point[i].y;
		}
		if (m == 1)
		{
			printf("%.4lf\n", 1e20);
			continue;
		}
		sort(point, m);
		printf("%.4lf\n", fun(point, m));
	}
	return 0;
}