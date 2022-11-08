// 设 M 是一个n乘以n的整数矩阵，其中每一行（从左到右）和每一列（从上到下）的元素都按升序排列。
// 设计分治算法确定一个给定的整数x是否在 M 中。
#include "stdio.h"
int M[5][5]=
{
	{ 1, 2, 3, 4, 5},
	{ 6, 7, 8, 9,10},
	{11,12,13,14,15},
	{16,17,18,19,20},
	{21,22,23,24,25}
};
int x=26;
int MatrixBinary(int M[5][5],int rb,int re,int cb,int ce)
{
	int rm=(rb+re)/2;
	int cm=(cb+ce)/2;
	if (rb>re || cb>ce)
	{
		return 0;
	}
	if(x==M[rm][cm])
	{
		printf("rowStart=%d colStart=%d M[rm][cm]=%d\n",rm,cm,M[rm][cm]);
		return 1;
	}
	else if (rb==re && cb==ce)
	{
		return 0;
	}
	if(x>M[rm][cm])
	{	
		return MatrixBinary(M,rb,re,cm+1,ce)||MatrixBinary(M,rm+1,re,cb,cm);
	}
	else
	{	
		return MatrixBinary(M,rb,rm-1,cb,ce)||MatrixBinary(M,rm,re,cb,cm-1);
	}
}
int main()
{
	int a=MatrixBinary(M,0,4,0,4);
	printf("flag=%d\n",a);
	return 0;
}