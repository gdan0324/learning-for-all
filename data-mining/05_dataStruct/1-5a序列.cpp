// 给定一个有𝑛(𝑛 ≥ 1)个整数的序列，求出其中最大连续子序列的和，并分析其
// 时间复杂度。
// 例如序列（-2，11，-4，13，-5，-2）的最大子序列和为 20。规定一个序列的最
// 大连续子序列和至少是 0，如果小于 0，其结果为 0。

#include<stdio.h>
int Maxsum(int a[], int n) {
	int i, j, k;
	int maxSum = 0, curSum;
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			curSum = 0;
			for (k = i; k <= j; k++)
				curSum += a[k];
			if (curSum > maxSum)
				maxSum = curSum;
		}
	}
	return maxSum;
}
int main() {
	int a[] = { -2,11,-4,13,-5,-2 };
	int n = sizeof(a) / sizeof(a[0]);
	printf("a序列最大和：%ld\n", Maxsum(a, n));
}
