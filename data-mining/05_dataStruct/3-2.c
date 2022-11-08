// 在一个非递减序列中出现次数最多的元素称为众数，用分治法寻找有序序列中的众数。

#include <stdio.h>
#include <stdlib.h>
#define MAX 20
int mode = 0;//众数
int multiplicity = 0;//重数

void FindMode(int n[], int low, int high)
{
    //找到中位数
    int mid = n[(low + high) / 2];

    //与中位数相等的第一个数与最后一个数的位置
    int first = 0, last = 0;
    for (int i = low; i < high; i++)
    {
        if (n[i] == mid)
        {
            first = i;
            break;
        }
    }
    for (int i = first; i < high; i++)
    {
        if (n[i] != mid)
        {
            last = i;
            break;
        }
    }

    if (multiplicity < last - first + 1)
    {
        multiplicity = last - first + 1;
        mode = mid;
    }

    if (first + 1 > last - first + 1)
    {
        FindMode(n, low, first);
    }

    if (high - last + 1 > last - first + 1)
    {
        FindMode(n, last, high);
    }
}

void QuickSort(int* arr, int low, int high)
{
    if (low < high)
    {
        int i = low;
        int j = high;
        int k = arr[low];
        while (i < j)
        {
            while (i < j && arr[j] >= k)     // 从右向左找第一个小于k的数
            {
                j--;
            }

            if (i < j)
            {
                arr[i++] = arr[j];
            }

            while (i < j && arr[i] < k)      // 从左向右找第一个大于等于k的数
            {
                i++;
            }

            if (i < j)
            {
                arr[j--] = arr[i];
            }
        }

        arr[i] = k;

        // 递归调用
        QuickSort(arr, low, i - 1);     // 排序k左边
        QuickSort(arr, i + 1, high);    // 排序k右边
    }
}

int main()
{
    int a[MAX];
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &a[i]);
    }

    //快速排序
    QuickSort(a, 0, n - 1);
    for (int i = 0; i < n; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");

    FindMode(a, 0, n - 1);
    printf("众数为：%d", mode);
    return 0;
}