#include <iostream>

using namespace std;

void insertion_sort(int a[], int len)
{
    int i, j, temp;
    for (i = 1; i < len; i++) //控制趟数
    {
        temp = a[i];
        for (j = i; j > 0 && temp < a[j - 1]; j--) // 无序区的数据与有序区的数据元素比较
        {
            a[j] = a[j - 1]; //将有序区的元素后移
        }
        a[j] = temp;
    }
}

int main()
{
    int a[] = {22, 66, 34, 13, 6, 5};

    insertion_sort(a, 6);

    for (int i = 0; i < 6; i++)
    {
        cout << a[i] << " ";
    }

    return 0;
}