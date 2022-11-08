#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stack>
using namespace std;

long long ans;
int a[1000000];
void merge(int *a, int first, int mid, int last)
{
    int *temp = new int[last - first + 1];
    int first1 = first, last1 = mid;
    int first2 = mid + 1, last2 = last;
    int index = 0;
    while (first1 <= last1 && first2 <= last2)
    {
        if (a[first1] <= a[first2])
        {
            temp[index++] = a[first1++];
        }
        else
        {
            ans += last1 - first1 + 1;
            temp[index++] = a[first2++];
        }
    }
    while (first1 <= last1)
    {
        temp[index++] = a[first1++];
    }
    while (first2 <= last2)
    {
        temp[index++] = a[first2++];
    }
    int i;
    for (i = first; i <= last; i++)
    {
        a[i] = temp[i - first];
    }
    delete[] temp;
    return;
}
void inv_pair(int *a, int first, int last)
{
    if (last - first > 0)
    {
        int mid = (last + first) / 2;
        inv_pair(a, first, mid);
        inv_pair(a, mid + 1, last);
        merge(a, first, mid, last);
    }
    return;
}
int main()
{
    int n, i;
    while (cin >> n)
    {
        ans = 0;
        for (i = 0; i < n; i++)
            cin >> a[i];
        inv_pair(a, 0, n - 1);
        cout << ans << endl;
    }
    return 0;
}