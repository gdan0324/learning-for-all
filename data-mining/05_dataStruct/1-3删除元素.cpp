// 设计算法，在数组 r[n]中删除所有元素值为 x 的元素，要求时间复杂度为𝑂(𝑛)，
// 空间复杂性为𝑂(1)。


#include<cstdio>
#include<cstdlib>
using namespace std;
void delete_x(int* a, int* b, int len);
static int x = 7;
int main() {
    int a[] = { 1,2,3,4,5,6,7,7,7 }, len, * newarry;
    len = sizeof(a) / sizeof(a[0]);
    newarry = (int*)malloc(len * sizeof(int));//动态开辟一个新数组
    delete_x(a, newarry, len);
    return 0;
}
void delete_x(int* a, int* b, int len) {
    int co = 0;
    for (int i = 0, j = 0; i < len; i++) {
        if (a[i] == x)
            continue;
        else {
            b[j] = a[i];
            j++;
        }
        co = j;
    }
    for (int i = 0; i < co; i++)//length<-co 输出删减后的数组
        printf("%d ", b[i]);
}
