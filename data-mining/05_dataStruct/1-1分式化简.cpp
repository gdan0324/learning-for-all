// 分式化简。设计算法，将一个给定的真分数化简为最简分数形式。例如，将 6/8 化简为 3/4。

#include<iostream>
#include<stdio.h>
using namespace std;
//namespace是指标识符的各种可见范围
int main() {
    int n;//分子
    int m;//分母
    int factor;//最大因子
    int factor1;
    cout << "输入一个真分数的分子和分母：" << endl;
    cin >> n >> m;

    int r = m % n;
    factor1 = m;
    factor = n;
    while (r != 0)
    {
        factor1 = factor;
        factor = r;
        r = factor1 % factor;
    }
    cout << "输出该真分数的最简形式:" << (n / factor) << "/" << (m / factor) << endl;
    return 0;
}
