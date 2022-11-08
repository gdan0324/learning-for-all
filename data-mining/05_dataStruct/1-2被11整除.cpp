// 设计算法，判断一个大整数能否被 11 整除。可以通过以下方法：将该数的十
// 进制表示从右端开始，每两位一组构成一个整数，然后将这些数相加，判断其和
// 能否被 11 整除。例如，将 562843748 分割成 5、62、84、37 和 48，然后判断
// （5+62+84+37+48）能否被 11 整除。

#include<iostream>
#include<cstring>
using namespace std;
void JudgeDivision(int r[], int n)
{
    long int sum = 0;
    for (int i = n - 1; i >= 0; i = i - 2)
        if (i == 0)
            sum = sum + r[0];
        else
            sum = sum + r[i - 1] * 10 + r[i];
    if (sum % 11 == 0)
        cout << "Yes";
    else
        cout << "No";
    return;
}

int main()
{
    string str;
    //查询当前环境下string能容纳的字符个数
    //cout<<str.max_size()<<endl;
    cin >> str;
    int n = str.length();
    int r[1000] = { 0 };
    for (int i = 0; i < n; i++)
        r[i] = str[i] - '0';
    JudgeDivision(r, n);
    return 0;
}
