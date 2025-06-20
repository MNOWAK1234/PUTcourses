#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int n, x, y;

int euklides1(int a, int b)
{
    if (b > a)
        swap(a, b);
    if (b == 0)
        return a;
    else
        return euklides1(a - b, b);
}

int main()
{

    cin >> n;
    for (int i = 0; i < n; i++)
    {
        cin >> x >> y;
        if (y == 0)
            y = x;
        cout << euklides1(x, y) << endl;
    }
    return 0;
}