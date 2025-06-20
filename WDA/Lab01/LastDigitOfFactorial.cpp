#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int p;
unsigned long long a;

unsigned long long silnia(unsigned long long n)
{
    unsigned long long w = 1;
    if (n == 1 || n == 0)
        return w;
    else
    {
        for (unsigned long long i = 2; i <= n; ++i)
        {
            w *= i;
            w = w % 10000000000;
            while (w % 10 == 0)
            {
                w /= 10;
            }
        }
        return w;
    }
}

int main()
{
    cin >> p;
    for (int i = 0; i < p; i++)
    {
        cin >> a;
        cout << silnia(a) % 10 << endl;
    }
    return 0;
}