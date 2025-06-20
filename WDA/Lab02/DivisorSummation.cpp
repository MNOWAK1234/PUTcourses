#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int n, t;
int sum;

int main()
{
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        cin >> t;
        sum = 1;
        for (int j = 2; j * j <= t; j++)
        {
            if (t % j == 0)
            {
                if (j != t / j)
                {
                    sum += j;
                    sum += t / j;
                }
                else
                {
                    sum += j;
                }
            }
        }
        if (t == 1)
            sum = 0;
        cout << sum << endl;
    }
    return 0;
}