#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

unsigned long long t, test, res, five;
bool ex;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        ex = true;
        res = 0;
        five = (unsigned long long)5;
        cin >> test;
        while (ex)
        {
            if (five > test)
            {
                ex = false;
            }
            else
            {
                res += test / five;
                five *= (unsigned long long)5;
            }
        }
        cout << res << endl;
    }
}