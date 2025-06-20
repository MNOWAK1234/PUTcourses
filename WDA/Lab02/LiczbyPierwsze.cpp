#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int n;
int t;

string prime(int n)
{
    string w = "TAK";
    if (n <= 1)
        return "NIE";
    else if (n == 2)
        return "TAK";
    else
    {
        if (n % 2 == 0)
            w = "NIE";
        for (int i = 3; i * i <= n; i += 2)
        {
            if (n % i == 0)
                w = "NIE";
        }
        return w;
    }
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        cin >> t;
        cout << prime(t) << endl;
    }
    return 0;
}