#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int t;
string a;
double h1, h2, m1, m2, t1, t2;
double h;
int result;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    h = double(12 * 60) / double(11);
    while (t--)
    {
        cin >> a;
        h1 = atoi(a.substr(0, 2).c_str());
        m1 = atoi(a.substr(3).c_str());
        cin >> a;
        h2 = atoi(a.substr(0, 2).c_str());
        m2 = atoi(a.substr(3).c_str());
        t1 = 60 * h1 + m1;
        t2 = 60 * h2 + m2;
        result = (int)t2 / h - int(t1 / h);
        cout << result << "\n";
    }
    return 0;
}