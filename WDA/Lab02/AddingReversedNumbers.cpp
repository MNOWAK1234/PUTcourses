#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace std;
int n;
int a, b;
string s;
string s1, s2;
int r1, r2;
int space;
int sum;
string rs;

int main()
{
    getline(cin, s);
    n = atoi(s.c_str());
    for (int i = 0; i < n; i++)
    {
        s = "";
        s1 = "";
        s2 = "";
        rs = "";
        getline(cin, s);
        space = s.find(' ');
        for (int j = space - 1; j >= 0; j--)
        {
            s1 += s[j];
        }
        for (int j = s.size() - 1; j > space; j--)
        {
            s2 += s[j];
        }
        r1 = atoi(s1.c_str());
        r2 = atoi(s2.c_str());
        sum = r1 + r2;
        while (sum > 0 && sum % 10 == 0)
            sum /= 10;
        s = "";
        s = to_string(sum);
        for (int j = s.size() - 1; j >= 0; j--)
        {
            rs += s[j];
        }
        cout << rs << endl;
    }
    return 0;
}