#include <cmath>
#include <iostream>

using namespace std;

string dod(string x, string y)
{
    string w = "";
    int bin = 0;
    int sum = 0;
    while (x.size() < y.size())
        x = "0" + x;
    while (x.size() > y.size())
        y = "0" + y;
    for (int i = x.size() - 1; i >= 0; i--)
    {
        sum = int(x[i] - '0') + int(y[i] - '0') + bin;
        if (sum > 9)
        {
            sum -= 10;
            bin = 1;
        }
        else
            bin = 0;
        w = char(sum + int('0')) + w;
    }
    if (bin == 1)
        w = "1" + w;
    return w;
}
string odj(string x, string y)
{
    string w = "";
    int bin = 0;
    int odj = 0;
    while (x.size() < y.size())
        x = "0" + x;
    while (x.size() > y.size())
        y = "0" + y;
    for (int i = x.size() - 1; i >= 0; i--)
    {
        odj = int(x[i] - '0') - int(y[i] - '0') - bin;
        if (odj < 0)
        {
            odj += 10;
            bin = 1;
        }
        else
            bin = 0;
        w = char(odj + int('0')) + w;
    }
    if (bin == 1)
        w = "1" + w;
    return w;
}
int t;
string b, c, d;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> b >> c >> d;
        b = odj(c, b);
        d = odj(c, d);
        c = dod(b, d);
        cout << c << "\n";
    }
}