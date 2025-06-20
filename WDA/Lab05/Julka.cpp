#include <iostream>
#include <string>

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
string div(string x, int y)
{
    string w = "";
    int liczba = 0;
    int bin = 0;
    int dziel = 0;
    for (int i = 0; i < x.size(); i++)
    {
        liczba = int(x[i] - '0') + 10 * bin;
        dziel = liczba / y;
        bin = liczba % y;
        w += char(dziel + int('0'));
    }
    while (w[0] == '0' && w.size() != 1)
        w.erase(0, 1);
    if (bin == 1)
        w += ".5";
    return w;
}

string a, b;
string mniej, wiecej;
int main()
{
    for (int i = 0; i < 10; i++)
    {
        cin >> a >> b;
        wiecej = dod(a, b);
        wiecej = div(wiecej, 2);
        mniej = odj(a, b);
        mniej = div(mniej, 2);
        cout << wiecej << "\n"
             << mniej << "\n";
    }
}
