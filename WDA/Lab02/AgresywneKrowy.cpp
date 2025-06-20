#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int t;
int n, c;
int d;
vector<int> stajnia;
int h;
int mx, mn;

int dasie(int step, int d)
{
    int pomijam = d;
    int poprzedni = 0;
    int licznik = 1;
    while (pomijam >= 0 && poprzedni < n)
    {
        while (licznik < n && stajnia[licznik] - stajnia[poprzedni] < step)
        {
            licznik++;
            pomijam--;
        }
        poprzedni = licznik;
        licznik = poprzedni + 1;
    }
    if (pomijam >= 0)
        return 1;
    else
        return 0;
}

void bs(int mn, int mx)
{
    int p = mn;
    int k = mx;
    int s;
    while (k - p > 1)
    {
        s = (k + p) / 2;
        if (dasie(s, d) == 1)
        {
            p = s;
        }
        else
            k = s;
    }
    cout << p << endl;
}

int main()
{
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> n >> c;
        d = n - c;
        for (int j = 0; j < n; j++)
        {
            cin >> h;
            stajnia.push_back(h);
        }
        sort(stajnia.begin(), stajnia.end());
        mx = stajnia[stajnia.size() - 1];
        mn = 1;
        bs(mn, mx);
        stajnia.clear();
    }
    return 0;
}
