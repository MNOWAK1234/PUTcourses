#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int t;
int a, b;
int pa, pb;
int wynik;
int tab[100000];
int licznik;

int prime(int n)
{
    int w = 1;
    if (n <= 1)
        return 0;
    else if (n == 2)
        return 1;
    else
    {
        if (n % 2 == 0)
            w = 0;
        for (int i = 3; i * i <= n; i += 2)
        {
            if (n % i == 0)
                w = 0;
        }
        return w;
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    tab[2] = 1;
    licznik = 1;
    for (int i = 3; i * i < 1000000000; i++)
    {
        if (prime(i) == 1)
        {
            licznik++;
        }
        tab[i] = licznik;
    }
    for (int j = 0; j < t; j++)
    {
        cin >> a >> b;
        if (a == 1)
            a++;
        if (b == 1)
            b++;
        pa = int(sqrt(a));
        pb = int(sqrt(b));
        wynik = tab[pb] - tab[pa];
        if ((int)sqrt(a - 1) != pa)
        {
            if (tab[pa] != tab[pa - 1])
            {
                wynik++;
            }
        }
        cout << wynik << "\n";
    }
    return 0;
}