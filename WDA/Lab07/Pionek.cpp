#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int n, help;
int tab[7];

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> n;
    n--;
    cin >> help;
    while (n--)
    {
        cin >> tab[0];
        tab[0] += max(tab[1], max(tab[2], max(tab[3], max(tab[4], max(tab[5], tab[6])))));
        for (int i = 6; i > 0; i--)
            tab[i] = tab[i - 1];
    }
    cout << tab[0] + help;
    return 0;
}
