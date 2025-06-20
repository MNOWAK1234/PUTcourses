#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int n, k, q;
int h;
int sum;
int tab[1000];
int nowe[1000];

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> n >> k >> q;
    while (n--)
    {
        for (int i = 0; i <= k; i++)
            tab[i] = 1;
        for (int i = k + 1; i < 1000; i++)
            tab[i] = 0;
        cin >> h;
        h--;
        sum = 0;
        while (h--)
        {
            for (int i = 1; i < k; i++)
            {
                nowe[i] = tab[i] + tab[i - 1] + tab[i + 1];
            }
            nowe[0] = tab[0] + tab[1];
            nowe[k] = tab[k] + tab[k - 1];
            for (int i = 0; i <= k; i++)
            {
                tab[i] = nowe[i] % q;
            }
        }
        for (int i = 0; i <= k; i++)
            sum += tab[i];
        sum %= q;
        cout << sum << " ";
    }
}
