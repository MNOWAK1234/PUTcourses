#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int t, n;
int tab[100];

int main()
{

    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> n;
        for (int j = 0; j < 100; j++)
            tab[j] = 0;
        for (int j = n - 1; j >= 0; j--)
        {
            cin >> tab[j];
        }
        for (int j = 0; j < n; j++)
        {
            cout << tab[j] << " ";
        }
        cout << endl;
    }
    return 0;
}