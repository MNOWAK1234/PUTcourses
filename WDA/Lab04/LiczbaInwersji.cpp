#include <iostream>

using namespace std;

long long t;
int n;
int tab[200004];
long long inwersje;
void scal(int start, int srodek, int koniec, int tab[])
{
    int i = start;
    int j = srodek + 1;
    int k = start;
    int *pomoc = new int[n];
    while (i <= srodek && j <= koniec)
    {
        if (tab[i] <= tab[j])
        {
            pomoc[k] = tab[i];
            i++;
            k++;
        }
        else
        {
            pomoc[k] = tab[j];
            inwersje += (j - k);
            j++;
            k++;
        }
    }
    while (i <= srodek)
    {
        pomoc[k] = tab[i];
        i++;
        k++;
    }
    while (j <= koniec)
    {
        pomoc[k] = tab[j];
        j++;
        k++;
    }
    for (int k = start; k <= koniec; k++)
    {
        tab[k] = pomoc[k];
    }
    delete[] pomoc;
}
void mergesort(int start, int koniec, int tab[])
{
    if (start < koniec)
    {
        int s = (start + koniec) / 2;
        mergesort(start, s, tab);
        mergesort(s + 1, koniec, tab);
        scal(start, s, koniec, tab);
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    for (long long a = 0; a < t; a++)
    {
        cin >> n;
        inwersje = 0;
        for (int b = 0; b < n; b++)
        {
            cin >> tab[b];
        }
        mergesort(0, n - 1, tab);
        for (int b = 0; b < 200004; b++)
            tab[b] = 0;
        cout << inwersje << "\n";
        inwersje = 0;
    }
}