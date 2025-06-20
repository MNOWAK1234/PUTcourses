#include <iostream>
#include <vector>

using namespace std;

int g;
unsigned long long n, m, s, m2;
unsigned long long reszta, krotsze, iled, ilek, pomoc;
unsigned long long najdluzsza, krawedzie, maks, bez, wynik;
unsigned long long wynik2, najdluzsza2;
unsigned long long wynik3, najdluzsza3;

vector<unsigned long long> res;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> g;
    while (g--)
    {
        wynik = 0;
        cin >> n >> m >> s;
        m2 = m;
        najdluzsza = s / (n - 1);
        krotsze = najdluzsza;
        reszta = s % (n - 1);
        iled = (n - 1 + n - reszta) * reszta / 2;
        if (reszta != 0)
            najdluzsza++;
        krawedzie = m - (n - 1);
        maks = n * (n - 1) / 2;
        ilek = maks - iled;
        wynik += s;
        ilek -= (n - 1 - reszta);
        iled -= reszta;
        m -= (n - 1);
        if (m > 0)
        {
            if (m > ilek)
            {
                wynik += ilek * krotsze;
                m -= ilek;
                wynik += m * najdluzsza;
            }
            else
            {
                wynik += m * krotsze;
            }
        }
        wynik2 = 0;
        wynik3 = 0;
        pomoc = s / (n - 1);
        najdluzsza2 = s - (n - 2);
        najdluzsza3 = s - (n - 2) * pomoc;
        krawedzie = m2 - (n - 1);
        maks = n * (n - 1) / 2;
        bez = maks - (n - 2);
        if (m2 > bez)
        {
            wynik2 += (m2 - bez) * najdluzsza2;
            wynik3 += (m2 - bez) * najdluzsza3;
            krawedzie -= (m2 - bez);
        }
        wynik2 += krawedzie;
        wynik3 += (krawedzie * pomoc);
        wynik2 += s;
        wynik3 += s;
        cout << min(wynik, min(wynik2, wynik3)) << endl;
    }
}
