#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>

using namespace std;

vector<int> przyjazd[103], powrot[103];
int n, c, r;
string garage;
int garaz;
int licznik;
vector<string> popsute;
string broken;
string liczba;
int cost;
string pierwsze, drugie, znak;
int p, d;
vector<string> wszystkie;
int dp[103][2];
int dd[103][2];
int suma;
int car;

int findindex(string s)
{
    for (int i = 0; i < wszystkie.size(); i++)
    {
        if (s == wszystkie[i])
            return i;
    }
    wszystkie.push_back(s);
    return wszystkie.size() - 1;
}
void Dijkstra(int v, int tab[103][2], vector<int> vec[103])
{
    int mn;
    int przetwarzany = v;
    tab[przetwarzany][1] = 0;
    for (int i = 0; i < n; i++)
    {
        mn = 100000000;
        for (int j = 0; j < n; j++)
        {
            if (tab[j][0] == 0 && tab[j][1] < mn)
            {
                mn = tab[j][1];
                przetwarzany = j;
            }
        }
        tab[przetwarzany][0] = 1;
        for (int j = 0; j < vec[przetwarzany].size(); j += 2)
        {
            if (tab[vec[przetwarzany][j]][1] > (tab[przetwarzany][1] + vec[przetwarzany][j + 1]))
            {
                tab[vec[przetwarzany][j]][1] = (tab[przetwarzany][1] + vec[przetwarzany][j + 1]);
            }
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> c >> r;
    while (n != 0)
    {
        for (int i = 0; i < 103; i++)
        {
            przyjazd[i].clear();
            powrot[i].clear();
            dp[i][0] = 0;
            dp[i][1] = 1000000;
            dd[i][0] = 0;
            dd[i][1] = 1000000;
        }
        popsute.clear();
        cin >> garage;
        licznik++;
        suma = 0;
        cout << licznik << ". ";
        for (int i = 0; i < c; i++)
        {
            cin >> broken;
            popsute.push_back(broken);
        }
        for (int i = 0; i < r; i++)
        {
            cin >> pierwsze >> znak >> drugie;
            p = findindex(pierwsze);
            d = findindex(drugie);
            liczba = "";
            for (int j = 2; j < znak.size() - 2; j++)
            {
                liczba += znak[j];
            }
            cost = atoi(liczba.c_str());
            if (znak[0] == '<')
            {
                przyjazd[d].push_back(p);
                przyjazd[d].push_back(cost);
                powrot[p].push_back(d);
                powrot[p].push_back(cost);
            }
            if (znak[znak.size() - 1] == '>')
            {
                przyjazd[p].push_back(d);
                przyjazd[p].push_back(cost);
                powrot[d].push_back(p);
                powrot[d].push_back(cost);
            }
        }
        garaz = findindex(garage);
        Dijkstra(garaz, dp, przyjazd);
        Dijkstra(garaz, dd, powrot);
        for (int i = 0; i < c; i++)
        {
            suma += dp[findindex(popsute[i])][1];
            suma += dd[findindex(popsute[i])][1];
        }
        cout << suma << "\n";
        cin >> n >> c >> r;
    }
}
