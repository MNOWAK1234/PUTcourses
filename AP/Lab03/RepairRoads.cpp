#include <iostream>
#include <vector>

using namespace std;

vector<int> lss[10007];
int t;
int n;
int przetworzone[10007];
int wierzcholki[10007];
int roboty[10007];
void listas(int e)
{
    int a, b;
    for (int i = 0; i < e; i++)
    {
        cin >> a >> b;
        lss[a].push_back(b);
        lss[b].push_back(a);
    }
}
int wynik;
int licznik;
void DFStree(int v)
{
    licznik++;
    przetworzone[v] = licznik;
    wierzcholki[v] = 1;
    for (int i = 0; i < (int)lss[v].size(); i++)
    {
        if (przetworzone[lss[v][i]] == 0)
        {
            DFStree(lss[v][i]);
            if (roboty[lss[v][i]] > 0 && wierzcholki[lss[v][i]] == 1)
            {
                if (wierzcholki[v] < 2)
                    wierzcholki[v] = 1;
            }
            else
                wierzcholki[v] = wierzcholki[lss[v][i]] + 1;
            if (wierzcholki[lss[v][i]] >= 2)
                roboty[v]++;
        }
    }
    if (roboty[v] % 2 == 1)
    {
        wierzcholki[v] = 2;
        roboty[v]--;
    }
    else if (roboty[v] > 0)
    {
        wierzcholki[v] = 1;
    }
    wynik += roboty[v] / 2;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> n;
        licznik = 0;
        for (int i = 0; i < n + 1; i++)
        {
            przetworzone[i] = 0;
            roboty[i] = 0;
            wierzcholki[i] = 0;
            lss[i].clear();
        }
        listas(n - 1);
        wynik = 0;
        DFStree(0);
        if (wierzcholki[0] > 1)
            wynik++;
        cout << wynik << endl;
    }
}
