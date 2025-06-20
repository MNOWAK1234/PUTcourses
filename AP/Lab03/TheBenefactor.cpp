#include <iostream>
#include <vector>

using namespace std;

vector<int> lss[50007];
int przetworzone[10007];
int wartosci[50007];
int mws[10007]; // maksymalna wartosc sciezki;
void listas(int e)
{
    int a, b, c;
    for (int i = 0; i < e; i++)
    {
        cin >> a >> b >> c;
        lss[a].push_back(b);
        lss[a].push_back(c);
        lss[b].push_back(a);
        lss[b].push_back(c);
    }
}
int n;
int mx;
int licznik;
int p;
int t;
void DFStree(int v)
{
    licznik++;
    przetworzone[v] = licznik;
    long long p1 = 0;
    long long p2 = 0;
    for (int i = 0; i < (int)lss[v].size(); i += 2)
    {
        if (przetworzone[lss[v][i]] == 0)
        {
            wartosci[lss[v][i]] = lss[v][i + 1];
            DFStree(lss[v][i]);
            if (wartosci[lss[v][i]] + mws[lss[v][i]] > p1)
            {
                p2 = p1;
                p1 = wartosci[lss[v][i]] + mws[lss[v][i]];
            }
            else if (wartosci[lss[v][i]] + mws[lss[v][i]] > p2)
            {
                p2 = wartosci[lss[v][i]] + mws[lss[v][i]];
            }
        }
    }
    mws[v] = p1 + wartosci[v];
    if (mws[v] > mx)
        mx = mws[v];
    if (p1 + p2 > mx)
        mx = p1 + p2;
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
        mx = 0;
        for (int i = 0; i < n + 1; i++)
        {
            przetworzone[i] = 0;
            mws[i] = 0;
            wartosci[i] = 0;
        }
        for (int i = 0; i < 50007; i++)
            lss[i].clear();
        listas(n - 1);
        mx = 0;
        DFStree(1);
        cout << mx / 2 << endl;
    }
}