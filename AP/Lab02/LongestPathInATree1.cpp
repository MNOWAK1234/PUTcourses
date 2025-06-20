#include <iostream>
#include <vector>

using namespace std;

vector<int> lss[10007];
int n;
long long przetworzone[10007];
int mws[10007]; // maksymalna wartosc sciezki;
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
int mx;
int licznik;
int p;
bool sukces;
int spr;
void DFStree(int v)
{
    licznik++;
    przetworzone[v] = licznik;
    long long p1 = -2;
    long long p2 = -2;
    for (int i = 0; i < lss[v].size(); i++)
    {
        if (przetworzone[lss[v][i]] == 0)
        {
            DFStree(lss[v][i]);
        }
    }
    if (lss[v].size() == 1 && przetworzone[lss[v][0]] < przetworzone[v])
    {
        mws[v] == 0;
        p1 = 0;
        p2 = 0;
    }
    else if (lss[v].size() == 2)
    {
        if (przetworzone[lss[v][0]] > przetworzone[v])
        {
            if (przetworzone[lss[v][1]] > przetworzone[v])
                mws[v] = mws[lss[v][0]] + 2;
            else
                mws[v] = mws[lss[v][0]] + 1;
        }
        else
        {
            mws[v] = mws[lss[v][1]] + 1;
        }
    }
    else
    {
        for (int i = 0; i < lss[v].size(); i++)
        {
            if (przetworzone[lss[v][i]] > przetworzone[v])
            {
                if (mws[lss[v][i]] > p2)
                {
                    if (mws[lss[v][i]] >= p1)
                    {
                        p2 = p1;
                        p1 = mws[lss[v][i]];
                    }
                    else
                    {
                        p2 = mws[lss[v][i]];
                    }
                }
            }
        }
        if (p1 + p2 + 2 > mx)
            mx = p1 + p2 + 2;
        mws[v] = p1 + 1;
    }
    if (mws[v] > mx)
        mx = mws[v];
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> n;
    for (int i = 0; i < n + 1; i++)
    {
        przetworzone[i] = 0;
        mws[i] = 0;
    }
    listas(n - 1);
    mx = 0;
    DFStree(1);
    cout << mx << endl;
}
