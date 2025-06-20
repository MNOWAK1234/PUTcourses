#include <iostream>
#include <vector>

using namespace std;
int licznik;
int wynik;
vector<int> lss[100008];
int przetworzone[100008];
bool maszyny[100008];
int male[100008];
int a, b, c;
int n, k;

void DFS(int v)
{
    licznik++;
    przetworzone[v] = licznik;
    for (int i = 0; i < (int)lss[v].size(); i += 2)
    {
        if (przetworzone[lss[v][i]] == 0)
        {
            DFS(lss[v][i]);
            if (maszyny[lss[v][i]] == true || male[lss[v][i]] != 1000000000)
            {
                if (male[v] != 1000000000 && maszyny[lss[v][i]] == false)
                {
                    if (lss[v][i + 1] < male[lss[v][i]])
                        male[lss[v][i]] = lss[v][i + 1];
                    wynik += min(male[v], male[lss[v][i]]);
                    male[v] = max(male[v], male[lss[v][i]]);
                }
                else if (male[v] != 1000000000)
                {
                    wynik += min(male[v], lss[v][i + 1]);
                    male[v] = max(male[v], lss[v][i + 1]);
                }
                else
                {
                    if (lss[v][i + 1] < male[lss[v][i]])
                        male[v] = lss[v][i + 1];
                    else
                        male[v] = male[lss[v][i]];
                }
            }
        }
    }
    if (maszyny[v] == true)
    {
        if (male[v] != 1000000000)
        {
            wynik += male[v];
            male[v] = 1000000000;
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> k;
    for (int i = 0; i < n - 1; i++)
    {
        cin >> a >> b >> c;
        lss[a].push_back(b);
        lss[a].push_back(c);
        lss[b].push_back(a);
        lss[b].push_back(c);
    }
    for (int i = 0; i < n; i++)
    {
        przetworzone[i] = 0;
        maszyny[i] = false;
        male[i] = 1000000000;
    }
    for (int i = 0; i < k; i++)
    {
        cin >> a;
        maszyny[a] = true;
    }
    DFS(0);
    cout << wynik << endl;
}
