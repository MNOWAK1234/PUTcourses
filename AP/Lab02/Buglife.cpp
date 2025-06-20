#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int t;
int v, e;
vector<int> lss[2004];
int sex[2004];
int a, b;
bool sus;
void DFS(int v)
{
    for (int i = 0; i < (int)lss[v].size(); i++)
    {
        if (sex[lss[v][i]] == 0)
        {
            if (sex[v] == 1)
            {
                if (sex[lss[v][i]] == 1)
                    sus = true;
                else
                    sex[lss[v][i]] = 2;
            }
            else if (sex[v] == 2)
            {
                if (sex[lss[v][i]] == 2)
                    sus = true;
                else
                    sex[lss[v][i]] = 1;
            }
            DFS(lss[v][i]);
        }
        else
        {
            if (sex[lss[v][i]] == 2)
            {
                if (sex[v] == 2)
                    sus = true;
            }
            if (sex[lss[v][i]] == 1)
            {
                if (sex[v] == 1)
                    sus = true;
            }
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    for (int j = 1; j <= t; j++)
    {
        cout << "Scenario #" << j << ":\n";
        cin >> v >> e;
        sus = false;
        for (int i = 0; i < 2004; i++)
        {
            lss[i].clear();
            sex[i] = 0;
        }
        for (int i = 0; i < e; i++)
        {
            cin >> a >> b;
            lss[a].push_back(b);
            lss[b].push_back(a);
        }
        for (int j = 1; j <= v; j++)
        {
            if (sex[j] == 0)
            {
                sex[j] = 1;
                DFS(j);
            }
        }
        if (sus == true)
            cout << "Suspicious bugs found!\n";
        else
            cout << "No suspicious bugs found!\n";
    }
    return 0;
}
