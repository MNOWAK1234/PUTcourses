#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

vector<int> mapa[200];
vector<int> jedynki;
int t;
int n, m;
char c;
queue<int> kolejka;
int x, y;

void wypisz()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << mapa[i][j] << "\t";
        }
        cout << endl;
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        jedynki.clear();
        for (int i = 0; i < 200; i++)
            mapa[i].clear();
        cin >> n >> m;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                cin >> c;
                if (c == '0')
                    mapa[i].push_back(1000);
                else
                {
                    mapa[i].push_back(0);
                    jedynki.push_back(i);
                    jedynki.push_back(j);
                }
            }
        }
        for (int i = 0; i < (int)jedynki.size(); i += 2)
        {
            kolejka.push(jedynki[i]);
            kolejka.push(jedynki[i + 1]);
            while (!kolejka.empty())
            {
                x = kolejka.front();
                kolejka.pop();
                y = kolejka.front();
                kolejka.pop();
                if (x > 0 && mapa[x - 1][y] != 0 && mapa[x - 1][y] > mapa[x][y] + 1)
                {
                    kolejka.push(x - 1);
                    kolejka.push(y);
                    mapa[x - 1][y] = mapa[x][y] + 1;
                }
                if (x < n - 1 && mapa[x + 1][y] != 0 && mapa[x + 1][y] > mapa[x][y] + 1)
                {
                    kolejka.push(x + 1);
                    kolejka.push(y);
                    mapa[x + 1][y] = mapa[x][y] + 1;
                }
                if (y > 0 && mapa[x][y - 1] != 0 && mapa[x][y - 1] > mapa[x][y] + 1)
                {
                    kolejka.push(x);
                    kolejka.push(y - 1);
                    mapa[x][y - 1] = mapa[x][y] + 1;
                }
                if (y < m - 1 && mapa[x][y + 1] != 0 && mapa[x][y + 1] > mapa[x][y] + 1)
                {
                    kolejka.push(x);
                    kolejka.push(y + 1);
                    mapa[x][y + 1] = mapa[x][y] + 1;
                }
            }
        }
        wypisz();
    }
}
