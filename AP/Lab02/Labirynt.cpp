#include <iostream>

using namespace std;

int one[1004][1004];
int two[1004][1004];

int t;
int r, c;
char h;
int mx;
int px, py;

void DFS(int x, int y, int tab[1004][1004])
{
    if (tab[x][y] > mx)
    {
        mx = tab[x][y];
        px = x;
        py = y;
    }
    if (tab[x - 1][y] == 0)
    {
        tab[x - 1][y] = tab[x][y] + 1;
        DFS(x - 1, y, tab);
    }
    if (tab[x + 1][y] == 0)
    {
        tab[x + 1][y] = tab[x][y] + 1;
        DFS(x + 1, y, tab);
    }
    if (tab[x][y - 1] == 0)
    {
        tab[x][y - 1] = tab[x][y] + 1;
        DFS(x, y - 1, tab);
    }
    if (tab[x][y + 1] == 0)
    {
        tab[x][y + 1] = tab[x][y] + 1;
        DFS(x, y + 1, tab);
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
        cin >> c >> r;
        mx = 0;
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                cin >> h;
                if (h == '#')
                {
                    one[i][j] = -1;
                    two[i][j] = -1;
                }
                else
                {
                    one[i][j] = 0;
                    two[i][j] = 0;
                }
            }
        }
        for (int i = 1; i < r - 1; i++)
        {
            for (int j = 1; j < c - 1; j++)
            {
                if (one[i][j] == 0)
                {
                    one[i][j] = 1;
                    DFS(i, j, one);
                }
            }
        }
        mx = 0;
        two[px][py] = 1;
        DFS(px, py, two);
        mx--;
        cout << "Maximum rope length is " << mx << ".\n";
    }
}
