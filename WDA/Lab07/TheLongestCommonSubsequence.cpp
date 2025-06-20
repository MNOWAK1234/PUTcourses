#include <iostream>
#include <vector>

using namespace std;

int tab[103][103];
vector<int> a, b, result;
int h;
int n, m, N, M;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> n >> m;
    N = n;
    M = m;
    for (int i = 0; i < n; i++)
    {
        cin >> h;
        a.push_back(h);
    }
    for (int i = 0; i < m; i++)
    {
        cin >> h;
        b.push_back(h);
    }
    for (int i = 1; i <= (int)a.size(); i++)
    {
        for (int j = 1; j <= (int)b.size(); j++)
        {
            if (b[j - 1] != a[i - 1])
                tab[i][j] = max(tab[i - 1][j], tab[i][j - 1]);
            else
                tab[i][j] = tab[i - 1][j - 1] + 1;
        }
    }
    h = tab[n][m];
    while (h > 0)
    {
        if (tab[n][m] > max(tab[n - 1][m], tab[n][m - 1]))
        {
            h--;
            result.push_back(a[n - 1]);
            tab[n][m] = 0;
            n--;
            m--;
        }
        else if (tab[n][m] == tab[n - 1][m])
            n--;
        else
            m--;
    }
    for (int i = (int)result.size() - 1; i >= 0; i--)
        cout << result[i] << " ";
}