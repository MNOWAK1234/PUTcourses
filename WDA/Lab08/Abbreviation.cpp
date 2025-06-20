#include <iostream>

using namespace std;

bool d[1007][1007];
int t;
string a, b;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> a >> b;
        for (int i = 0; i < 1007; i++)
        {
            for (int j = 0; j < 1007; j++)
            {
                d[i][j] = false;
            }
        }
        d[0][0] = true;
        for (int i = 0; i < (int)a.size(); i++)
        {
            for (int j = 0; j <= (int)b.size(); j++)
            {
                if (d[i][j])
                {
                    if (j < (int)b.size() && toupper(a[i]) == b[j])
                    {
                        d[i + 1][j + 1] = true;
                    }
                    if (islower(a[i]))
                    {
                        d[i + 1][j] = true;
                    }
                }
            }
        }

        if (d[a.size()][b.size()])
            cout << "YES\n";
        else
            cout << "NO\n";
    }
}
