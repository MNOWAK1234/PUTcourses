#include <iostream>
#include <vector>
#include <set>

using namespace std;

int t;
vector<pair<int, int>> v;
pair<int, int> m;
set<double> z;
int x, y;
double help;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    while (t)
    {
        v.clear();
        z.clear();
        for (int i = 0; i < t; i++)
        {
            cin >> x >> y;
            m = make_pair(x, y);
            v.push_back(m);
        }
        for (int i = 0; i < t; i++)
        {
            for (int j = i + 1; j < t; j++)
            {
                if (((double)v[j].first - (double)v[i].first) == 0)
                    z.insert(100000);
                else
                {
                    help = ((double)v[j].second - (double)v[i].second) / ((double)v[j].first - (double)v[i].first);
                    z.insert(help);
                }
            }
        }
        cout << z.size() << "\n";
        cin >> t;
    }
}
