#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

vector<long double> tg[1503];
int zero[1503];
pair<long double, long double> h[1503];
int t;
int one, two;
long long res;
int s, e, ms, me;
long double help;
const long double pi = 3.14159265358979323846;

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> h[i].first >> h[i].second;
    }
    for (int i = 0; i < t; i++)
    {
        for (int j = i + 1; j < t; j++)
        {
            help = (long double)atan2(h[j].second - h[i].second, h[j].first - h[i].first);
            if (help < 0)
                help += pi;
            tg[i].push_back(help);
            tg[j].push_back(help);
        }
        sort(tg[i].begin(), tg[i].end());
        for (int j = 0; j < t - 1 && tg[i][j] <= pi / 2 + 1e-15; j++)
            res += upper_bound(tg[i].begin(), tg[i].end(), tg[i][j] + pi / 2 + 1e-15) - lower_bound(tg[i].begin(), tg[i].end(), tg[i][j] + pi / 2 - 1e-15);
    }
    cout << res << endl;
}