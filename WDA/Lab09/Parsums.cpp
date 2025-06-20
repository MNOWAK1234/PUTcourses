#include <iostream>
#include <vector>

using namespace std;

int t, h;
int res, last, sum, point, aid;
int help;
vector<int> v, minusy;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    while (t != 0)
    {
        v.clear();
        minusy.clear();
        sum = 0;
        for (int i = 0; i < t; i++)
        {
            cin >> h;
            v.push_back(h);
            if (h < 0)
                minusy.push_back(i);
        }
        res = t - minusy.size();
        for (int i = minusy.size() - 1; i >= 0; i--)
        {
            help = minusy[i];
            sum = v[help];
            last = minusy[i];
            aid = 0;
            while (sum < 0 && help > 0)
            {
                help--;
                sum += v[help];
                if (v[help] < 0)
                    i--;
                else if (sum < 0)
                {
                    res--;
                    aid += v[help] + 1;
                    v[help] = -1;
                }
                else if (sum >= 0)
                    v[help] += aid;
            }
        }
        point = t - 1;
        while (sum < 0 && point > last)
        {
            sum += v[point];
            if (sum < 0 && v[point] >= 0)
                res--;
            point--;
        }
        cout << res << "\n";
        cin >> t;
    }
}
