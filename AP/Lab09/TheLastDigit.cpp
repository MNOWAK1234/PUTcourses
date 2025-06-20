#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int t, a, b;
vector<int> last[10];
int rest;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    last[0].push_back(0);
    last[1].push_back(1);
    last[2].push_back(6);
    last[2].push_back(2);
    last[2].push_back(4);
    last[2].push_back(8);
    last[3].push_back(1);
    last[3].push_back(3);
    last[3].push_back(9);
    last[3].push_back(7);
    last[4].push_back(6);
    last[4].push_back(4);
    last[5].push_back(5);
    last[6].push_back(6);
    last[7].push_back(1);
    last[7].push_back(7);
    last[7].push_back(9);
    last[7].push_back(3);
    last[8].push_back(6);
    last[8].push_back(8);
    last[8].push_back(4);
    last[8].push_back(2);
    last[9].push_back(1);
    last[9].push_back(9);
    while (t--)
    {
        cin >> a >> b;
        if (b == 0)
            cout << 1 << endl;
        else
        {
            a = a % 10;
            b %= last[a].size();
            cout << last[a][b] << endl;
        }
    }
}