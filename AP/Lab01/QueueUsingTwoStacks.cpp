#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stack>
using namespace std;

int t;
int type;
stack<int> s, q;
vector<int> three;
int cnt;
int help;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> type;
        if (type == 1)
        {
            cin >> help;
            s.push(help);
        }
        else
        {
            if (type == 2)
                cnt++;
            else
            {
                three.push_back(cnt);
            }
        }
    }
    while (!s.empty())
    {
        help = s.top();
        s.pop();
        q.push(help);
    }
    int prev = 0;
    for (int i = 0; i < (int)three.size(); i++)
    {
        while (prev < three[i])
        {
            q.pop();
            prev++;
        }
        cout << q.top() << endl;
    }
}
