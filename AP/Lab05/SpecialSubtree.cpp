#include <iostream>
#include <vector>
#include <queue>

using namespace std;

priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
vector<pair<int, int>> lss[3007];
bool przetworzone[3007];
int v, e;
int a, b, c;
int res;
int current;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> v >> e;
    for (int i = 0; i < e; i++)
    {
        cin >> a >> b >> c;
        lss[a].push_back(make_pair(c, b));
        lss[b].push_back(make_pair(c, a));
    }
    v--;
    current = 1;
    while (v--)
    {
        przetworzone[current] = true;
        for (int i = 0; i < (int)lss[current].size(); i++)
        {
            pq.push(lss[current][i]);
        }
        while (przetworzone[pq.top().second] == true)
            pq.pop();
        res += pq.top().first;
        current = pq.top().second;
        pq.pop();
    }
    cout << res << endl;
}