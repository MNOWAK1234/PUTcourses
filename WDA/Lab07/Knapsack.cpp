#include <iostream>
#include <vector>

using namespace std;

vector<int> arr;
int tab[2004];
int t, n, k, h;

int knapsack(int n, int k, vector<int> arr)
{
    int mx = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        tab[arr[i]] = 1;
        if (arr[i] <= k)
        {
            if (arr[i] > mx)
                mx = arr[i];
        }
        if (arr[i] == k)
            return k;
    }
    for (int i = 1; i < k; i++)
    {
        if (tab[i] == 0)
            continue;
        if (i > mx)
            mx = i;
        for (int j = 0; j < arr.size(); j++)
        {
            if (arr[j] + i <= k)
            {
                if (arr[j] + i == k)
                    return k;
                tab[i + arr[j]] = 1;
            }
        }
    }
    return mx;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> n >> k;
        for (int i = 0; i < 2000; i++)
            tab[i] = 0;
        while (n--)
        {
            cin >> h;
            arr.push_back(h);
        }
        cout << knapsack(n, k, arr) << "\n";
        arr.clear();
    }
}