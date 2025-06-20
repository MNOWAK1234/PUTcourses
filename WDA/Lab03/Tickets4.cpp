#include <iostream>

using namespace std;

int t;
int n, k;
long long a;
long long tab[100004];
long long curr, mx, position;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    for (int j = 0; j < t; j++)
    {
        cin >> n >> k;
        mx = 0;
        curr = 0;
        position = 0;
        for (int i = 0; i < k; i++)
        {
            cin >> a;
            tab[a]++;
            cin >> a;
            tab[a]--;
        }
        for (int i = 0; i < n; i++)
        {
            curr += tab[i];
            if (curr > mx)
            {
                mx = curr;
                position = i;
            }
        }
        for (int i = 0; i < 100004; i++)
        {
            tab[i] = 0;
        }
        cout << position << " " << mx << "\n";
    }
}