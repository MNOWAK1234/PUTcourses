#include <iostream>
#include <vector>

using namespace std;

int t;
unsigned long long sum;
vector<int> v, two;
int help;
int arr[100004], tab[100004];
void dod(unsigned long long a)
{
    sum += (a + 1) * a / 2;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> arr[i];
    }
    if (t < 3)
    {
        if (t == 1)
            cout << 1 << "\n";
        else
        {
            if (arr[0] == arr[1])
                cout << 2 << "\n";
            else
                cout << 3 << "\n";
        }
    }
    else
    {
        if (arr[1] >= arr[0])
        {
            tab[0] = 1;
            v.push_back(0);
        }
        for (int i = 1; i < t - 1; i++)
        {
            if (arr[i - 1] >= arr[i] && arr[i + 1] >= arr[i])
            {
                tab[i] = 1;
                v.push_back(i);
            }
            else if (arr[i - 1] <= arr[i] && arr[i + 1] <= arr[i])
            {
                tab[i] = 2;
                two.push_back(i);
            }
        }
        if (arr[t - 1] <= arr[t - 2])
        {
            tab[t - 1] = 1;
            v.push_back(t - 1);
        }
        dod(v[0] + 1);
        dod(t - v[v.size() - 1]);
        int j = 0;
        for (int i = 0; i < (int)two.size(); i++)
        {
            while (two[i] > v[j + 1])
            {
                j++;
                sum++;
            }
            help = 0;
            while (i + 1 < (int)two.size() && two[i + 1] < v[j + 1])
            {
                i++;
                help++;
                v[j]++;
            }
            dod(max(two[i] - v[j], v[j + 1] - two[i]) + 1);
            dod(min(two[i] - v[j], v[j + 1] - two[i]));
            sum += help * (min(two[i] - v[j], v[j + 1] - two[i]) + 1);
            j++;
            sum--;
        }
        sum--;
        sum += (v.size() - j - 1);
        cout << sum << "\n";
    }
}