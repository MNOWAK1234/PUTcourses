#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

string a, b;
int t;

int LongestCommonSubsequence(string one, string two)
{
    vector<vector<int>> table;
    vector<int> help;
    for (int i = 0; i <= two.size(); i++)
        help.push_back(0);
    for (int i = 0; i <= one.size(); i++)
        table.push_back(help);
    for (int i = 1; i <= one.size(); i++)
    {
        for (int j = 1; j <= two.size(); j++)
        {
            if (one[j - 1] == two[i - 1])
            {
                table[i][j] = table[i - 1][j - 1] + 1;
            }
            else
            {
                table[i][j] = max(table[i - 1][j], table[i][j - 1]);
            }
        }
    }
    return one.size() - table[one.size()][two.size()];
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> a;
        b = "";
        for (int i = 0; i < a.size(); i++)
            b += a[a.size() - 1 - i];
        cout << LongestCommonSubsequence(a, b) << endl;
    }
}
