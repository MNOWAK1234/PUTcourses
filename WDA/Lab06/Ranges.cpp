#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int t;
int n;
int wynik;
int p;
vector<pair<int, int>> ranges;
pair<int, int> help;

int main()
{
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        wynik = 0;
        cin >> n;
        for (int j = 0; j < n; j++)
        {
            cin >> help.first >> help.second;
            ranges.push_back(help);
        }
        sort(ranges.begin(), ranges.end());
        wynik += ranges[0].second - ranges[0].first + 1;
        p = ranges[0].second;
        for (int j = 1; j < n; j++)
        {
            if (ranges[j].second > p)
            {
                if (ranges[j].first <= p)
                {
                    wynik += ranges[j].second - p;
                    p = ranges[j].second;
                }
                else
                {
                    wynik += ranges[j].second - ranges[j].first + 1;
                    p = ranges[j].second;
                }
            }
        }
        cout << wynik << "\n";
        ranges.clear();
    }
}