#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

vector<int> group1[50002];
vector<int> group2[50002];
bool visitedGroup1[50002];
bool visitedGroup2[50002];
int matchingGroup1[50002];
int matchingGroup2[50002];

bool matching(int current, int group)
{
    if (group == 1)
    {
        visitedGroup1[current] = true;
        for (int i = 0; i < group1[current].size(); i++)
        {
            if (matchingGroup2[group1[current][i]] == -1)
            {
                matchingGroup2[group1[current][i]] = current;
                matchingGroup1[current] = group1[current][i];
                return true;
            }
        }
        for (int i = 0; i < group1[current].size(); i++)
        {
            if (visitedGroup2[group1[current][i]] == false && matching(group1[current][i], 2))
            {
                matchingGroup2[group1[current][i]] = current;
                matchingGroup1[current] = group1[current][i];
                return true;
            }
        }
        return false;
    }
    else
    {
        visitedGroup2[current] = true;
        return matching(matchingGroup2[current], 1);
    }
}

int bipartiteMatching(int sizeGroup1, int sizeGroup2, vector<pair<int, int>> edges)
{
    int result = 0;
    for (int i = 0; i < edges.size(); i++)
    {
        group1[edges[i].first].push_back(edges[i].second);
        group2[edges[i].second].push_back(edges[i].first);
    }
    for (int j = 0; j <= sizeGroup1; j++)
        matchingGroup1[j] = -1;
    for (int j = 0; j <= sizeGroup2; j++)
        matchingGroup2[j] = -1;
    bool extend = true;
    while (extend == true)
    {
        for (int j = 1; j <= sizeGroup1; j++)
            visitedGroup1[j] = false;
        for (int j = 1; j <= sizeGroup2; j++)
            visitedGroup2[j] = false;
        extend = false;
        for (int j = 1; j <= sizeGroup1; j++)
        {
            if (matchingGroup1[j] == -1)
            {
                if (matching(j, 1) == true)
                {
                    result++;
                    extend = true;
                }
            }
        }
    }
    return result;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int n, m, p, one, two;
    cin >> n >> m >> p;
    pair<int, int> para;
    vector<pair<int, int>> krawedzie;
    for (int i = 0; i < p; i++)
    {
        cin >> one >> two;
        para = make_pair(one, two);
        krawedzie.push_back(para);
    }
    int wynik = 0;
    wynik = bipartiteMatching(n, m, krawedzie);
    cout << wynik << endl;
}
