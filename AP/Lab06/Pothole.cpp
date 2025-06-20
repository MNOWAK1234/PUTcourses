#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <climits>

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

int bipartiteMatchingSlower(int sizeGroup1, int sizeGroup2, vector<pair<int, int>> edges)
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
        for (int j = 0; j <= sizeGroup1; j++)
            visitedGroup1[j] = false;
        for (int j = 0; j <= sizeGroup2; j++)
            visitedGroup2[j] = false;
        extend = false;
        for (int j = 0; j <= sizeGroup1; j++)
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

vector<int> odl[207];

vector<vector<int>> FloydWarshall(int vertices)
{
    vector<vector<int>> distances;
    vector<int> row;
    int OverflowBuffer = 1000;
    for (int i = 0; i < vertices; i++)
        row.push_back(INT_MAX / 2 - OverflowBuffer);
    for (int i = 0; i < vertices; i++)
        distances.push_back(row);
    for (int i = 0; i < vertices; i++)
    {
        distances[i][i] = 0;
        for (int j = 0; j < odl[i].size(); j++)
        {
            distances[i][odl[i][j]] = 1;
        }
    }
    for (int k = 0; k < vertices; k++)
    {
        for (int i = 0; i < vertices; i++)
        {
            for (int j = 0; j < vertices; j++)
            {
                if ((distances[i][k] + distances[k][j]) < distances[i][j])
                {
                    distances[i][j] = distances[i][k] + distances[k][j];
                }
            }
        }
    }
    return distances;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin >> t;
    int komory, dol;
    int chamber;
    vector<int> in, out;
    vector<pair<int, int>> edges;
    while (t--)
    {
        for (int i = 0; i < 207; i++)
        {
            odl[i].clear();
            group1[i].clear();
            group2[i].clear();
        }
        int m = 0;
        in.clear();
        out.clear();
        edges.clear();
        cin >> komory;
        cin >> dol;
        for (int i = 0; i < dol; i++)
        {
            cin >> chamber;
            if (chamber == komory)
            {
                m = 1;
                continue;
            }
            in.push_back(chamber - 1);
            odl[0].push_back(chamber - 1);
        }
        for (int i = 1; i < komory - 1; i++)
        {
            cin >> dol;
            for (int j = 0; j < dol; j++)
            {
                cin >> chamber;
                if (chamber == komory)
                {
                    out.push_back(i);
                }
                odl[i].push_back(chamber - 1);
            }
        }
        vector<vector<int>> dist = FloydWarshall(komory);
        for (int i = 0; i < in.size(); i++)
        {
            for (int j = 0; j < out.size(); j++)
            {
                if (dist[in[i]][out[j]] < 2000)
                {
                    edges.push_back(make_pair(in[i], out[j]));
                }
            }
        }
        cout << bipartiteMatchingSlower(komory, komory, edges) + m << endl;
    }
}
