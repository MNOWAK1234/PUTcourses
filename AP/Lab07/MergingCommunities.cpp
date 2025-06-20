#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stack>

using namespace std;

char query;
int point[200007];
int value[200007];
int n, q;
int who;
int a, b;
int help;
int first_size, second_size;
int point_a, point_b;
bool which;
bool possible;
void fq(int v)
{
    if (point[v] != 0)
    {
        fq(point[v]);
        point[v] = who;
    }
    else
    {
        cout << value[v] << endl;
        who = v;
    }
}
void fsecond(int v)
{
    if (point[v] != 0)
    {
        fsecond(point[v]);
        if (which)
            point[v] = point_a;
        else
            point[v] = point_b;
    }
    else
    {
        second_size = value[v];
        point_b = v;
        if (point_a != point_b)
            possible = true;
        else
            possible = false;
        if (first_size > second_size)
            which = true;
        else
            which = false;
    }
}
void ffirst(int v)
{
    if (point[v] != 0)
    {
        ffirst(point[v]);
        if (which)
            point[v] = point_a;
        else
            point[v] = point_b;
    }
    else
    {
        first_size = value[v];
        point_a = v;
        fsecond(b);
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> q;
    for (int i = 1; i <= n; i++)
    {
        value[i] = 1;
    }
    while (q--)
    {
        cin >> query;
        if (query == 'Q')
        {
            cin >> who;
            fq(who);
        }
        else
        {
            cin >> a >> b;
            ffirst(a);
            if (possible)
            {
                if (which)
                {
                    point[point_b] = point_a;
                    value[point_a] += value[point_b];
                }
                else
                {
                    point[point_a] = point_b;
                    value[point_b] += value[point_a];
                }
            }
        }
    }
}
