#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int n, q;
long long a, b;
vector<long long> w[262144];
vector<long long> pp[262144];
long long minf = -1000000000;
long long pinf = 1000000000;
long long xf, yf, xs, ys;
vector<string> odp;
bool clause;
long double crossd;
long long cross;
int mem, wmem;

long long value(long long a, long long b, long long x)
{
    return a * x + b;
}
void build(int p, int k, int where)
{
    pp[where].push_back(minf);
    if (p < k)
    {
        int s = (p + k) / 2;
        build(p, s, 2 * where + 1);
        build(s + 1, k, 2 * where + 2);
        int left = (int)w[2 * where + 1].size() / 2;
        int right = (int)w[2 * where + 2].size() / 2;
        int iterl = 0;
        int iterr = 0;
        int which = 0;
        long long curr = minf;
        long long vleft = value(w[2 * where + 1][2 * iterl], w[2 * where + 1][2 * iterl + 1], curr);
        long long vright = value(w[2 * where + 2][2 * iterr], w[2 * where + 2][2 * iterr + 1], curr);
        if (vleft > vright)
        {
            which = 1;
            w[where].push_back(w[2 * where + 1][2 * iterl]);
            w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
        }
        else if (vleft < vright)
        {
            which = 2;
            w[where].push_back(w[2 * where + 2][2 * iterr]);
            w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
        }
        else
        {
            which = 0;
        }
        while (iterl < left && iterr < right)
        {
            if (which == 1)
            {
                if (w[2 * where + 1][2 * iterl] == w[2 * where + 2][2 * iterr])
                    iterr++;
                else
                {
                    crossd = ((long double)w[2 * where + 2][2 * iterr + 1] - (long double)w[2 * where + 1][2 * iterl + 1]) / ((long double)w[2 * where + 1][2 * iterl] - (long double)w[2 * where + 2][2 * iterr]);
                    cross = (long long)crossd;
                    if (crossd != cross && crossd < 0)
                        cross--;
                    if (cross <= pp[where][pp[where].size() - 1])
                        iterr++;
                    else if (cross >= pp[2 * where + 2][iterr + 1])
                        iterr++;
                    else if (cross < pp[2 * where + 1][iterl + 1])
                    {
                        if (cross > pp[2 * where + 2][iterr + 1])
                            iterr++;
                        else if (cross == pp[2 * where + 2][iterr + 1])
                        {
                            iterr++;
                            iterl++;
                            which = 2;
                            pp[where].push_back(cross);
                            w[where].push_back(w[2 * where + 2][2 * iterr]);
                            w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                        }
                        else
                        {
                            which = 2;
                            pp[where].push_back(cross);
                            w[where].push_back(w[2 * where + 2][2 * iterr]);
                            w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                            iterl++;
                        }
                    }
                    else if (cross > pp[2 * where + 1][iterl + 1])
                    {
                        which = 1;
                        iterl++;
                        pp[where].push_back(pp[2 * where + 1][iterl]);
                        w[where].push_back(w[2 * where + 1][2 * iterl]);
                        w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                    }
                    else
                    {
                        pp[where].push_back(cross);
                        which = 3;
                        iterl++;
                    }
                }
            }
            else if (which == 2)
            {
                if (w[2 * where + 1][2 * iterl] == w[2 * where + 2][2 * iterr])
                    iterl++;
                else
                {
                    crossd = ((long double)w[2 * where + 2][2 * iterr + 1] - (long double)w[2 * where + 1][2 * iterl + 1]) / ((long double)w[2 * where + 1][2 * iterl] - (long double)w[2 * where + 2][2 * iterr]);
                    cross = (long long)crossd;
                    if (crossd != cross && crossd < 0)
                        cross--;
                    if (cross <= pp[where][pp[where].size() - 1])
                        iterl++;
                    else if (cross >= pp[2 * where + 1][iterl + 1])
                        iterl++;
                    else if (cross < pp[2 * where + 2][iterr + 1])
                    {
                        if (cross > pp[2 * where + 1][iterl + 1])
                            iterl++;
                        else if (cross == pp[2 * where + 1][iterl + 1])
                        {
                            iterl++;
                            iterr++;
                            which = 1;
                            pp[where].push_back(cross);
                            w[where].push_back(w[2 * where + 1][2 * iterl]);
                            w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                        }
                        else
                        {
                            which = 1;
                            pp[where].push_back(cross);
                            w[where].push_back(w[2 * where + 1][2 * iterl]);
                            w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                            iterr++;
                        }
                    }
                    else if (cross > pp[2 * where + 2][iterr + 1])
                    {
                        which = 2;
                        iterr++;
                        pp[where].push_back(pp[2 * where + 2][iterr]);
                        w[where].push_back(w[2 * where + 2][2 * iterr]);
                        w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                    }
                    else
                    {
                        pp[where].push_back(cross);
                        which = 3;
                        iterr++;
                    }
                }
            }
            else if (which == 0)
            {
                if (w[2 * where + 1][2 * iterl] > w[2 * where + 2][2 * iterr])
                {
                    which = 1;
                    iterr++;
                    w[where].push_back(w[2 * where + 1][2 * iterl]);
                    w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                }
                else if (w[2 * where + 1][2 * iterl] < w[2 * where + 2][2 * iterr])
                {
                    which = 2;
                    iterl++;
                    w[where].push_back(w[2 * where + 2][2 * iterr]);
                    w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                }
                else
                {
                    which = 0;
                    if (pp[2 * where + 1][iterl + 1] < pp[2 * where + 2][iterr + 1])
                    {
                        pp[where].push_back(pp[2 * where + 1][iterl + 1]);
                        w[where].push_back(w[2 * where + 1][2 * iterl]);
                        w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                        iterl++;
                    }
                    else
                    {
                        pp[where].push_back(pp[2 * where + 2][iterr + 1]);
                        w[where].push_back(w[2 * where + 2][2 * iterr]);
                        w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                        iterr++;
                    }
                }
            }
            else
            {
                curr = pp[where][pp[where].size() - 1] + 1;
                vleft = value(w[2 * where + 1][2 * iterl], w[2 * where + 1][2 * iterl + 1], curr);
                vright = value(w[2 * where + 2][2 * iterr], w[2 * where + 2][2 * iterr + 1], curr);
                if (vleft > vright)
                {
                    which = 1;
                    w[where].push_back(w[2 * where + 1][2 * iterl]);
                    w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
                }
                else if (vleft < vright)
                {
                    which = 2;
                    w[where].push_back(w[2 * where + 2][2 * iterr]);
                    w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
                }
                else
                {
                    which = 0;
                }
            }
        }
        while (iterr < right - 1)
        {
            pp[where].push_back(pp[2 * where + 2][iterr + 1]);
            iterr++;
            w[where].push_back(w[2 * where + 2][2 * iterr]);
            w[where].push_back(w[2 * where + 2][2 * iterr + 1]);
        }
        while (iterl < left - 1)
        {
            pp[where].push_back(pp[2 * where + 1][iterl + 1]);
            iterl++;
            w[where].push_back(w[2 * where + 1][2 * iterl]);
            w[where].push_back(w[2 * where + 1][2 * iterl + 1]);
        }
    }
    else
    {
        cin >> a >> b;
        w[where].push_back(a);
        w[where].push_back(b);
    }
    pp[where].push_back(pinf);
    mem++;
}
void check(int p, int k, int where)
{
    if (a <= p && b >= k)
    {
        int fr = 0;
        int en = (int)pp[where].size() - 1;
        int mid;
        while (fr < en)
        {
            mid = (fr + en) / 2;
            if (pp[where][mid] < xf)
                fr = mid + 1;
            else
                en = mid;
        }
        if (mid == fr)
            mid--;
        if (value(w[where][mid * 2], w[where][mid * 2 + 1], xf) >= yf)
        {
            clause = true;
        }
    }
    else
    {
        int s = (p + k) / 2;
        if (s >= a)
            check(p, s, 2 * where + 1);
        if (s + 1 <= b)
            check(s + 1, k, 2 * where + 2);
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    build(0, n - 1, 0);
    cin >> q;
    for (int i = 1; i <= q; i++)
    {
        cin >> a >> b >> xf >> yf;
        a--;
        b--;
        clause = false;
        check(0, n - 1, 0);
        if (clause)
            cout << "YES\n";
        else
            cout << "NO\n";
    }
}
