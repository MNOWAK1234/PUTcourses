#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <deque>
#include <queue>
#include <stack>
#include <string>
#include <bitset>
#include <cstdio>
#include <limits>
#include <vector>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <iostream>

#include <vector>

using namespace std;

int euklides(int a, int b)
{
    if (b == 0)
        return a;
    else
        return euklides(b, a % b);
}

int n, x, y;
int l[1000001];

int main()
{

    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> n >> x >> y;
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = i; j < n + 1; j++)
        {
            l[j * j - i * i] += 1;
        }
    }
    int e = euklides(x, y);
    int mx = n * n;
    int rx = int(x / e);
    int ry = int(y / e);
    int px = rx;
    int py = ry;
    int wynik = 0;
    wynik += l[0] * l[0];
    while (px <= mx && py <= mx)
    {
        wynik += l[px] * l[py] * 2;
        px += rx;
        py += ry;
    }
    cout << wynik << endl;
}
