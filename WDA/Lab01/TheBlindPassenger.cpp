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

int t, n;
int row;

using namespace std;
int main()
{
    cin >> t;
    for (int i = 0; i < t; ++i)
    {
        cin >> n;
        if (n == 1)
            cout << "poor conductor" << endl;
        else
        {
            n -= 2;
            row = (n / 5) + 1;
            n++;
            cout << row << " ";
            if (row % 2 == 1)
            {
                switch (n % 5)
                {
                case 1:
                    cout << "W L" << endl;
                    break;
                case 2:
                    cout << "A L" << endl;
                    break;
                case 3:
                    cout << "A R" << endl;
                    break;
                case 4:
                    cout << "M R" << endl;
                    break;
                case 0:
                    cout << "W R" << endl;
                    break;
                }
            }
            else
            {
                switch (n % 5)
                {
                case 0:
                    cout << "W L" << endl;
                    break;
                case 4:
                    cout << "A L" << endl;
                    break;
                case 3:
                    cout << "A R" << endl;
                    break;
                case 2:
                    cout << "M R" << endl;
                    break;
                case 1:
                    cout << "W R" << endl;
                    break;
                }
            }
        }
    }
    return 0;
}