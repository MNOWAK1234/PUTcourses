#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

string c;

int main()
{

    while (getline(cin, c))
    {
        for (int i = 0; i < (int)c.size(); ++i)
        {
            if (c[i] == ' ')
                cout << " ";
            else
            {
                cout << char((int(c[i]) - int('A') + 3) % 26 + int('A'));
            }
        }
        cout << endl;
    }
    return 0;
}