#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

int t, a, b;

int main()
{

    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> a >> b;
        if ((a + b) % 2 == 0)
            cout << "NO\n";
        else
            cout << "YES\n";
    }
    return 0;
}
