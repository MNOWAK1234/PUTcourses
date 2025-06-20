#include <iostream>
#include <climits>

using namespace std;

int n;
unsigned long long a, b, c, roznica;

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> n;
    while (n--)
    {
        c = ULLONG_MAX;
        cin >> a >> b;
        roznica = b - a;
        a &= b;
        while (roznica > 0)
        {
            c *= 2;
            roznica /= 2;
        }
        a &= c;
        cout << a << "\n";
    }
}
