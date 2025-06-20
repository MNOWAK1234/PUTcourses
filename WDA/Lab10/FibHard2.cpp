#include <iostream>
#include <string>

using namespace std;

int t;
long long p, np;
string n;
long long a, b, c, d, num;
const long long mod = 998244353;
const long long period = 1996488708;

void doubling_fib(long long num)
{
    if (num == 0)
    {
        p = 0;
        np = 1;
        return;
    }
    doubling_fib(num / 2);
    a = p;
    b = np;
    c = 2 * b - a;
    if (c < 0)
        c += mod;
    c = (a * c) % mod;
    d = (a * a + b * b) % mod;
    if (num % 2 == 0)
    {
        p = c;
        np = d;
    }
    else
    {
        p = d;
        np = c + d;
    }
}
int main()
{

    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> n;
        num = 0;
        for (long long i = 0; i < n.size(); i++)
        {
            num = (num * 2 + (long long)(n[i] - '0')) % period;
        }
        p = 0;
        np = 0;
        doubling_fib(num);
        cout << p << "\n";
    }
}
