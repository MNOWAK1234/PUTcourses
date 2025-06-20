#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int n;
unsigned long long x;
unsigned long long zmieniona;
const int greatest = 285112;
const int power = 20;
unsigned long long cnt[greatest + 8];
unsigned long long sum[greatest + 8];
unsigned long long digits[greatest + 8][power + 1];
const unsigned long long maxx = 10000000000000000;
unsigned long long twos[20];
int half;
int value;

int binsearch(unsigned long long x)
{
    int p = 0;
    int k = greatest;
    int s;
    while (p < k)
    {
        s = (p + k) / 2;
        if (sum[s] < x)
            p = s + 1;
        else
            k = s;
    }
    return p;
}
string accurate(unsigned long long v)
{
    string result = "";
    unsigned long long oper;
    unsigned long long possibilities;
    bool take;
    possibilities = cnt[v];
    for (int i = power; i > 1; i--)
    {
        take = false;
        for (int j = 9; j > 0; j--)
        {
            oper = j * twos[i - 1];
            if (v >= oper)
            {
                if (x > possibilities - digits[v - oper][i - 1])
                {
                    result += char(j + int('0'));
                    x -= (possibilities - digits[v - oper][i - 1]);
                    possibilities = digits[v - oper][i - 1];
                    v = v - oper;
                    take = true;
                    break;
                }
                else
                {
                    possibilities -= digits[v - oper][i - 1];
                }
            }
        }
        if (take == false)
        {
            if (result != "")
                result += '0';
        }
    }
    result += char(v + int('0'));
    return result;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    cnt[0] = 1;
    sum[0] = 1;
    cnt[1] = 1;
    cnt[2] = 2;
    cnt[3] = 2;
    cnt[4] = 4;
    cnt[5] = 4;
    cnt[6] = 6;
    cnt[7] = 6;
    cnt[8] = 10;
    cnt[9] = 10;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 1; j <= power; j++)
        {
            digits[i][j] = 1;
        }
    }
    for (int i = 2; i <= power; i++)
    {
        digits[2][i] = 2;
        digits[3][i] = 2;
        digits[4][i] = 4;
        digits[5][i] = 4;
        digits[6][i] = 6;
        digits[7][i] = 6;
        digits[8][i] = 10;
        digits[9][i] = 10;
    }
    digits[4][2] = 3;
    digits[5][2] = 3;
    digits[6][2] = 4;
    digits[7][2] = 4;
    digits[8][2] = 5;
    digits[8][3] = 9;
    digits[9][2] = 5;
    digits[9][3] = 9;
    for (int i = 1; i < 10; i++)
        sum[i] = sum[i - 1] + cnt[i];
    for (int i = 10; i <= greatest; i++)
    {
        half = i / 2;
        cnt[i] = cnt[half] + cnt[half - 1] + cnt[half - 2] + cnt[half - 3] + cnt[half - 4];
        sum[i] = sum[i - 1] + cnt[i];
        for (int j = 1; j <= power; j++)
        {
            digits[i][j] = digits[half][j - 1] + digits[half - 1][j - 1] + digits[half - 2][j - 1] + digits[half - 3][j - 1] + digits[half - 4][j - 1];
        }
    }
    sum[-1] = 0;
    twos[0] = 1;
    for (int i = 1; i < 20; i++)
        twos[i] = 2 * twos[i - 1];
    while (n--)
    {
        cin >> x;
        value = binsearch(x);
        x -= sum[value - 1];
        cout << accurate(value) << endl;
    }
}
