#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

string s;
string l;
string r;
int ll, lx, rl, rx;
int n;
int f;
double wynik;
void podziallewo(string a)
{
    string pomoc = "";
    int x = 0;
    for (int i = 0; i < (int)a.size(); i++)
    {
        if (a[i] == '-' || a[i] == '+')
        {
            if (pomoc.size() != 0)
            {
                if (x == 1)
                {
                    lx += atoi(pomoc.c_str());
                }
                else
                {
                    ll += atoi(pomoc.c_str());
                }
            }
            pomoc = "";
            x = 0;
            if (a[i] == '-')
                pomoc += '-';
        }
        else
        {
            if (a[i] == 'x')
            {
                if (i > 0)
                {
                    if (a[i - 1] == '-' || a[i - 1] == '+')
                        pomoc += '1';
                }
                else
                {
                    pomoc += '1';
                }
                x = 1;
            }
            pomoc += a[i];
        }
    }
    if (x == 1)
    {
        lx += atoi(pomoc.c_str());
    }
    else
    {
        ll += atoi(pomoc.c_str());
    }
}
void podzialprawo(string a)
{
    string pomoc = "";
    int x = 0;
    for (int i = 0; i < (int)a.size(); i++)
    {
        if (a[i] == '-' || a[i] == '+')
        {
            if (pomoc.size() != 0)
            {
                if (x == 1)
                {
                    rx += atoi(pomoc.c_str());
                }
                else
                {
                    rl += atoi(pomoc.c_str());
                }
            }
            pomoc = "";
            x = 0;
            if (a[i] == '-')
                pomoc += '-';
        }
        else
        {
            if (a[i] == 'x')
            {
                if (i > 0)
                {
                    if (a[i - 1] == '-' || a[i - 1] == '+')
                        pomoc += '1';
                }
                else
                {
                    pomoc += '1';
                }
                x = 1;
            }
            pomoc += a[i];
        }
    }
    if (x == 1)
    {
        rx += atoi(pomoc.c_str());
    }
    else
    {
        rl += atoi(pomoc.c_str());
    }
}

int main()
{
    getline(cin, s);
    n = atoi(s.c_str());
    for (int i = 0; i < n; i++)
    {
        ll = 0;
        lx = 0;
        rx = 0;
        rl = 0;
        wynik = 0;
        l = "";
        r = "";
        getline(cin, s);
        f = s.find('=');
        for (int j = 0; j < f; j++)
            l += s[j];
        for (int j = f + 1; j < (int)s.size(); j++)
            r += s[j];
        podziallewo(l);
        podzialprawo(r);
        lx -= rx;
        rl -= ll;
        if (lx == 0)
            cout << "NO" << endl;
        else
        {
            wynik = (double)rl / (double)lx;
            cout << wynik << endl;
        }
    }
    return 0;
}
