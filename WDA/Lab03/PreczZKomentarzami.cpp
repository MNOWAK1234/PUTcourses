#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

string s;
string w;
int lk;
int k1, k2;
int cudzyslow;
char m1, m2;

int main()
{
    while (getline(cin, s))
    {
        w = "";
        k2 = 0;
        lk = k1;
        for (int i = 0; i < (int)s.size(); i++)
        {
            if (k1 == 0)
            {
                if (k2 == 0)
                {
                    if (cudzyslow == 0)
                    {
                        if (s[i] == '/')
                        {
                            if (i != (int)s.size() - 1)
                            {
                                if (s[i + 1] == '/')
                                {
                                    k2 = 1;
                                }
                                else if (s[i + 1] == '*')
                                {
                                    k1 = 1;
                                    i++;
                                }
                                else
                                {
                                    w += '/';
                                }
                            }
                            else
                            {
                                w += '/';
                            }
                        }
                        else
                        {
                            if (s[i] == '"' && m1 != '\'')
                            {
                                cudzyslow = 1;
                            }
                            w += s[i];
                        }
                    }
                    else
                    {
                        w += s[i];
                        if ((s[i] == '\n' && m1 != '\\') || (s[i] == '"' && m1 != '\\') || (s[i] == '"' && m1 == '\\' && m2 == '\\'))
                        {
                            cudzyslow = 0;
                        }
                    }
                }
            }
            else
            {
                if (s[i] == '*')
                {
                    if (i != (int)s.size() - 1)
                    {
                        if (s[i + 1] == '/')
                        {
                            k1 = 0;
                            i++;
                        }
                    }
                }
            }
            m2 = m1;
            m1 = s[i];
        }
        if (lk != 1)
            cout << w << endl;
        else if (w.size() > 0)
            cout << w << endl;
    }
    return 0;
}
