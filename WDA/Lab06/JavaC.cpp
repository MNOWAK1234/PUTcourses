#include <iostream>

using namespace std;

string a;
bool java, c, error;

int main()
{
    while (getline(cin, a))
    {
        java = false;
        c = false;
        error = false;
        for (unsigned long long i = 0; i < a.size(); i++)
        {
            if (a[i] == '_')
            {
                c = true;
                if (i + 1 < a.size())
                {
                    if (a[i + 1] == '_')
                        error = true;
                }
            }
            if (int(a[i]) <= int('Z') && int(a[i]) >= int('A'))
                java = true;
        }
        if (java == true && c == true)
            error = true;
        if (a[0] == '_' || a[a.size() - 1] == '_')
            error = true;
        if (int(a[0]) <= int('Z') && int(a[0]) >= int('A'))
            error = true;
        if (error == true)
            a = "Error!";
        else
        {
            for (unsigned long long i = 0; i < a.size(); i++)
            {
                if (a[i] == '_')
                {
                    a.erase(i, 1);
                    a[i] = toupper(a[i]);
                }
                else if (int(a[i]) <= int('Z') && int(a[i]) >= int('A'))
                {
                    a[i] = tolower(a[i]);
                    a.insert(i, "_");
                }
            }
        }
        cout << a << "\n";
    }
}