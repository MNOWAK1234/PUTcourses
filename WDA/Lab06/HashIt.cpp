#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int t;
int n;
string tab[101];
string word, real;
int num;

int hash(string key)
{
    int wynik = 0;
    for (unsigned long long i = 0; i < key.size(); i++)
    {
        wynik += int(key[i]) * (i + 1);
        wynik %= 101;
    }
    wynik *= 19;
    wynik %= 101;
    return wynik;
}

bool add(string tab[101], string key)
{
    int wynik = 0;
    int next;
    for (unsigned long long i = 0; i < key.size(); i++)
    {
        wynik += int(key[i]) * (i + 1);
        wynik %= 101;
    }
    wynik *= 19;
    wynik %= 101;
    int help = wynik;
    if (tab[help] == key)
        return false;
    else
    {
        for (int j = 1; j <= 19; j++)
        {
            next = help + (23 * j) + (j * j);
            next %= 101;
            if (tab[next] == key)
                return false;
        }
    }
    if (tab[help] == "")
    {
        tab[help] = key;
        return true;
    }
    for (int j = 1; j <= 19; j++)
    {
        next = help + (j * j) + (23 * j);
        next %= 101;
        if (tab[next] == "")
        {
            tab[next] = key;
            return true;
        }
    }
    return false;
}

bool del(string tab[], string key)
{
    for (int i = 0; i < 101; i++)
    {
        if (tab[i] == key)
        {
            tab[i] = "";
            return true;
        }
    }
    return false;
}

int main()
{
    cin >> t;
    while (t--)
    {
        num = 0;
        for (int i = 0; i < 101; i++)
            tab[i] = "";
        cin >> n;
        while (n--)
        {
            cin >> word;
            real = word.substr(4);
            if (word[0] == 'A')
            {
                if (add(tab, real) == true)
                    num++;
            }
            else
            {
                if (del(tab, real) == true)
                    num--;
            }
        }
        cout << num << "\n";
        for (int i = 0; i < 101; i++)
        {
            if (tab[i] != "")
                cout << i << ":" << tab[i] << "\n";
        }
    }
}