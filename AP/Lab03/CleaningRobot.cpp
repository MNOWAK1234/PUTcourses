#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

int a, b;
char c;
long long odleglosci[11][11];
int grid[11][22][22];
int miejsca[11][2];
int licznik;
long long wynik;
long long suma;
queue<int> kolejka;
int x, y;
bool zajete[11];

void spr(int p, int gdzie)
{
    for (int i = 1; i < licznik; i++)
    {
        if (zajete[i] == false)
        {
            bool kopiuj[11];
            for (int k = 0; k < 11; k++)
                kopiuj[k] = zajete[k];
            long long checkpoint = suma;
            suma += odleglosci[gdzie][i];
            zajete[i] = true;
            if (p + 1 < licznik)
                spr(p + 1, i);
            else
            {
                if (suma < wynik)
                    wynik = suma;
            }
            for (int k = 0; k < 11; k++)
                zajete[k] = kopiuj[k];
            suma = checkpoint;
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> a >> b;
    while (a > 0)
    {
        licznik = 1;
        wynik = 1000000;
        suma = 0;
        for (int i = 0; i < 11; i++)
        {
            for (int j = 0; j < 22; j++)
            {
                for (int k = 0; k < 22; k++)
                {
                    if (j == 0 || k == 0)
                        grid[i][j][k] = -1;
                    else if (j == b + 1 || k == a + 1)
                        grid[i][j][k] = -1;
                    else
                        grid[i][j][k] = 10000;
                }
            }
        }
        for (int i = 0; i < 11; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                odleglosci[i][j] = 0;
            }
        }
        for (int i = 0; i < 11; i++)
        {
            miejsca[i][0] = 0;
            miejsca[i][1] = 1;
        }
        for (int i = 1; i <= b; i++)
        {
            for (int j = 1; j <= a; j++)
            {
                cin >> c;
                if (c == 'o')
                {
                    miejsca[0][0] = i;
                    miejsca[0][1] = j;
                    grid[0][i][j] = 0;
                }
                else if (c == 'x')
                {
                    for (int k = 0; k < 11; k++)
                        grid[k][i][j] = -1;
                }
                else if (c == '*')
                {
                    grid[licznik][i][j] = 0;
                    miejsca[licznik][0] = i;
                    miejsca[licznik][1] = j;
                    licznik++;
                }
            }
        }
        for (int i = 0; i < licznik; i++)
        {
            kolejka.push(miejsca[i][0]);
            kolejka.push(miejsca[i][1]);
            while (!kolejka.empty())
            {
                x = kolejka.front();
                kolejka.pop();
                y = kolejka.front();
                kolejka.pop();
                if (grid[i][x - 1][y] != -1 && grid[i][x - 1][y] > grid[i][x][y] + 1)
                {
                    kolejka.push(x - 1);
                    kolejka.push(y);
                    grid[i][x - 1][y] = grid[i][x][y] + 1;
                }
                if (grid[i][x + 1][y] != -1 && grid[i][x + 1][y] > grid[i][x][y] + 1)
                {
                    kolejka.push(x + 1);
                    kolejka.push(y);
                    grid[i][x + 1][y] = grid[i][x][y] + 1;
                }
                if (grid[i][x][y - 1] != -1 && grid[i][x][y - 1] > grid[i][x][y] + 1)
                {
                    kolejka.push(x);
                    kolejka.push(y - 1);
                    grid[i][x][y - 1] = grid[i][x][y] + 1;
                }
                if (grid[i][x][y + 1] != -1 && grid[i][x][y + 1] > grid[i][x][y] + 1)
                {
                    kolejka.push(x);
                    kolejka.push(y + 1);
                    grid[i][x][y + 1] = grid[i][x][y] + 1;
                }
            }
            for (int j = 0; j < licznik; j++)
            {
                if (i != j)
                {
                    odleglosci[i][j] = grid[i][miejsca[j][0]][miejsca[j][1]];
                    if (odleglosci[i][j] == 10000)
                        wynik = -1;
                }
            }
        }
        if (wynik == -1)
            cout << wynik << endl;
        else
        {
            spr(1, 0);
            if (wynik == 1000000)
                wynik = 0;
            cout << wynik << endl;
        }
        cin >> a >> b;
    }
}
