#include <iostream>

using namespace std;

struct pkt
{
    short int x;
    short int y;
    short int z;
};
char porownaj(pkt a, pkt b)
{
    if (a.x < b.x)
        return 'm';
    else if (a.x > b.x)
        return 'w';
    else
    {
        if (a.y < b.y)
            return 'm';
        else if (a.y > b.y)
            return 'w';
        else
        {
            if (a.z < b.z)
                return 'm';
            else if (a.z > b.z)
                return 'w';
            else
                return 'r';
        }
    }
}
int t;
pkt punkty[1000];
void quicksort(pkt tab[], int p, int k)
{
    pkt s = tab[(p + k) / 2];
    short int i = p;
    short int j = k;
    while (i <= j)
    {
        while (porownaj(tab[i], s) == 'm')
            i++;
        while (porownaj(tab[j], s) == 'w')
            j--;
        if (i <= j)
        {
            pkt pomoc = tab[i];
            tab[i] = tab[j];
            tab[j] = pomoc;
            i++;
            j--;
        }
    }
    if (p < j)
        quicksort(tab, p, j);
    if (i < k)
        quicksort(tab, i, k);
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cin >> t;
    for (int i = 0; i < t; i++)
    {
        cin >> punkty[i].x >> punkty[i].y >> punkty[i].z;
    }
    quicksort(punkty, 0, t - 1);
    for (int m = 0; m < t; m++)
        cout << punkty[m].x << " " << punkty[m].y << " " << punkty[m].z << "\n";
}