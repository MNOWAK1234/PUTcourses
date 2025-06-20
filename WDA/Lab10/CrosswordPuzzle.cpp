#include <iostream>
#include <vector>

using namespace std;

string b, words;
vector<string> crossword;
vector<int> spaces[100];
vector<string> passwords[11];
vector<string> taken;
int cross[10][10];
int iter = 1;
int num;
int spaceiter;
int gapiter[50];
int gapmax[50];
bool iscross(int x, int y)
{
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    if (x > 0 && crossword[x - 1][y] == '-')
        up = true;
    if (x < 9 && crossword[x + 1][y] == '-')
        down = true;
    if (y > 0 && crossword[x][y - 1] == '-')
        left = true;
    if (y < 9 && crossword[x][y + 1] == '-')
        right = true;
    return ((left || right) && (up || down));
}
bool isone(int x, int y)
{
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    if (x > 0 && crossword[x - 1][y] == '-')
        up = true;
    if (x < 9 && crossword[x + 1][y] == '-')
        down = true;
    if (y > 0 && crossword[x][y - 1] == '-')
        left = true;
    if (y < 9 && crossword[x][y + 1] == '-')
        right = true;
    if (crossword[x][y] == '-')
        return !(left || right || up || down);
    else
        return false;
}
void sol()
{
    string a = "";
    for (int i = 0; i < words.size(); i++)
    {
        if (words[i] == ';')
        {
            passwords[a.size()].push_back(a);
            a = "";
        }
        else
            a += words[i];
    }
    passwords[a.size()].push_back(a);
}
void intersect()
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (crossword[i][j] == '-' && iscross(i, j))
            {
                cross[i][j] = iter;
                iter++;
            }
        }
    }
}
void gaps()
{
    int start = -1;
    int stop;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (crossword[i][j] != '+' && start == -1)
            {
                start = j;
            }
            if (crossword[i][j] == '+' && start != -1)
            {
                stop = j - 1;
                if (start != stop)
                {
                    if (passwords[stop - start + 1].size() == 1)
                    {
                        for (int k = start; k <= stop; k++)
                        {
                            crossword[i][k] = passwords[stop - start + 1][0][k - start];
                        }
                    }
                    else
                    {
                        spaces[spaceiter].push_back(stop - start + 1);
                        spaces[spaceiter].push_back(i);
                        spaces[spaceiter].push_back(start);
                        spaces[spaceiter].push_back(i);
                        spaces[spaceiter].push_back(stop);
                        spaceiter++;
                    }
                }
                start = -1;
            }
        }
        if (start != -1)
            stop = 9;
        if (start != -1 && start != stop)
        {
            if (passwords[stop - start + 1].size() == 1)
            {
                for (int k = start; k <= stop; k++)
                {
                    crossword[i][k] = passwords[stop - start + 1][0][k - start];
                }
            }
            else
            {
                spaces[spaceiter].push_back(stop - start + 1);
                spaces[spaceiter].push_back(i);
                spaces[spaceiter].push_back(start);
                spaces[spaceiter].push_back(i);
                spaces[spaceiter].push_back(stop);
                spaceiter++;
            }
        }
        start = -1;
    }
    start = -1;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (crossword[j][i] != '+' && start == -1)
            {
                start = j;
            }
            if (crossword[j][i] == '+' && start != -1)
            {
                stop = j - 1;
                if (start != stop)
                {
                    if (passwords[stop - start + 1].size() == 1)
                    {
                        for (int k = start; k <= stop; k++)
                        {
                            crossword[k][i] = passwords[stop - start + 1][0][k - start];
                        }
                    }
                    else
                    {
                        spaces[spaceiter].push_back(stop - start + 1);
                        spaces[spaceiter].push_back(start);
                        spaces[spaceiter].push_back(i);
                        spaces[spaceiter].push_back(stop);
                        spaces[spaceiter].push_back(i);
                        spaceiter++;
                    }
                }
                start = -1;
            }
        }
        if (start != -1)
            stop = 9;
        if (start != -1 && start != stop)
        {
            if (passwords[stop - start + 1].size() == 1)
            {
                for (int k = start; k <= stop; k++)
                {
                    crossword[k][i] = passwords[stop - start + 1][0][k - start];
                }
            }
            else
            {
                spaces[spaceiter].push_back(stop - start + 1);
                spaces[spaceiter].push_back(start);
                spaces[spaceiter].push_back(i);
                spaces[spaceiter].push_back(stop);
                spaces[spaceiter].push_back(i);
                spaceiter++;
            }
        }
        start = -1;
    }
}
void ones()
{
    int cnt = 0;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (isone(i, j))
            {
                crossword[i][j] = passwords[1][cnt][0];
                cnt++;
            }
        }
    }
}
void reset(int i)
{
    for (int j = i; j < spaceiter; j++)
        gapiter[j] = 0;
}
void check()
{
    for (int i = 0; i < 10; i++)
    {
        cout << i << ": ";
        for (int j = 0; j < passwords[i].size(); j++)
        {
            cout << passwords[i][j] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < spaceiter; i++)
    {
        cout << spaces[i][0] << " " << spaces[i][1] << " " << spaces[i][2] << " " << spaces[i][3] << " " << spaces[i][4] << endl;
    }
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            cout << crossword[i][j];
        }
        cout << endl;
    }
    cout << spaceiter << endl;
    cout << endl;
}
void show()
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            cout << crossword[i][j];
        }
        cout << endl;
    }
}

void solve()
{
    vector<string> grid = crossword;
    vector<string> comb = taken;
    while (gapiter[num] < gapmax[num])
    {
        if (num < spaceiter)
        {
            while (taken[spaces[num][0]][gapiter[num]] != '0' && gapiter[num] < gapmax[num])
            {
                gapiter[num]++;
            }
            if (gapiter[num] < gapmax[num])
            {
                taken[spaces[num][0]][gapiter[num]] = '1';
                bool possible = true;
                if (spaces[num][1] == spaces[num][3])
                {
                    for (int k = 0; k < spaces[num][0]; k++)
                    {
                        if (crossword[spaces[num][1]][spaces[num][2] + k] == '-' || crossword[spaces[num][1]][spaces[num][2] + k] == passwords[spaces[num][0]][gapiter[num]][k])
                            crossword[spaces[num][1]][spaces[num][2] + k] = passwords[spaces[num][0]][gapiter[num]][k];
                        else
                        {
                            possible = false;
                            crossword = grid;
                            taken = comb;
                            break;
                        }
                    }
                }
                else
                {
                    for (int k = 0; k < spaces[num][0]; k++)
                    {
                        if (crossword[spaces[num][1] + k][spaces[num][2]] == '-' || crossword[spaces[num][1] + k][spaces[num][2]] == passwords[spaces[num][0]][gapiter[num]][k])
                            crossword[spaces[num][1] + k][spaces[num][2]] = passwords[spaces[num][0]][gapiter[num]][k];
                        else
                        {
                            possible = false;
                            crossword = grid;
                            taken = comb;
                            break;
                        }
                    }
                }
                if (possible == true && num == spaceiter - 1)
                {
                    show();
                    return;
                }
                num++;
                if (possible == true)
                {
                    solve();
                }
                num--;
            }
        }
        gapiter[num]++;
        crossword = grid;
        taken = comb;
    }
    reset(num);
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    for (int i = 0; i < 10; i++)
    {
        cin >> b;
        crossword.push_back(b);
    }
    cin >> words;
    sol();
    ones();
    intersect();
    gaps();
    num = 0;
    for (int i = 0; i < spaceiter; i++)
    {
        gapmax[i] = passwords[spaces[i][0]].size();
    }
    for (int i = 0; i < 11; i++)
    {
        taken.push_back("");
        for (int j = 0; j < passwords[i].size(); j++)
        {
            taken[i] += "0";
        }
    }
    if (spaceiter == 0)
        show();
    else
        solve();
}
