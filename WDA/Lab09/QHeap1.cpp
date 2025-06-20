#include <iostream>
#include <vector>
#include <queue>

using namespace std;

vector<int> heap;
queue<int> seek;
int parent, child;
int mn;
bool finished;
int position;

void swapnodes(int a, int b)
{
    int help = heap[a];
    heap[a] = heap[b];
    heap[b] = help;
}
void showheap()
{
    for (int i = 0; i < heap.size(); i++)
        cout << heap[i] << " ";
    cout << endl
         << endl;
}
void add(int where)
{
    finished = false;
    while (!finished && where > 0)
    {
        parent = (where - 1) / 2;
        if (heap[parent] > heap[where])
        {
            swapnodes(parent, where);
        }
        else
        {
            finished = true;
            position = where;
        }
        where = parent;
    }
    if (!finished)
    {
        position = 0;
    }
}
void down()
{
    finished = false;
    int where = position;
    while (!finished && where < heap.size() / 2)
    {
        child = (where * 2) + 1;
        if (heap[child] < heap[where])
        {
            if (child + 1 < heap.size())
            {
                if (heap[child + 1] < heap[child])
                {
                    swapnodes(child + 1, where);
                    where = child + 1;
                }
                else
                {
                    swapnodes(child, where);
                    where = child;
                }
            }
            else
            {
                swapnodes(child, where);
                where = child;
            }
        }
        else if (child + 1 < heap.size())
        {
            if (heap[child + 1] < heap[where])
            {
                swapnodes(child + 1, where);
                where = child + 1;
            }
            else
                finished = true;
        }
        else
            finished = true;
    }
}
void findnremove(int x)
{
    finished = false;
    while (!seek.empty())
        seek.pop();
    int help;
    seek.push(0);
    while (!finished && !seek.empty())
    {
        help = seek.front();
        seek.pop();
        if (heap[help] == x)
        {
            position = help;
            swapnodes(position, heap.size() - 1);
            heap.pop_back();
            if (position != heap.size())
                add(position);
            down();
            finished = true;
        }
        if (2 * help + 1 < heap.size())
        {
            if (heap[2 * help + 1] <= x)
            {
                seek.push(2 * help + 1);
            }
        }
        if (2 * help + 2 < heap.size())
        {
            if (heap[2 * help + 2] <= x)
            {
                seek.push(2 * help + 2);
            }
        }
    }
}

int q;
int a, b;

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> q;
    while (q--)
    {
        cin >> a;
        if (a == 1)
        {
            cin >> b;
            heap.push_back(b);
            add(heap.size() - 1);
        }
        else if (a == 2)
        {
            cin >> b;
            findnremove(b);
        }
        else
        {
            cout << heap[0] << "\n";
        }
    }
}
