#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

vector<int> heap;
int days;
int receipts, temp;
long long sum;

void add(int element)
{
    heap.push_back(element);
    int position = heap.size() - 1;
    while (position > 0)
    {
        int parent = (position - 1) / 2;
        if (heap[parent] < heap[position])
        {
            swap(heap[parent], heap[position]);
        }
        else
            break;
        position = parent;
    }
}

void showheap()
{
    cout << "HEAP:" << endl;
    for (int i = 0; i < heap.size(); i++)
        cout << heap[i] << " ";
    cout << endl
         << endl;
}

void removeNode(int position)
{
    if (position == heap.size() - 1)
    {
        heap.pop_back();
        return;
    }
    else
    {
        swap(heap[position], heap[heap.size() - 1]);
        heap.pop_back();
    }
    while (position > 0)
    {
        int parent = (position - 1) / 2;
        if (heap[parent] < heap[position])
        {
            swap(heap[parent], heap[position]);
        }
        else
            break;
        position = parent;
    }
    while (position < heap.size() / 2)
    {
        int leftChild = (2 * position) + 1;
        int rightChild = (2 * position) + 2;
        int greaterChild;
        if (rightChild >= heap.size())
        {
            greaterChild = leftChild;
        }
        else
        {
            if (heap[leftChild] > heap[rightChild])
            {
                greaterChild = leftChild;
            }
            else
            {
                greaterChild = rightChild;
            }
        }
        if (heap[greaterChild] > heap[position])
        {
            swap(heap[greaterChild], heap[position]);
        }
        else
            break;
        position = greaterChild;
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> days;
    while (days--)
    {
        cin >> receipts;
        while (receipts--)
        {
            cin >> temp;
            add(temp);
        }
        int mn = 10000000;
        int pos;
        for (int i = heap.size() / 2; i < heap.size(); i++)
        {
            if (heap[i] < mn)
            {
                mn = heap[i];
                pos = i;
            }
        }
        sum -= heap[pos];
        sum += heap[0];
        removeNode(pos);
        removeNode(0);
    }
    cout << sum << endl;
}