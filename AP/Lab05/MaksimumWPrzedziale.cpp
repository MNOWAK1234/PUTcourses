#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

vector<int> segmentTree;

int join(int leftObject, int rightObject)
{
    return max(leftObject, rightObject);
}

void populate(int left, int right, int position, vector<int> &elements)
{
    if (left < right)
    {
        int MidPoint = (left + right) / 2;
        populate(left, MidPoint, 2 * position + 1, elements);
        populate(MidPoint + 1, right, 2 * position + 2, elements);
        segmentTree[position] = join(segmentTree[2 * position + 1], segmentTree[2 * position + 2]);
    }
    else
    {
        segmentTree[position] = elements[left];
    }
}

void build(int number_of_elements, vector<int> &elements)
{
    int layers = 0;
    int number = number_of_elements;
    while (number > 0)
    {
        layers++;
        number /= 2;
    }
    layers++;
    int size = pow(2, layers) - 1;
    for (int i = 0; i < size; i++)
    {
        segmentTree.push_back(0);
    }
    populate(0, number_of_elements - 1, 0, elements);
}

void change(int left, int right, int position, int indexOfChange, int newValue)
{
    if (left < right)
    {
        int MidPoint = (left + right) / 2;
        if (indexOfChange <= MidPoint)
        {
            change(left, MidPoint, 2 * position + 1, indexOfChange, newValue);
        }
        else
        {
            change(MidPoint + 1, right, 2 * position + 2, indexOfChange, newValue);
        }
        segmentTree[position] = join(segmentTree[2 * position + 1], segmentTree[2 * position + 2]);
    }
    else
    {
        segmentTree[position] = newValue;
    }
}

int query(int left, int right, int leftQuery, int rightQuery, int position)
{
    if (leftQuery <= left && rightQuery >= right)
    {
        return segmentTree[position];
    }
    else
    {
        int MidPoint = (left + right) / 2;
        bool goLeft = false;
        bool goRight = false;
        int LeftObject, RightObject;
        if (leftQuery <= MidPoint)
        {
            goLeft = true;
        }
        if (MidPoint + 1 <= rightQuery)
        {
            goRight = true;
        }
        if (goLeft == false)
        {
            RightObject = query(MidPoint + 1, right, leftQuery, rightQuery, 2 * position + 2);
            return RightObject;
        }
        else if (goRight == false)
        {
            LeftObject = query(left, MidPoint, leftQuery, rightQuery, 2 * position + 1);
            return LeftObject;
        }
        else
        {
            LeftObject = query(left, MidPoint, leftQuery, rightQuery, 2 * position + 1);
            RightObject = query(MidPoint + 1, right, leftQuery, rightQuery, 2 * position + 2);
            return join(LeftObject, RightObject);
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    vector<int> elements;
    int ile, pytania;
    int rodzaj;
    int co, naco;
    int lewy, prawy;
    cin >> ile >> pytania;
    for (int i = 0; i < ile; i++)
    {
        elements.push_back(0);
    }
    build(ile, elements);
    while (pytania--)
    {
        cin >> rodzaj;
        if (rodzaj == 1)
        {
            cin >> co >> naco;
            change(0, ile - 1, 0, co, naco);
        }
        else
        {
            cin >> lewy >> prawy;
            cout << query(0, ile - 1, lewy, prawy, 0) << endl;
        }
    }
}