#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

struct node
{
    long long mx, suma, lewa, prawa;
};
vector<node> segmentTree;

node join(node leftObject, node rightObject)
{
    node curr;
    curr.lewa = max(leftObject.lewa, leftObject.suma + rightObject.lewa);
    curr.prawa = max(rightObject.prawa, rightObject.suma + leftObject.prawa);
    curr.suma = leftObject.suma + rightObject.suma;
    curr.mx = max(leftObject.mx, rightObject.mx);
    curr.mx = max(curr.mx, leftObject.prawa + rightObject.lewa);
    return curr;
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
        node singular;
        singular.lewa = elements[left];
        singular.prawa = elements[left];
        singular.mx = elements[left];
        singular.suma = elements[left];
        segmentTree[position] = singular;
    }
}

void build(int number_of_elements, vector<int> &elements)
{
    int size = 2 * number_of_elements - 1;
    node dummy;
    for (int i = 0; i < size; i++)
    {
        segmentTree.push_back(dummy);
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
        node singular;
        singular.lewa = newValue;
        singular.prawa = newValue;
        singular.mx = newValue;
        singular.suma = newValue;
        segmentTree[position] = singular;
    }
}

node query(int left, int right, int leftQuery, int rightQuery, int position)
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
        node LeftObject, RightObject;
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
    int lewy, prawy;
    int a;
    cin >> ile;
    for (int i = 0; i < ile; i++)
    {
        cin >> a;
        elements.push_back(a);
    }
    build(ile, elements);
    cin >> pytania;
    while (pytania--)
    {
        cin >> lewy >> prawy;
        lewy--;
        prawy--;
        node ans = query(0, ile - 1, lewy, prawy, 0);
        cout << ans.mx << endl;
    }
}